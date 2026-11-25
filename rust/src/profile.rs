// SPDX-License-Identifier: GPL-3.0-only
//! Phase timings, counters and heap accounting for `--profile-json`.
//!
//! Spans nest on the calling thread and record wall time together with the
//! heap in use at entry and exit and the heap high-water mark inside the
//! span. Heap figures are nonzero only when the binary installs
//! [`CountingAlloc`] as its global allocator. Recording is always on; the
//! kernels open a few dozen spans per run, so the cost is negligible.

use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Instant;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static STATE: Mutex<State> = Mutex::new(State::new());

/// The system allocator, counting bytes in use and their high-water mark.
pub struct CountingAlloc;

// SAFETY: Every call forwards to `System` with the caller's layout; the
// counters do not affect allocation.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            grow(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        CURRENT.fetch_sub(layout.size(), Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            if new_size >= layout.size() {
                grow(new_size - layout.size());
            } else {
                CURRENT.fetch_sub(layout.size() - new_size, Relaxed);
            }
        }
        new
    }
}

#[inline]
fn grow(bytes: usize) {
    let now = CURRENT.fetch_add(bytes, Relaxed) + bytes;
    if now > PEAK.load(Relaxed) {
        PEAK.fetch_max(now, Relaxed);
    }
}

/// Bytes currently allocated through [`CountingAlloc`].
pub fn heap_current() -> usize {
    CURRENT.load(Relaxed)
}

/// Process-lifetime heap high-water mark through [`CountingAlloc`].
pub fn heap_peak() -> usize {
    let state = lock();
    state
        .saved_peaks
        .iter()
        .copied()
        .fold(PEAK.load(Relaxed), usize::max)
}

struct State {
    origin: Option<Instant>,
    spans: Vec<SpanRecord>,
    open: Vec<usize>,
    saved_peaks: Vec<usize>,
    counters: Vec<(&'static str, u64)>,
    buffers: Vec<BufferRecord>,
}

impl State {
    const fn new() -> Self {
        Self {
            origin: None,
            spans: Vec::new(),
            open: Vec::new(),
            saved_peaks: Vec::new(),
            counters: Vec::new(),
            buffers: Vec::new(),
        }
    }

    fn origin(&mut self) -> Instant {
        *self.origin.get_or_insert_with(Instant::now)
    }
}

fn lock() -> std::sync::MutexGuard<'static, State> {
    STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Debug)]
pub struct SpanRecord {
    pub name: String,
    pub depth: usize,
    pub start_s: f64,
    pub seconds: f64,
    pub heap_start: usize,
    pub heap_end: usize,
    pub heap_peak: usize,
}

#[derive(Clone, Debug)]
pub struct BufferRecord {
    pub name: &'static str,
    pub span: String,
    pub bytes: usize,
}

/// Starts the profile clock. Spans opened before this call start it implicitly.
pub fn start() {
    lock().origin();
}

/// A timed phase; it ends when dropped.
#[must_use = "a span ends when it is dropped"]
pub struct Span {
    index: usize,
    start: Instant,
}

/// Opens a nested phase.
pub fn span(name: impl Into<String>) -> Span {
    let mut state = lock();
    let origin = state.origin();
    let start = Instant::now();
    let heap = CURRENT.load(Relaxed);
    // The enclosing peak is restored on exit, so this span sees only its own.
    let saved = PEAK.swap(heap, Relaxed);
    state.saved_peaks.push(saved);
    let index = state.spans.len();
    let depth = state.open.len();
    state.spans.push(SpanRecord {
        name: name.into(),
        depth,
        start_s: (start - origin).as_secs_f64(),
        seconds: 0.0,
        heap_start: heap,
        heap_end: heap,
        heap_peak: heap,
    });
    state.open.push(index);
    Span { index, start }
}

impl Span {
    /// Seconds since the span opened.
    pub fn elapsed(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let seconds = self.start.elapsed().as_secs_f64();
        let mut state = lock();
        let peak = PEAK.load(Relaxed);
        let saved = state.saved_peaks.pop().unwrap_or(0);
        PEAK.fetch_max(saved, Relaxed);
        if let Some(position) = state.open.iter().rposition(|&i| i == self.index) {
            state.open.truncate(position);
        }
        let record = &mut state.spans[self.index];
        record.seconds = seconds;
        record.heap_end = CURRENT.load(Relaxed);
        record.heap_peak = peak.max(record.heap_end);
    }
}

/// Records a count. Repeated names accumulate.
pub fn count(name: &'static str, value: u64) {
    let mut state = lock();
    if let Some(entry) = state.counters.iter_mut().find(|(n, _)| *n == name) {
        entry.1 += value;
    } else {
        state.counters.push((name, value));
    }
}

/// Records the size of a significant working buffer in the current span.
pub fn buffer(name: &'static str, bytes: usize) {
    let mut state = lock();
    let span = state
        .open
        .last()
        .map(|&i| state.spans[i].name.clone())
        .unwrap_or_default();
    state.buffers.push(BufferRecord { name, span, bytes });
}

/// Peak working set and peak committed bytes of this process, when available.
pub fn process_peaks() -> (Option<u64>, Option<u64>) {
    os::peaks()
}

#[cfg(windows)]
mod os {
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut Counters, cb: u32) -> i32;
    }

    pub fn peaks() -> (Option<u64>, Option<u64>) {
        let mut counters = Counters {
            cb: size_of::<Counters>() as u32,
            ..Default::default()
        };
        // SAFETY: The pseudo-handle is always valid, and the structure is
        // the documented PROCESS_MEMORY_COUNTERS layout with its size set.
        let ok = unsafe {
            K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) != 0
        };
        if ok {
            (
                Some(counters.peak_working_set_size as u64),
                Some(counters.peak_pagefile_usage as u64),
            )
        } else {
            (None, None)
        }
    }
}

#[cfg(not(windows))]
mod os {
    pub fn peaks() -> (Option<u64>, Option<u64>) {
        let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
            return (None, None);
        };
        let field = |key: &str| {
            status.lines().find_map(|line| {
                let rest = line.strip_prefix(key)?;
                let kib: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
                Some(kib * 1024)
            })
        };
        (field("VmHWM:"), field("VmPeak:"))
    }
}

/// Everything recorded so far.
pub struct Report {
    pub spans: Vec<SpanRecord>,
    pub counters: Vec<(&'static str, u64)>,
    pub buffers: Vec<BufferRecord>,
    pub heap_peak: usize,
}

pub fn report() -> Report {
    let heap_peak = heap_peak();
    let state = lock();
    Report {
        spans: state.spans.clone(),
        counters: state.counters.clone(),
        buffers: state.buffers.clone(),
        heap_peak,
    }
}

impl Report {
    /// Renders the profile as JSON. `context` holds preformatted members
    /// (`"key": value` pairs) placed before the recorded sections.
    pub fn to_json(&self, context: &[(&str, String)], cells: usize) -> String {
        let per_cell = |bytes: usize| bytes as f64 / cells.max(1) as f64;
        let mut out = String::from("{\n");
        for (key, value) in context {
            let _ = writeln!(out, "  {}: {value},", quote(key));
        }
        out.push_str("  \"spans\": [\n");
        for (n, s) in self.spans.iter().enumerate() {
            let _ = write!(
                out,
                "    {{\"name\": {}, \"depth\": {}, \"start_s\": {:.6}, \"seconds\": {:.6}, \
                 \"heap_start\": {}, \"heap_end\": {}, \"heap_peak\": {}, \
                 \"heap_peak_bytes_per_cell\": {:.3}}}",
                quote(&s.name),
                s.depth,
                s.start_s,
                s.seconds,
                s.heap_start,
                s.heap_end,
                s.heap_peak,
                per_cell(s.heap_peak)
            );
            out.push_str(if n + 1 < self.spans.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        out.push_str("  ],\n  \"counters\": {");
        for (n, (name, value)) in self.counters.iter().enumerate() {
            let _ = write!(
                out,
                "{}\n    {}: {value}",
                if n == 0 { "" } else { "," },
                quote(name)
            );
        }
        out.push_str(if self.counters.is_empty() {
            "},\n"
        } else {
            "\n  },\n"
        });
        out.push_str("  \"buffers\": [\n");
        for (n, b) in self.buffers.iter().enumerate() {
            let _ = write!(
                out,
                "    {{\"name\": {}, \"span\": {}, \"bytes\": {}, \"bytes_per_cell\": {:.3}}}",
                quote(b.name),
                quote(&b.span),
                b.bytes,
                per_cell(b.bytes)
            );
            out.push_str(if n + 1 < self.buffers.len() {
                ",\n"
            } else {
                "\n"
            });
        }
        let (wset, commit) = process_peaks();
        let opt = |v: Option<u64>| v.map_or("null".to_string(), |v| v.to_string());
        let opt_cell = |v: Option<u64>| {
            v.map_or("null".to_string(), |v| {
                format!("{:.3}", per_cell(v as usize))
            })
        };
        let _ = write!(
            out,
            "  ],\n  \"memory\": {{\"heap_peak_bytes\": {}, \"heap_peak_bytes_per_cell\": {:.3}, \
             \"process_peak_rss_bytes\": {}, \"process_peak_rss_bytes_per_cell\": {}, \
             \"process_peak_commit_bytes\": {}}}\n}}\n",
            self.heap_peak,
            per_cell(self.heap_peak),
            opt(wset),
            opt_cell(wset),
            opt(commit)
        );
        out
    }
}

/// A JSON string literal.
pub fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
