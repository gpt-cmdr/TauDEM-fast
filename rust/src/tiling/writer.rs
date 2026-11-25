// SPDX-License-Identifier: GPL-3.0-only
//! GeoTIFF output written by a background thread.
//!
//! Compute threads compress output blocks with terrio's [`TileEncoder`]
//! (cheap to clone, `Send` and `Sync`) and send them through a bounded
//! channel to one thread that owns the [`GeoTiffWriter`] and appends them
//! with [`GeoTiffWriter::write_compressed_tile`]. A compute thread waits on
//! I/O only when the channel is full. Blocks arrive in any order, so the
//! pixels are identical across thread counts while the order of the blocks
//! in the file is not.
//!
//! A compute tile need not match the file's blocks: [`TileSink::submit_region`]
//! takes any rectangle and holds the blocks it covers only partly until the
//! rectangles of neighboring tiles complete them.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use terrio::{
    CompressedTile, GeoTiffWriter, RasterDataset, RasterElement, RasterWriteOptions, TileEncoder,
    TileGrid,
};

use super::spec::Rect;
use super::{Error, Result};

/// Counters of an [`AsyncTileWriter`], from [`AsyncTileWriter::finish`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WriterStats {
    /// Blocks written, including those `SPARSE_OK` left out.
    pub blocks: u64,
    /// Blocks `SPARSE_OK` left out of the file (all nodata).
    pub sparse_blocks: u64,
    pub compressed_bytes: u64,
    /// Seconds compute threads spent compressing, summed over threads.
    pub compress_seconds: f64,
    /// Seconds compute threads waited for room in the channel.
    pub blocked_seconds: f64,
    /// Seconds the writer thread spent appending blocks and finishing the file.
    pub write_seconds: f64,
}

impl WriterStats {
    /// Records the counters for `--profile-json` under `write.<name>.*`.
    pub fn record_profile(&self, name: &str) {
        use crate::profile::count;
        // Profile counter names are static; only the known products are recorded.
        let names: [&'static str; 5] = match name {
            "fel" => [
                "write.fel.blocks",
                "write.fel.sparse_blocks",
                "write.fel.compress_us",
                "write.fel.blocked_us",
                "write.fel.write_us",
            ],
            "p" => [
                "write.p.blocks",
                "write.p.sparse_blocks",
                "write.p.compress_us",
                "write.p.blocked_us",
                "write.p.write_us",
            ],
            "sd8" => [
                "write.sd8.blocks",
                "write.sd8.sparse_blocks",
                "write.sd8.compress_us",
                "write.sd8.blocked_us",
                "write.sd8.write_us",
            ],
            "ad8" => [
                "write.ad8.blocks",
                "write.ad8.sparse_blocks",
                "write.ad8.compress_us",
                "write.ad8.blocked_us",
                "write.ad8.write_us",
            ],
            _ => [
                "write.other.blocks",
                "write.other.sparse_blocks",
                "write.other.compress_us",
                "write.other.blocked_us",
                "write.other.write_us",
            ],
        };
        count(names[0], self.blocks);
        count(names[1], self.sparse_blocks);
        count(names[2], (self.compress_seconds * 1e6) as u64);
        count(names[3], (self.blocked_seconds * 1e6) as u64);
        count(names[4], (self.write_seconds * 1e6) as u64);
    }
}

enum Message {
    Block(CompressedTile),
    Finish,
}

/// A block of the file partly given: its pixels and how many are set.
struct Partial<T> {
    data: Vec<T>,
    covered: usize,
}

struct Shared<T> {
    encoder: TileEncoder,
    grid: TileGrid,
    raster: (usize, usize),
    partial: Mutex<HashMap<(usize, usize), Partial<T>>>,
    failure: Mutex<Option<String>>,
    blocks: AtomicU64,
    sparse: AtomicU64,
    bytes: AtomicU64,
    compress_ns: AtomicU64,
    blocked_ns: AtomicU64,
}

/// A cloneable handle compute threads submit blocks through.
pub struct TileSink<T> {
    shared: Arc<Shared<T>>,
    sender: SyncSender<Message>,
}

impl<T> Clone for TileSink<T> {
    fn clone(&self) -> Self {
        Self {
            shared: self.shared.clone(),
            sender: self.sender.clone(),
        }
    }
}

/// A tiled GeoTIFF written in the background. Create it, submit every
/// block (directly or through [`TileSink`]s on other threads), then call
/// [`Self::finish`]. Dropping it without finishing abandons the file.
pub struct AsyncTileWriter<T> {
    sink: TileSink<T>,
    thread: Option<JoinHandle<std::result::Result<f64, String>>>,
}

impl<T: RasterElement + Default> AsyncTileWriter<T> {
    /// Creates a tiled GeoTIFF (`options.tile_size` must be set) with a
    /// channel of `queue_depth` compressed blocks (at least 1).
    pub fn create(
        path: &str,
        header: &RasterDataset,
        options: &RasterWriteOptions,
        queue_depth: usize,
    ) -> Result<Self> {
        Self::from_writer(GeoTiffWriter::create(path, header, options)?, queue_depth)
    }

    /// Starts writing an already created tiled GeoTIFF in the background.
    pub fn from_writer(writer: GeoTiffWriter, queue_depth: usize) -> Result<Self> {
        let grid = writer
            .tile_grid()
            .ok_or_else(|| Error::Invalid("the background writer needs a tiled GeoTIFF".into()))?;
        let encoder = writer.tile_encoder()?;
        let (across, down) = grid.tiles();
        let (_, _, last_w, last_h) = grid.window(across - 1, down - 1).expect("last block");
        let (tw, th) = grid.tile_size();
        let raster = ((across - 1) * tw + last_w, (down - 1) * th + last_h);
        let (sender, receiver) = sync_channel(queue_depth.max(1));
        let shared = Arc::new(Shared {
            encoder,
            grid,
            raster,
            partial: Mutex::new(HashMap::new()),
            failure: Mutex::new(None),
            blocks: AtomicU64::new(0),
            sparse: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            compress_ns: AtomicU64::new(0),
            blocked_ns: AtomicU64::new(0),
        });
        let thread_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("taudem-tile-writer".into())
            .spawn(move || run_writer(writer, receiver, &thread_shared))
            .map_err(Error::Io)?;
        Ok(Self {
            sink: TileSink { shared, sender },
            thread: Some(thread),
        })
    }

    /// A handle for another thread.
    pub fn sink(&self) -> TileSink<T> {
        self.sink.clone()
    }

    pub fn grid(&self) -> TileGrid {
        self.sink.shared.grid
    }

    pub fn encoder(&self) -> &TileEncoder {
        &self.sink.shared.encoder
    }

    /// [`TileSink::submit_block`].
    pub fn submit_block(&self, col: usize, row: usize, data: &[T]) -> Result<()> {
        self.sink.submit_block(col, row, data)
    }

    /// [`TileSink::submit_region`].
    pub fn submit_region(&self, rect: Rect, data: &[T]) -> Result<()> {
        self.sink.submit_region(rect, data)
    }

    /// Waits for every block to be written and finishes the file. Errors
    /// when a block was only partly given or the writer failed.
    pub fn finish(mut self) -> Result<WriterStats> {
        let shared = self.sink.shared.clone();
        let partial = shared
            .partial
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len();
        let thread = self.thread.take().expect("finish runs once");
        if partial > 0 {
            drop(self);
            let _ = thread.join();
            return Err(Error::Invalid(format!(
                "{partial} output blocks were only partly written"
            )));
        }
        let sent = self.sink.sender.send(Message::Finish).is_ok();
        drop(self);
        let write_seconds = thread
            .join()
            .map_err(|_| Error::WriterClosed("the writer thread panicked".into()))?
            .map_err(Error::WriterClosed)?;
        if !sent {
            return Err(Error::WriterClosed(shared.failure_message()));
        }
        let secs = |ns: &AtomicU64| ns.load(Relaxed) as f64 * 1e-9;
        Ok(WriterStats {
            blocks: shared.blocks.load(Relaxed),
            sparse_blocks: shared.sparse.load(Relaxed),
            compressed_bytes: shared.bytes.load(Relaxed),
            compress_seconds: secs(&shared.compress_ns),
            blocked_seconds: secs(&shared.blocked_ns),
            write_seconds,
        })
    }
}

impl<T> Drop for AsyncTileWriter<T> {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            // Closing the channel without `Finish` abandons the file; the
            // thread ends once the sinks still held elsewhere drop.
            let (sender, _) = sync_channel(1);
            drop(std::mem::replace(&mut self.sink.sender, sender));
            drop(thread);
        }
    }
}

fn run_writer<T>(
    mut writer: GeoTiffWriter,
    receiver: Receiver<Message>,
    shared: &Shared<T>,
) -> std::result::Result<f64, String> {
    let mut seconds = 0.0;
    let fail = |message: String| {
        *shared.failure.lock().unwrap_or_else(|p| p.into_inner()) = Some(message.clone());
        Err(message)
    };
    for message in receiver {
        let start = Instant::now();
        match message {
            Message::Block(tile) => {
                if let Err(error) = writer.write_compressed_tile(tile) {
                    // Returning drops the receiver, so submitters see the failure.
                    return fail(error.to_string());
                }
            }
            Message::Finish => {
                if let Err(error) = writer.finish() {
                    return fail(error.to_string());
                }
                return Ok(seconds + start.elapsed().as_secs_f64());
            }
        }
        seconds += start.elapsed().as_secs_f64();
    }
    fail("the file was abandoned before it was finished".into())
}

impl<T> Shared<T> {
    fn failure_message(&self) -> String {
        self.failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .unwrap_or_else(|| "the writer thread stopped".into())
    }
}

impl<T: RasterElement + Default> TileSink<T> {
    pub fn grid(&self) -> TileGrid {
        self.shared.grid
    }

    /// Compresses block `(col, row)` on this thread and queues it, waiting
    /// while the channel is full. `data` holds the block's window
    /// ([`TileGrid::window`]), unpadded at the raster's edges.
    pub fn submit_block(&self, col: usize, row: usize, data: &[T]) -> Result<()> {
        let start = Instant::now();
        let tile = self.shared.encoder.compress(col, row, data)?;
        self.shared
            .compress_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
        self.shared.blocks.fetch_add(1, Relaxed);
        if tile.is_empty() {
            self.shared.sparse.fetch_add(1, Relaxed);
        }
        self.shared.bytes.fetch_add(tile.len() as u64, Relaxed);
        let closed = || Error::WriterClosed(self.shared.failure_message());
        match self.sender.try_send(Message::Block(tile)) {
            Ok(()) => Ok(()),
            Err(TrySendError::Disconnected(_)) => Err(closed()),
            Err(TrySendError::Full(message)) => {
                let start = Instant::now();
                let sent = self.sender.send(message);
                self.shared
                    .blocked_ns
                    .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
                sent.map_err(|_| closed())
            }
        }
    }

    /// Submits the raster pixels of `rect` (`data`, row-major). Blocks that
    /// `rect` covers whole are compressed and queued at once; the others are
    /// held until the rest of their pixels arrive. Each pixel must be given once.
    pub fn submit_region(&self, rect: Rect, data: &[T]) -> Result<()> {
        let (cols, rows) = self.shared.raster;
        if rect.cells() == 0
            || rect.right() > cols
            || rect.bottom() > rows
            || data.len() != rect.cells()
        {
            return Err(Error::Invalid(format!(
                "region {rect:?} with {} values does not fit the {cols} x {rows} raster",
                data.len()
            )));
        }
        let (tw, th) = self.shared.grid.tile_size();
        let mut block = Vec::new();
        for row in rect.y / th..rect.bottom().div_ceil(th) {
            for col in rect.x / tw..rect.right().div_ceil(tw) {
                let (bx, by, bw, bh) = self
                    .shared
                    .grid
                    .window(col, row)
                    .expect("inside the raster");
                let window = Rect::new(bx, by, bw, bh);
                let x0 = bx.max(rect.x);
                let y0 = by.max(rect.y);
                let x1 = window.right().min(rect.right());
                let y1 = window.bottom().min(rect.bottom());
                let copy = |dst: &mut [T]| {
                    for y in y0..y1 {
                        let src = (y - rect.y) * rect.width + (x0 - rect.x);
                        let to = (y - by) * bw + (x0 - bx);
                        dst[to..to + (x1 - x0)].copy_from_slice(&data[src..src + (x1 - x0)]);
                    }
                };
                if (x0, y0, x1, y1) == (bx, by, window.right(), window.bottom()) {
                    block.clear();
                    block.resize(bw * bh, T::default());
                    copy(&mut block);
                    self.submit_block(col, row, &block)?;
                    continue;
                }
                let complete = {
                    let mut partial = self
                        .shared
                        .partial
                        .lock()
                        .unwrap_or_else(|p| p.into_inner());
                    let entry = partial.entry((col, row)).or_insert_with(|| Partial {
                        data: vec![T::default(); bw * bh],
                        covered: 0,
                    });
                    copy(&mut entry.data);
                    entry.covered += (x1 - x0) * (y1 - y0);
                    if entry.covered > bw * bh {
                        return Err(Error::Invalid(format!(
                            "pixels of block ({col}, {row}) were given twice"
                        )));
                    }
                    (entry.covered == bw * bh)
                        .then(|| partial.remove(&(col, row)).expect("present").data)
                };
                if let Some(pixels) = complete {
                    self.submit_block(col, row, &pixels)?;
                }
            }
        }
        Ok(())
    }
}
