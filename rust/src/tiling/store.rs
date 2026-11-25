// SPDX-License-Identifier: GPL-3.0-only
//! A three-tier tile store under one byte budget.
//!
//! - Tier 0, [`Tier::Ram`]: tiles as they were given, in RAM.
//! - Tier 1, [`Tier::Compressed`]: tiles compressed with a prefilter and
//!   LZ4 ([`super::codec`]), in RAM.
//! - Tier 2, [`Tier::Spilled`]: compressed tiles in a scratch file, read and
//!   written with positioned I/O (never a mutable memory map, whose dirty
//!   pages Windows counts in the working set).
//!
//! The budget bounds the bytes held in RAM by tiers 0 and 1 together. When
//! a store goes over it, the least recently used unpinned tile of tier 0 is
//! compressed; when tier 0 has no such tile, the least recently used
//! compressed tile is spilled. Without a scratch directory nothing spills
//! and the store goes over its budget, counting by how much
//! ([`TileStats::peak_over_budget_bytes`]). Compression and I/O run on the
//! calling thread outside the store's lock, so threads evict in parallel.
//!
//! One [`TileCache`] holds the budget, the eviction order and the scratch
//! file; each [`TileStore<T>`] is a typed layer of it (`fel`, `dir`, ...),
//! so the layers of a pipeline share one budget.

use std::any::Any;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use super::codec::{
    BlobKey, Prefilter, TileElement, blob_elements, decode_with_key, encode_with_key,
};
use super::spec::TileId;
use super::{Error, Result};

/// Where a tile is held.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Tier {
    Ram,
    Compressed,
    Spilled,
}

/// The budget and scratch directory of a [`TileCache`].
#[derive(Clone, Debug)]
pub struct CacheConfig {
    /// The most bytes the cache keeps in RAM (tiers 0 and 1).
    pub budget_bytes: u64,
    /// Where the spill file is created, on first use; `None` disables spilling.
    pub scratch_dir: Option<PathBuf>,
}

impl CacheConfig {
    pub fn new(budget_bytes: u64, scratch_dir: Option<PathBuf>) -> Self {
        Self {
            budget_bytes,
            scratch_dir,
        }
    }
}

/// How a [`TileStore`] keeps its tiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StoreOptions {
    /// The prefilter of compressed tiles; `None` uses the element type's
    /// [`TileElement::DEFAULT_PREFILTER`].
    pub prefilter: Option<Prefilter>,
    /// The tier a tile is put in: [`Tier::Ram`] keeps it uncompressed until
    /// it is evicted, [`Tier::Compressed`] compresses it at once (for tiles
    /// read much later), and [`Tier::Spilled`] writes it to the scratch file.
    pub initial: Tier,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            prefilter: None,
            initial: Tier::Ram,
        }
    }
}

/// Counters of a [`TileCache`], from [`TileCache::stats`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TileStats {
    pub budget_bytes: u64,
    /// Bytes of uncompressed tiles in RAM (their allocations).
    pub ram_bytes: u64,
    /// Bytes of compressed tiles in RAM.
    pub compressed_bytes: u64,
    /// Bytes of compressed tiles in the scratch file.
    pub spilled_bytes: u64,
    /// The scratch file's length.
    pub spill_file_bytes: u64,
    pub ram_tiles: u64,
    pub compressed_tiles: u64,
    pub spilled_tiles: u64,
    pub peak_ram_bytes: u64,
    pub peak_compressed_bytes: u64,
    /// The high-water mark of tiers 0 and 1 together.
    pub peak_resident_bytes: u64,
    pub peak_spilled_bytes: u64,
    /// How far over the budget the store went when nothing could be evicted.
    pub peak_over_budget_bytes: u64,
    pub compressions: u64,
    /// Uncompressed and compressed bytes of every compression.
    pub compress_in_bytes: u64,
    pub compress_out_bytes: u64,
    pub compress_seconds: f64,
    pub decompressions: u64,
    pub decompress_seconds: f64,
    pub spill_writes: u64,
    pub spill_write_bytes: u64,
    pub spill_write_seconds: f64,
    pub spill_reads: u64,
    pub spill_read_bytes: u64,
    pub spill_read_seconds: f64,
    /// Tiles read from each tier, and reads of tiles not in the store.
    pub ram_hits: u64,
    pub compressed_hits: u64,
    pub spilled_hits: u64,
    pub misses: u64,
}

impl TileStats {
    /// Records the counters for `--profile-json`.
    pub fn record_profile(&self) {
        use crate::profile::count;
        count("tiles.budget_bytes", self.budget_bytes);
        count("tiles.peak_ram_bytes", self.peak_ram_bytes);
        count("tiles.peak_compressed_bytes", self.peak_compressed_bytes);
        count("tiles.peak_resident_bytes", self.peak_resident_bytes);
        count("tiles.peak_spilled_bytes", self.peak_spilled_bytes);
        count("tiles.peak_over_budget_bytes", self.peak_over_budget_bytes);
        count("tiles.compressions", self.compressions);
        count("tiles.compress_in_bytes", self.compress_in_bytes);
        count("tiles.compress_out_bytes", self.compress_out_bytes);
        count("tiles.compress_us", (self.compress_seconds * 1e6) as u64);
        count("tiles.decompressions", self.decompressions);
        count(
            "tiles.decompress_us",
            (self.decompress_seconds * 1e6) as u64,
        );
        count("tiles.spill_writes", self.spill_writes);
        count("tiles.spill_write_bytes", self.spill_write_bytes);
        count(
            "tiles.spill_write_us",
            (self.spill_write_seconds * 1e6) as u64,
        );
        count("tiles.spill_reads", self.spill_reads);
        count("tiles.spill_read_bytes", self.spill_read_bytes);
        count(
            "tiles.spill_read_us",
            (self.spill_read_seconds * 1e6) as u64,
        );
    }
}

type Key = (u16, TileId);

/// An uncompressed tile of some element type.
trait RawTile: Send + Sync {
    fn bytes(&self) -> u64;
    fn encode(&self, filter: Prefilter, key: BlobKey, out: &mut Vec<u8>);
    fn as_any(&self) -> &dyn Any;
    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;
}

impl<T: TileElement> RawTile for Vec<T> {
    fn bytes(&self) -> u64 {
        (self.capacity() * size_of::<T>()) as u64
    }

    fn encode(&self, filter: Prefilter, key: BlobKey, out: &mut Vec<u8>) {
        encode_with_key(self, filter, key, out);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }
}

#[derive(Clone)]
enum Payload {
    Ram(Arc<dyn RawTile>),
    Compressed(Arc<[u8]>),
    Spilled(Arc<Extent>),
}

impl Payload {
    fn tier(&self) -> Tier {
        match self {
            Self::Ram(_) => Tier::Ram,
            Self::Compressed(_) => Tier::Compressed,
            Self::Spilled(_) => Tier::Spilled,
        }
    }

    fn bytes(&self) -> u64 {
        match self {
            Self::Ram(raw) => raw.bytes(),
            Self::Compressed(blob) => blob.len() as u64,
            Self::Spilled(extent) => extent.len,
        }
    }
}

struct Entry {
    payload: Payload,
    filter: Prefilter,
    generation: u64,
    pins: u32,
    /// The last use; the entry's key in its tier's LRU map unless it is
    /// pinned or being evicted.
    tick: u64,
    busy: bool,
}

#[derive(Default)]
struct State {
    entries: HashMap<Key, Entry>,
    lru: [BTreeMap<u64, Key>; 2],
    tick: u64,
    generation: u64,
    bytes: [u64; 3],
    tiles: [u64; 3],
    peaks: [u64; 3],
    peak_resident: u64,
    peak_over: u64,
}

impl State {
    fn next_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    fn resident(&self) -> u64 {
        self.bytes[0] + self.bytes[1]
    }

    fn account(&mut self, payload: &Payload, add: bool) {
        let tier = payload.tier() as usize;
        let bytes = payload.bytes();
        if add {
            self.bytes[tier] += bytes;
            self.tiles[tier] += 1;
            self.peaks[tier] = self.peaks[tier].max(self.bytes[tier]);
            self.peak_resident = self.peak_resident.max(self.resident());
        } else {
            self.bytes[tier] -= bytes;
            self.tiles[tier] -= 1;
        }
    }

    /// Makes an entry evictable at its tier's LRU position.
    fn enlist(&mut self, key: Key) {
        let entry = &self.entries[&key];
        if entry.pins == 0 && !entry.busy && entry.payload.tier() != Tier::Spilled {
            let (tier, tick) = (entry.payload.tier() as usize, entry.tick);
            self.lru[tier].insert(tick, key);
        }
    }

    fn delist(&mut self, key: Key) {
        let entry = &self.entries[&key];
        if entry.payload.tier() != Tier::Spilled {
            let (tier, tick) = (entry.payload.tier() as usize, entry.tick);
            self.lru[tier].remove(&tick);
        }
    }

    fn remove(&mut self, key: Key) -> Option<Entry> {
        if !self.entries.contains_key(&key) {
            return None;
        }
        self.delist(key);
        let entry = self.entries.remove(&key)?;
        self.account(&entry.payload, false);
        Some(entry)
    }
}

#[derive(Default)]
struct Timers {
    compressions: AtomicU64,
    compress_in: AtomicU64,
    compress_out: AtomicU64,
    compress_ns: AtomicU64,
    decompressions: AtomicU64,
    decompress_ns: AtomicU64,
    spill_writes: AtomicU64,
    spill_write_bytes: AtomicU64,
    spill_write_ns: AtomicU64,
    spill_reads: AtomicU64,
    spill_read_bytes: AtomicU64,
    spill_read_ns: AtomicU64,
    hits: [AtomicU64; 3],
    misses: AtomicU64,
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos() as u64
}

struct Inner {
    budget: u64,
    scratch_dir: Option<PathBuf>,
    state: Mutex<State>,
    spill: Mutex<Option<Arc<SpillFile>>>,
    timers: Timers,
    layers: AtomicU16,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn spill_file(&self) -> Result<Option<Arc<SpillFile>>> {
        let Some(dir) = &self.scratch_dir else {
            return Ok(None);
        };
        let mut spill = self.spill.lock().unwrap_or_else(|p| p.into_inner());
        if spill.is_none() {
            *spill = Some(Arc::new(SpillFile::create(dir)?));
        }
        Ok(spill.clone())
    }

    fn compress(&self, raw: &dyn RawTile, filter: Prefilter, key: BlobKey) -> Arc<[u8]> {
        let start = Instant::now();
        let mut blob = Vec::new();
        raw.encode(filter, key, &mut blob);
        let blob: Arc<[u8]> = blob.into();
        let t = &self.timers;
        t.compress_ns.fetch_add(elapsed_ns(start), Relaxed);
        t.compressions.fetch_add(1, Relaxed);
        t.compress_in.fetch_add(raw.bytes(), Relaxed);
        t.compress_out.fetch_add(blob.len() as u64, Relaxed);
        blob
    }

    fn write_spill(&self, file: &Arc<SpillFile>, blob: &[u8]) -> Result<Arc<Extent>> {
        let start = Instant::now();
        let extent = file.write(blob)?;
        let t = &self.timers;
        t.spill_write_ns.fetch_add(elapsed_ns(start), Relaxed);
        t.spill_writes.fetch_add(1, Relaxed);
        t.spill_write_bytes.fetch_add(blob.len() as u64, Relaxed);
        Ok(extent)
    }

    /// Inserts a payload, replacing any tile of the same key, then evicts
    /// until the store is within its budget.
    fn insert(&self, key: Key, payload: Payload, filter: Prefilter) -> Result<()> {
        let old = {
            let mut state = self.lock();
            let old = state.remove(key);
            let tick = state.next_tick();
            state.generation += 1;
            let generation = state.generation;
            state.account(&payload, true);
            state.entries.insert(
                key,
                Entry {
                    payload,
                    filter,
                    generation,
                    pins: old.as_ref().map_or(0, |e| e.pins),
                    tick,
                    busy: false,
                },
            );
            state.enlist(key);
            old
        };
        // A replaced spilled tile frees its extent here, outside the lock.
        drop(old);
        self.enforce_budget()
    }

    /// Evicts least recently used tiles until RAM tiers fit the budget.
    fn enforce_budget(&self) -> Result<()> {
        loop {
            let mut state = self.lock();
            let resident = state.resident();
            if resident <= self.budget {
                return Ok(());
            }
            let victim = state.lru[0]
                .first_key_value()
                .map(|(_, &key)| key)
                .or_else(|| {
                    self.scratch_dir
                        .as_ref()
                        .and_then(|_| state.lru[1].first_key_value().map(|(_, &key)| key))
                });
            let Some(key) = victim else {
                state.peak_over = state.peak_over.max(resident - self.budget);
                return Ok(());
            };
            state.delist(key);
            let entry = state.entries.get_mut(&key).expect("listed entries exist");
            entry.busy = true;
            let (payload, filter, generation) =
                (entry.payload.clone(), entry.filter, entry.generation);
            drop(state);
            self.transition(key, payload, filter, generation, None)?;
        }
    }

    /// Moves an entry one tier down (`target` `None`) or to `target`, then
    /// swaps the result in unless the entry was replaced, removed or pinned
    /// meanwhile. The entry is `busy` and out of the LRU maps on entry.
    fn transition(
        &self,
        key: Key,
        payload: Payload,
        filter: Prefilter,
        generation: u64,
        target: Option<Tier>,
    ) -> Result<()> {
        let blob_key = BlobKey {
            layer: key.0,
            tile: Some(key.1),
        };
        let result = (|| -> Result<Payload> {
            let target = target.unwrap_or(match payload.tier() {
                Tier::Ram => Tier::Compressed,
                _ => Tier::Spilled,
            });
            let blob = match &payload {
                Payload::Ram(raw) => self.compress(raw.as_ref(), filter, blob_key),
                Payload::Compressed(blob) => blob.clone(),
                Payload::Spilled(_) => return Ok(payload.clone()),
            };
            if target == Tier::Compressed {
                return Ok(Payload::Compressed(blob));
            }
            let file = self
                .spill_file()?
                .ok_or_else(|| Error::Invalid("spilling needs a scratch directory".into()))?;
            Ok(Payload::Spilled(self.write_spill(&file, &blob)?))
        })();
        let mut state = self.lock();
        let current = state
            .entries
            .get(&key)
            .is_some_and(|e| e.generation == generation);
        if !current {
            // Replaced or removed meanwhile: discard the work, outside the lock.
            drop(state);
            return result.map(drop);
        }
        let entry = state.entries.get_mut(&key).expect("current");
        entry.busy = false;
        let keep = entry.pins > 0;
        let discard = match result {
            Ok(new) if !keep => {
                let old = std::mem::replace(&mut entry.payload, new);
                let new = entry.payload.clone();
                state.account(&old, false);
                state.account(&new, true);
                Some(old)
            }
            // Pinned meanwhile: it stays where it was.
            Ok(new) => Some(new),
            Err(error) => {
                state.enlist(key);
                return Err(error);
            }
        };
        state.enlist(key);
        drop(state);
        drop(discard);
        Ok(())
    }

    fn get_payload(&self, key: Key, touch: bool) -> Option<(Payload, Prefilter)> {
        let mut state = self.lock();
        if !state.entries.contains_key(&key) {
            self.timers.misses.fetch_add(1, Relaxed);
            return None;
        }
        if touch {
            state.delist(key);
            let tick = state.next_tick();
            state.entries.get_mut(&key).expect("present").tick = tick;
            state.enlist(key);
        }
        let entry = &state.entries[&key];
        self.timers.hits[entry.payload.tier() as usize].fetch_add(1, Relaxed);
        Some((entry.payload.clone(), entry.filter))
    }

    fn decode<T: TileElement>(&self, key: Key, payload: &Payload, dst: &mut Vec<T>) -> Result<()> {
        let blob_key = BlobKey {
            layer: key.0,
            tile: Some(key.1),
        };
        match payload {
            Payload::Ram(raw) => {
                let data = raw
                    .as_any()
                    .downcast_ref::<Vec<T>>()
                    .ok_or_else(|| Error::Invalid("tile of another element type".into()))?;
                dst.clear();
                dst.extend_from_slice(data);
                Ok(())
            }
            Payload::Compressed(blob) => self.decompress(blob, blob_key, dst),
            Payload::Spilled(extent) => {
                let start = Instant::now();
                let blob = extent.read()?;
                let t = &self.timers;
                t.spill_read_ns.fetch_add(elapsed_ns(start), Relaxed);
                t.spill_reads.fetch_add(1, Relaxed);
                t.spill_read_bytes.fetch_add(blob.len() as u64, Relaxed);
                self.decompress(&blob, blob_key, dst)
            }
        }
    }

    fn decompress<T: TileElement>(
        &self,
        blob: &[u8],
        key: BlobKey,
        dst: &mut Vec<T>,
    ) -> Result<()> {
        let start = Instant::now();
        decode_with_key(blob, Some(key), dst)?;
        self.timers
            .decompress_ns
            .fetch_add(elapsed_ns(start), Relaxed);
        self.timers.decompressions.fetch_add(1, Relaxed);
        Ok(())
    }
}

/// The shared budget, eviction order and scratch file of [`TileStore`]
/// layers. Cloning shares the cache.
#[derive(Clone)]
pub struct TileCache {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for TileCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileCache")
            .field("budget", &self.inner.budget)
            .field("scratch_dir", &self.inner.scratch_dir)
            .finish_non_exhaustive()
    }
}

impl TileCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            inner: Arc::new(Inner {
                budget: config.budget_bytes,
                scratch_dir: config.scratch_dir,
                state: Mutex::new(State::default()),
                spill: Mutex::new(None),
                timers: Timers::default(),
                layers: AtomicU16::new(0),
            }),
        }
    }

    /// A new layer of tiles of `T`.
    pub fn store<T: TileElement>(&self, name: &str, options: StoreOptions) -> TileStore<T> {
        let layer = self.inner.layers.fetch_add(1, Relaxed);
        assert!(layer < u16::MAX, "too many tile store layers");
        TileStore {
            inner: self.inner.clone(),
            layer,
            name: name.to_owned(),
            filter: options.prefilter.unwrap_or(T::DEFAULT_PREFILTER),
            initial: options.initial,
            _element: std::marker::PhantomData,
        }
    }

    pub fn budget_bytes(&self) -> u64 {
        self.inner.budget
    }

    /// Evicts until the RAM tiers fit the budget again (after unpinning).
    pub fn enforce_budget(&self) -> Result<()> {
        self.inner.enforce_budget()
    }

    pub fn stats(&self) -> TileStats {
        let state = self.inner.lock();
        let t = &self.inner.timers;
        let secs = |ns: &AtomicU64| ns.load(Relaxed) as f64 * 1e-9;
        let spill_file_bytes = self
            .inner
            .spill
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map_or(0, |f| f.len());
        TileStats {
            budget_bytes: self.inner.budget,
            ram_bytes: state.bytes[0],
            compressed_bytes: state.bytes[1],
            spilled_bytes: state.bytes[2],
            spill_file_bytes,
            ram_tiles: state.tiles[0],
            compressed_tiles: state.tiles[1],
            spilled_tiles: state.tiles[2],
            peak_ram_bytes: state.peaks[0],
            peak_compressed_bytes: state.peaks[1],
            peak_resident_bytes: state.peak_resident,
            peak_spilled_bytes: state.peaks[2],
            peak_over_budget_bytes: state.peak_over,
            compressions: t.compressions.load(Relaxed),
            compress_in_bytes: t.compress_in.load(Relaxed),
            compress_out_bytes: t.compress_out.load(Relaxed),
            compress_seconds: secs(&t.compress_ns),
            decompressions: t.decompressions.load(Relaxed),
            decompress_seconds: secs(&t.decompress_ns),
            spill_writes: t.spill_writes.load(Relaxed),
            spill_write_bytes: t.spill_write_bytes.load(Relaxed),
            spill_write_seconds: secs(&t.spill_write_ns),
            spill_reads: t.spill_reads.load(Relaxed),
            spill_read_bytes: t.spill_read_bytes.load(Relaxed),
            spill_read_seconds: secs(&t.spill_read_ns),
            ram_hits: t.hits[0].load(Relaxed),
            compressed_hits: t.hits[1].load(Relaxed),
            spilled_hits: t.hits[2].load(Relaxed),
            misses: t.misses.load(Relaxed),
        }
    }

    /// The scratch file's path, once a tile has spilled.
    pub fn spill_path(&self) -> Option<PathBuf> {
        let spill = self.inner.spill.lock().unwrap_or_else(|p| p.into_inner());
        spill.as_ref().map(|f| f.path.clone())
    }
}

/// One layer of tiles of `T` in a [`TileCache`]. All methods take `&self`
/// and may be called from many threads. Dropping the store removes its tiles.
pub struct TileStore<T: TileElement> {
    inner: Arc<Inner>,
    layer: u16,
    name: String,
    filter: Prefilter,
    initial: Tier,
    _element: std::marker::PhantomData<fn() -> T>,
}

impl<T: TileElement> std::fmt::Debug for TileStore<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TileStore")
            .field("name", &self.name)
            .field("layer", &self.layer)
            .field("filter", &self.filter)
            .finish_non_exhaustive()
    }
}

impl<T: TileElement> TileStore<T> {
    /// A store with a cache of its own.
    pub fn new(config: CacheConfig, options: StoreOptions) -> Self {
        TileCache::new(config).store("tiles", options)
    }

    pub fn cache(&self) -> TileCache {
        TileCache {
            inner: self.inner.clone(),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn prefilter(&self) -> Prefilter {
        self.filter
    }

    fn key(&self, id: TileId) -> Key {
        (self.layer, id)
    }

    /// Stores a copy of `data` as tile `id`, replacing any tile there.
    pub fn put(&self, id: TileId, data: &[T]) -> Result<()> {
        self.put_owned(id, data.to_vec())
    }

    /// Stores `data` as tile `id` without copying it (in [`Tier::Ram`]).
    pub fn put_owned(&self, id: TileId, data: Vec<T>) -> Result<()> {
        let key = self.key(id);
        let raw: Arc<dyn RawTile> = Arc::new(data);
        let payload = match self.initial {
            Tier::Ram => Payload::Ram(raw),
            tier => {
                let blob_key = BlobKey {
                    layer: self.layer,
                    tile: Some(id),
                };
                let blob = self.inner.compress(raw.as_ref(), self.filter, blob_key);
                drop(raw);
                match (tier, self.inner.spill_file()?) {
                    (Tier::Spilled, Some(file)) => {
                        Payload::Spilled(self.inner.write_spill(&file, &blob)?)
                    }
                    _ => Payload::Compressed(blob),
                }
            }
        };
        self.inner.insert(key, payload, self.filter)
    }

    /// Copies tile `id` into `dst`, resized to the tile. Errors when the
    /// tile is absent or fails its integrity check.
    pub fn get_into(&self, id: TileId, dst: &mut Vec<T>) -> Result<()> {
        if self.try_get_into(id, dst)? {
            Ok(())
        } else {
            Err(Error::Invalid(format!(
                "no tile {id:?} in store {}",
                self.name
            )))
        }
    }

    /// [`Self::get_into`], returning `false` for an absent tile.
    pub fn try_get_into(&self, id: TileId, dst: &mut Vec<T>) -> Result<bool> {
        let key = self.key(id);
        let Some((payload, _)) = self.inner.get_payload(key, true) else {
            return Ok(false);
        };
        self.inner.decode(key, &payload, dst)?;
        Ok(true)
    }

    /// A copy of tile `id`, if present.
    pub fn get(&self, id: TileId) -> Result<Option<Vec<T>>> {
        let mut dst = Vec::new();
        Ok(self.try_get_into(id, &mut dst)?.then_some(dst))
    }

    /// Removes tile `id` and returns it; an uncompressed tile is returned
    /// without a copy unless another thread is reading it.
    pub fn take(&self, id: TileId) -> Result<Option<Vec<T>>> {
        let key = self.key(id);
        let entry = self.inner.lock().remove(key);
        let Some(entry) = entry else {
            return Ok(None);
        };
        if let Payload::Ram(raw) = entry.payload {
            let any = raw.into_any();
            let tile = any
                .downcast::<Vec<T>>()
                .map_err(|_| Error::Invalid("tile of another element type".into()))?;
            return Ok(Some(
                Arc::try_unwrap(tile).unwrap_or_else(|shared| shared.as_ref().clone()),
            ));
        }
        let mut dst = Vec::new();
        self.inner.decode(key, &entry.payload, &mut dst)?;
        Ok(Some(dst))
    }

    /// Removes tile `id`; `false` when it was absent.
    pub fn remove(&self, id: TileId) -> bool {
        let entry = self.inner.lock().remove(self.key(id));
        entry.is_some()
    }

    /// Removes every tile of this store.
    pub fn clear(&self) {
        let removed: Vec<Entry> = {
            let mut state = self.inner.lock();
            let keys: Vec<Key> = state
                .entries
                .keys()
                .filter(|k| k.0 == self.layer)
                .copied()
                .collect();
            keys.into_iter().filter_map(|k| state.remove(k)).collect()
        };
        drop(removed);
    }

    pub fn contains(&self, id: TileId) -> bool {
        self.inner.lock().entries.contains_key(&self.key(id))
    }

    /// The tier tile `id` is held in.
    pub fn tier(&self, id: TileId) -> Option<Tier> {
        let state = self.inner.lock();
        state.entries.get(&self.key(id)).map(|e| e.payload.tier())
    }

    /// The tiles of this store, row-major.
    pub fn ids(&self) -> Vec<TileId> {
        let state = self.inner.lock();
        let mut ids: Vec<_> = state
            .entries
            .keys()
            .filter(|k| k.0 == self.layer)
            .map(|k| k.1)
            .collect();
        ids.sort();
        ids
    }

    pub fn len(&self) -> usize {
        let state = self.inner.lock();
        state.entries.keys().filter(|k| k.0 == self.layer).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Keeps tile `id` in its tier until the returned guard drops. Pins nest.
    pub fn pin(&self, id: TileId) -> Result<TilePin<'_, T>> {
        let key = self.key(id);
        let mut state = self.inner.lock();
        if !state.entries.contains_key(&key) {
            return Err(Error::Invalid(format!(
                "no tile {id:?} in store {}",
                self.name
            )));
        }
        state.delist(key);
        state.entries.get_mut(&key).expect("present").pins += 1;
        Ok(TilePin { store: self, id })
    }

    fn unpin(&self, id: TileId) -> Result<()> {
        let key = self.key(id);
        {
            let mut state = self.inner.lock();
            let Some(entry) = state.entries.get_mut(&key) else {
                return Ok(());
            };
            entry.pins = entry.pins.saturating_sub(1);
            if entry.pins > 0 {
                return Ok(());
            }
            state.delist(key);
            state.enlist(key);
        }
        self.inner.enforce_budget()
    }

    /// Moves tile `id` down to `tier` now (compress, or compress and spill),
    /// whatever the budget. A tile already at or below `tier`, pinned, or
    /// being evicted by another thread is left as it is.
    pub fn evict(&self, id: TileId, tier: Tier) -> Result<()> {
        let key = self.key(id);
        let mut state = self.inner.lock();
        let Some(entry) = state.entries.get(&key) else {
            return Err(Error::Invalid(format!(
                "no tile {id:?} in store {}",
                self.name
            )));
        };
        if entry.payload.tier() >= tier || entry.pins > 0 || entry.busy {
            return Ok(());
        }
        state.delist(key);
        let entry = state.entries.get_mut(&key).expect("present");
        entry.busy = true;
        let (payload, filter, generation) = (entry.payload.clone(), entry.filter, entry.generation);
        drop(state);
        self.inner
            .transition(key, payload, filter, generation, Some(tier))
    }

    /// The number of elements of tile `id`, without decoding it.
    pub fn elements(&self, id: TileId) -> Result<Option<u64>> {
        let key = self.key(id);
        let Some((payload, _)) = self.inner.get_payload(key, false) else {
            return Ok(None);
        };
        Ok(Some(match payload {
            Payload::Ram(raw) => raw
                .as_any()
                .downcast_ref::<Vec<T>>()
                .map_or(0, |v| v.len() as u64),
            Payload::Compressed(blob) => blob_elements(&blob)?,
            Payload::Spilled(extent) => blob_elements(&extent.read()?)?,
        }))
    }

    /// Flips one bit of the stored bytes of tile `id`, to test integrity checks.
    #[doc(hidden)]
    pub fn corrupt_for_test(&self, id: TileId, byte: usize) -> Result<()> {
        let key = self.key(id);
        let mut state = self.inner.lock();
        let entry = state
            .entries
            .get_mut(&key)
            .ok_or_else(|| Error::Invalid("no such tile".into()))?;
        match &mut entry.payload {
            Payload::Ram(_) => Err(Error::Invalid(
                "uncompressed tiles carry no checksum".into(),
            )),
            Payload::Compressed(blob) => {
                let mut bytes = blob.to_vec();
                let byte = byte % bytes.len();
                bytes[byte] ^= 0x01;
                *blob = bytes.into();
                Ok(())
            }
            Payload::Spilled(extent) => {
                let mut bytes = extent.read()?;
                let byte = byte % bytes.len();
                bytes[byte] ^= 0x01;
                extent.file.write_at(&bytes, extent.offset)?;
                Ok(())
            }
        }
    }
}

impl<T: TileElement> Drop for TileStore<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

/// A pin from [`TileStore::pin`].
pub struct TilePin<'a, T: TileElement> {
    store: &'a TileStore<T>,
    id: TileId,
}

impl<T: TileElement> TilePin<'_, T> {
    pub fn id(&self) -> TileId {
        self.id
    }
}

impl<T: TileElement> Drop for TilePin<'_, T> {
    fn drop(&mut self) {
        // Over-budget eviction after unpinning can only fail with an I/O
        // error, which the next put reports again.
        let _ = self.store.unpin(self.id);
    }
}

/// Bytes the scratch file allocates in; extents are reused once freed.
const SPILL_ALIGN: u64 = 4096;

/// A scratch file deleted when closed (Windows) or at once (Unix).
struct SpillFile {
    file: File,
    path: PathBuf,
    space: Mutex<Space>,
}

#[derive(Default)]
struct Space {
    /// Free extents by offset, coalesced.
    free: BTreeMap<u64, u64>,
    end: u64,
}

impl Space {
    fn allocate(&mut self, len: u64) -> u64 {
        let len = len.next_multiple_of(SPILL_ALIGN);
        let found = self
            .free
            .iter()
            .find(|&(_, &l)| l >= len)
            .map(|(&o, &l)| (o, l));
        if let Some((offset, free)) = found {
            self.free.remove(&offset);
            if free > len {
                self.free.insert(offset + len, free - len);
            }
            return offset;
        }
        let offset = self.end;
        self.end += len;
        offset
    }

    fn release(&mut self, mut offset: u64, len: u64) {
        let mut len = len.next_multiple_of(SPILL_ALIGN);
        if let Some((&before, &l)) = self.free.range(..offset).next_back()
            && before + l == offset
        {
            self.free.remove(&before);
            offset = before;
            len += l;
        }
        if let Some(l) = self.free.remove(&(offset + len)) {
            len += l;
        }
        if offset + len == self.end {
            self.end = offset;
        } else {
            self.free.insert(offset, len);
        }
    }
}

static SPILL_FILES: AtomicU64 = AtomicU64::new(0);

impl SpillFile {
    fn create(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let n = SPILL_FILES.fetch_add(1, Relaxed);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let path = dir.join(format!(
            "taudem-rs-{}-{nanos:09}-{n}.spill",
            std::process::id()
        ));
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            const FILE_ATTRIBUTE_TEMPORARY: u32 = 0x100;
            const FILE_FLAG_DELETE_ON_CLOSE: u32 = 0x0400_0000;
            options
                .attributes(FILE_ATTRIBUTE_TEMPORARY)
                .custom_flags(FILE_FLAG_DELETE_ON_CLOSE);
        }
        let file = options.open(&path)?;
        #[cfg(unix)]
        std::fs::remove_file(&path)?;
        Ok(Self {
            file,
            path,
            space: Mutex::new(Space::default()),
        })
    }

    fn space(&self) -> MutexGuard<'_, Space> {
        self.space.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn len(&self) -> u64 {
        self.space().end
    }

    fn write(self: &Arc<Self>, blob: &[u8]) -> Result<Arc<Extent>> {
        let len = blob.len() as u64;
        let offset = self.space().allocate(len);
        let extent = Arc::new(Extent {
            file: self.clone(),
            offset,
            len,
        });
        // On failure the extent drops and its space is released.
        self.write_at(blob, offset)?;
        Ok(extent)
    }

    #[cfg(windows)]
    fn write_at(&self, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = self.file.seek_write(buf, offset)?;
            if n == 0 {
                return Err(std::io::ErrorKind::WriteZero.into());
            }
            buf = &buf[n..];
            offset += n as u64;
        }
        Ok(())
    }

    #[cfg(unix)]
    fn write_at(&self, buf: &[u8], offset: u64) -> std::io::Result<()> {
        std::os::unix::fs::FileExt::write_all_at(&self.file, buf, offset)
    }

    #[cfg(windows)]
    fn read_at(&self, mut buf: &mut [u8], mut offset: u64) -> std::io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            let n = self.file.seek_read(buf, offset)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            buf = &mut buf[n..];
            offset += n as u64;
        }
        Ok(())
    }

    #[cfg(unix)]
    fn read_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
        std::os::unix::fs::FileExt::read_exact_at(&self.file, buf, offset)
    }
}

/// A blob in the scratch file; its space is released when the last
/// reference drops, so a concurrent read never sees it reused.
struct Extent {
    file: Arc<SpillFile>,
    offset: u64,
    len: u64,
}

impl Extent {
    fn read(&self) -> Result<Vec<u8>> {
        let mut blob = vec![0; self.len as usize];
        self.file.read_at(&mut blob, self.offset)?;
        Ok(blob)
    }
}

impl Drop for Extent {
    fn drop(&mut self) {
        self.file.space().release(self.offset, self.len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs of equal values: compresses well.
    fn smooth(seed: u32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 100.0 + ((i as u32 / 50 + seed) % 7) as f32 * 0.25)
            .collect()
    }

    /// Pseudo-random values: compresses hardly at all.
    fn tile(seed: u32, n: usize) -> Vec<f32> {
        let mut state = u64::from(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 * 0.001
            })
            .collect()
    }

    fn scratch() -> PathBuf {
        std::env::temp_dir().join("taudem-rs-store-tests")
    }

    #[test]
    fn tiles_round_trip_in_every_tier() {
        let cache = TileCache::new(CacheConfig::new(u64::MAX, Some(scratch())));
        for initial in [Tier::Ram, Tier::Compressed, Tier::Spilled] {
            for prefilter in Prefilter::ALL {
                let fel = cache.store::<f32>(
                    "fel",
                    StoreOptions {
                        prefilter: Some(prefilter),
                        initial,
                    },
                );
                let dir = cache.store::<u8>(
                    "dir",
                    StoreOptions {
                        prefilter: Some(prefilter),
                        initial,
                    },
                );
                let fall = cache.store::<u32>(
                    "fall",
                    StoreOptions {
                        prefilter: Some(prefilter),
                        initial,
                    },
                );
                for t in 0..4u32 {
                    let id = TileId::new(t, t + 1);
                    let n = 1000 + t as usize * 333;
                    fel.put(id, &tile(t, n)).unwrap();
                    dir.put(id, &(0..n).map(|i| (i % 9) as u8).collect::<Vec<_>>())
                        .unwrap();
                    fall.put(id, &(0..n as u32).map(|i| i / 7).collect::<Vec<_>>())
                        .unwrap();
                    assert_eq!(fel.tier(id), Some(initial));
                }
                for t in 0..4u32 {
                    let id = TileId::new(t, t + 1);
                    let n = 1000 + t as usize * 333;
                    let mut out = Vec::new();
                    fel.get_into(id, &mut out).unwrap();
                    assert_eq!(out, tile(t, n));
                    assert_eq!(
                        dir.get(id).unwrap().unwrap(),
                        (0..n).map(|i| (i % 9) as u8).collect::<Vec<_>>()
                    );
                    assert_eq!(fall.elements(id).unwrap(), Some(n as u64));
                    assert_eq!(
                        fall.take(id).unwrap().unwrap(),
                        (0..n as u32).map(|i| i / 7).collect::<Vec<_>>()
                    );
                    assert!(!fall.contains(id));
                }
                assert!(fel.get(TileId::new(9, 9)).unwrap().is_none());
                assert!(fel.get_into(TileId::new(9, 9), &mut Vec::new()).is_err());
            }
        }
        let stats = cache.stats();
        assert_eq!(
            stats.ram_bytes + stats.compressed_bytes + stats.spilled_bytes,
            0
        );
        assert!(stats.spill_writes > 0 && stats.spill_reads > 0 && stats.decompressions > 0);
    }

    #[test]
    fn layers_share_one_budget_and_evict_lru_first() {
        let n = 10_000; // 40 kB per f32 tile
        let cache = TileCache::new(CacheConfig::new(100_000, Some(scratch())));
        let a = cache.store::<f32>("a", StoreOptions::default());
        let b = cache.store::<f32>("b", StoreOptions::default());
        a.put(TileId::new(0, 0), &smooth(0, n)).unwrap();
        b.put(TileId::new(0, 0), &smooth(1, n)).unwrap();
        // Touch a's tile so b's becomes the least recently used.
        a.get(TileId::new(0, 0)).unwrap();
        a.put(TileId::new(1, 0), &smooth(2, n)).unwrap();
        assert_eq!(b.tier(TileId::new(0, 0)), Some(Tier::Compressed));
        assert_eq!(a.tier(TileId::new(0, 0)), Some(Tier::Ram));
        let stats = cache.stats();
        assert!(stats.ram_bytes + stats.compressed_bytes <= 100_000);
        // Many more tiles: everything still reads back, and the RAM tiers
        // stay within the budget once eviction reaches the spill file.
        for t in 0..60u32 {
            a.put(TileId::new(t, 5), &tile(t, n)).unwrap();
        }
        let stats = cache.stats();
        assert!(
            stats.ram_bytes + stats.compressed_bytes <= 100_000,
            "{stats:?}"
        );
        assert!(stats.spilled_tiles > 0);
        assert_eq!(stats.peak_over_budget_bytes, 0);
        for t in 0..60u32 {
            assert_eq!(a.get(TileId::new(t, 5)).unwrap().unwrap(), tile(t, n));
        }
        assert_eq!(b.get(TileId::new(0, 0)).unwrap().unwrap(), smooth(1, n));
        drop(a);
        drop(b);
        let stats = cache.stats();
        assert_eq!(
            (stats.ram_tiles, stats.compressed_tiles, stats.spilled_tiles),
            (0, 0, 0)
        );
        assert_eq!(
            stats.spill_file_bytes, 0,
            "freed extents coalesce back to an empty file"
        );
    }

    #[test]
    fn without_scratch_dir_the_budget_is_exceeded_and_counted() {
        let cache = TileCache::new(CacheConfig::new(10_000, None));
        let s = cache.store::<f32>("s", StoreOptions::default());
        for t in 0..10 {
            s.put(TileId::new(t, 0), &tile(t, 10_000)).unwrap();
        }
        let stats = cache.stats();
        assert_eq!(stats.ram_tiles, 0);
        assert_eq!(stats.compressed_tiles, 10);
        assert!(stats.peak_over_budget_bytes > 0);
        assert!(s.evict(TileId::new(0, 0), Tier::Spilled).is_err());
    }

    #[test]
    fn pinned_tiles_stay_put_and_explicit_eviction_moves_tiles() {
        let n = 10_000;
        let cache = TileCache::new(CacheConfig::new(50_000, Some(scratch())));
        let s = cache.store::<f32>("s", StoreOptions::default());
        let hot = TileId::new(0, 0);
        s.put(hot, &tile(0, n)).unwrap();
        {
            let _pin = s.pin(hot).unwrap();
            for t in 1..8 {
                s.put(TileId::new(t, 0), &tile(t, n)).unwrap();
            }
            assert_eq!(s.tier(hot), Some(Tier::Ram));
            assert!(s.evict(hot, Tier::Spilled).is_ok());
            assert_eq!(s.tier(hot), Some(Tier::Ram));
        }
        // Unpinned and over budget with others already compressed or spilled.
        s.put(TileId::new(9, 0), &tile(9, n)).unwrap();
        assert_ne!(s.tier(hot), Some(Tier::Ram));
        let id = TileId::new(9, 0);
        s.evict(id, Tier::Spilled).unwrap();
        assert_eq!(s.tier(id), Some(Tier::Spilled));
        assert_eq!(s.get(id).unwrap().unwrap(), tile(9, n));
        assert!(s.pin(TileId::new(99, 0)).is_err());
    }

    #[test]
    fn corrupted_tiles_are_refused() {
        let cache = TileCache::new(CacheConfig::new(u64::MAX, Some(scratch())));
        let s = cache.store::<f32>("s", StoreOptions::default());
        let id = TileId::new(1, 2);
        for tier in [Tier::Compressed, Tier::Spilled] {
            for byte in [3, 40, 100, 5000] {
                s.put(id, &tile(7, 20_000)).unwrap();
                s.evict(id, tier).unwrap();
                assert_eq!(s.tier(id), Some(tier));
                s.corrupt_for_test(id, byte).unwrap();
                assert!(
                    matches!(s.get(id), Err(Error::Corrupt(_))),
                    "{tier:?} byte {byte}"
                );
            }
        }
        // A blob of one tile is never accepted as another.
        let other = cache.store::<f32>("other", StoreOptions::default());
        other.put(id, &tile(7, 100)).unwrap();
        other.evict(id, Tier::Compressed).unwrap();
        assert_eq!(other.get(id).unwrap().unwrap(), tile(7, 100));
    }

    #[test]
    fn spill_space_is_reused() {
        let mut space = Space::default();
        let a = space.allocate(5000);
        let b = space.allocate(100);
        let c = space.allocate(9000);
        assert_eq!((a, b, c), (0, 8192, 12288));
        space.release(b, 100);
        assert_eq!(space.allocate(4096), 8192);
        space.release(8192, 4096);
        space.release(a, 5000);
        assert_eq!(space.free.get(&0), Some(&12288));
        space.release(c, 9000);
        assert!(space.free.is_empty());
        assert_eq!(space.end, 0);
    }

    #[test]
    fn concurrent_puts_and_gets_keep_every_tile() {
        let n = 4096;
        let cache = TileCache::new(CacheConfig::new(200_000, Some(scratch())));
        let s = cache.store::<f32>("s", StoreOptions::default());
        let ids: Vec<TileId> = (0..64).map(|i| TileId::new(i % 8, i / 8)).collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        use rayon::prelude::*;
        pool.install(|| {
            for round in 0..3u32 {
                ids.par_iter().for_each(|&id| {
                    let seed = id.tx * 31 + id.ty * 7 + round;
                    s.put(id, &tile(seed, n)).unwrap();
                    let mut back = Vec::new();
                    s.get_into(id, &mut back).unwrap();
                    assert_eq!(back, tile(seed, n));
                });
            }
        });
        for &id in &ids {
            let seed = id.tx * 31 + id.ty * 7 + 2;
            assert_eq!(s.get(id).unwrap().unwrap(), tile(seed, n));
        }
        let stats = cache.stats();
        assert!(
            stats.ram_bytes + stats.compressed_bytes <= 200_000 + 8 * (n as u64) * 4,
            "{stats:?}"
        );
    }
}
