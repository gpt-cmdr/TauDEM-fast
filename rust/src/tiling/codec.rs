// SPDX-License-Identifier: GPL-3.0-only
//! The compressed tile format of [`super::TileStore`]: optional delta and
//! byte-shuffle prefilters, LZ4 blocks (`lz4_flex`), and checksums.
//!
//! A blob is a 64-byte header followed by chunks of at most [`CHUNK_BYTES`]
//! raw bytes, each compressed on its own. LZ4's match window is 64 KiB, so
//! chunking costs almost no ratio, and it bounds the per-thread scratch
//! memory at about two chunks whatever the tile size. Each chunk is
//! `u32 length | STORED flag` followed by its bytes; a chunk LZ4 cannot
//! shrink is stored filtered but uncompressed.

use std::cell::RefCell;

use lz4_flex::block::{
    CompressTable, compress_into_with_table, decompress_into, get_maximum_output_size,
};

use super::spec::TileId;
use super::{Error, Result};

/// Raw bytes per compressed chunk.
pub const CHUNK_BYTES: usize = 1 << 20;
const MAGIC: [u8; 4] = *b"TDT1";
const VERSION: u8 = 1;
const HEADER: usize = 64;
const STORED: u32 = 1 << 31;

mod sealed {
    pub trait Sealed {}
}

/// The sample types tiles hold: plain numbers whose every bit pattern is a
/// valid value, so tiles convert to and from bytes without copying.
pub trait TileElement: sealed::Sealed + Copy + Default + PartialEq + Send + Sync + 'static {
    /// A tag recorded in compressed blobs, checked when they are decoded.
    const TAG: u8;
    /// The prefilter that compresses this type's typical tiles best, as
    /// measured by `examples/tile_codec_bench.rs` on a LiDAR DEM: byte
    /// shuffle for floats and 16-bit codes (`fel`, `ad8`, `p` as i16), delta
    /// for 32- and 64-bit integers (BFS distances such as `fall`/`rise`),
    /// none for bytes (`p` as u8).
    const DEFAULT_PREFILTER: Prefilter;

    /// The bytes of `data`, in native order.
    fn as_bytes(data: &[Self]) -> &[u8] {
        // SAFETY: Sealed to primitive numbers, which have no padding, so all
        // `size_of_val(data)` bytes are initialized.
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) }
    }

    /// The bytes of `data`, writable.
    fn as_bytes_mut(data: &mut [Self]) -> &mut [u8] {
        // SAFETY: As for `as_bytes`; every bit pattern of these types is a
        // valid value, so any bytes written leave valid elements.
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), size_of_val(data)) }
    }
}

macro_rules! element {
    ($($t:ty: $tag:expr, $filter:ident;)*) => {$(
        impl sealed::Sealed for $t {}
        impl TileElement for $t {
            const TAG: u8 = $tag;
            const DEFAULT_PREFILTER: Prefilter = Prefilter::$filter;
        }
    )*};
}

element! {
    u8: 1, None;
    i8: 2, None;
    u16: 3, Shuffle;
    i16: 4, Shuffle;
    u32: 5, Delta;
    i32: 6, Delta;
    u64: 7, Delta;
    i64: 8, Delta;
    f32: 9, Shuffle;
    f64: 10, Shuffle;
}

/// A reversible transform applied before LZ4.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum Prefilter {
    /// The samples' bytes as they are.
    #[default]
    None,
    /// Byte planes: every sample's first byte, then every second byte, and so on.
    /// Separates the slowly varying sign/exponent bytes of floats.
    Shuffle,
    /// Each sample minus the one before it (row-major, wrapping, on the
    /// sample's bits as an unsigned integer): small for smooth integer grids.
    Delta,
    /// Delta, then shuffle.
    DeltaShuffle,
}

impl Prefilter {
    pub const ALL: [Self; 4] = [Self::None, Self::Shuffle, Self::Delta, Self::DeltaShuffle];

    const fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::Shuffle => 1,
            Self::Delta => 2,
            Self::DeltaShuffle => 3,
        }
    }

    fn from_code(code: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.code() == code)
    }

    const fn delta(self) -> bool {
        matches!(self, Self::Delta | Self::DeltaShuffle)
    }

    const fn shuffle(self) -> bool {
        matches!(self, Self::Shuffle | Self::DeltaShuffle)
    }
}

/// The bytes of a blob's header.
pub const fn encoded_header_bytes() -> usize {
    HEADER
}

/// The identity recorded in a blob: which store layer and tile it holds.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BlobKey {
    pub layer: u16,
    pub tile: Option<TileId>,
}

struct Scratch {
    filtered: Vec<u8>,
    compressed: Vec<u8>,
    table: CompressTable,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
        filtered: Vec::new(),
        compressed: Vec::new(),
        table: CompressTable::large(),
    });
}

/// Compresses `data` with `filter` into `out` (cleared first).
pub fn encode_tile<T: TileElement>(data: &[T], filter: Prefilter, out: &mut Vec<u8>) {
    encode_with_key(data, filter, BlobKey::default(), out);
}

/// Decodes a blob of [`encode_tile`] into `dst` (resized to the tile),
/// verifying its checksums, element type and sizes.
pub fn decode_tile<T: TileElement>(blob: &[u8], dst: &mut Vec<T>) -> Result<()> {
    decode_with_key(blob, None, dst)
}

pub(crate) fn encode_with_key<T: TileElement>(
    data: &[T],
    filter: Prefilter,
    key: BlobKey,
    out: &mut Vec<u8>,
) {
    let size = size_of::<T>();
    // A single-byte sample has one plane: shuffle is the identity.
    let filter = match (filter, size) {
        (Prefilter::Shuffle, 1) => Prefilter::None,
        (Prefilter::DeltaShuffle, 1) => Prefilter::Delta,
        (f, _) => f,
    };
    let bytes = T::as_bytes(data);
    out.clear();
    out.resize(HEADER, 0);
    let chunk_elements = CHUNK_BYTES / size;
    SCRATCH.with_borrow_mut(|s| {
        let mut previous = 0u64;
        for chunk in bytes.chunks(chunk_elements * size) {
            let input: &[u8] = if filter == Prefilter::None {
                chunk
            } else {
                s.filtered.resize(chunk.len(), 0);
                previous = forward(chunk, &mut s.filtered, size, filter, previous);
                &s.filtered
            };
            s.compressed.resize(get_maximum_output_size(input.len()), 0);
            let n = compress_into_with_table(input, &mut s.compressed, &mut s.table)
                .expect("output sized by get_maximum_output_size");
            if n < input.len() {
                out.extend_from_slice(&(n as u32).to_le_bytes());
                out.extend_from_slice(&s.compressed[..n]);
            } else {
                out.extend_from_slice(&(input.len() as u32 | STORED).to_le_bytes());
                out.extend_from_slice(input);
            }
        }
    });
    let payload = (out.len() - HEADER) as u64;
    let checksum = hash64(&out[HEADER..]);
    let header = &mut out[..HEADER];
    header[0..4].copy_from_slice(&MAGIC);
    header[4] = VERSION;
    header[5] = T::TAG;
    header[6] = size as u8;
    header[7] = filter.code();
    header[8..16].copy_from_slice(&(data.len() as u64).to_le_bytes());
    header[16..24].copy_from_slice(&payload.to_le_bytes());
    header[24..32].copy_from_slice(&checksum.to_le_bytes());
    header[32..34].copy_from_slice(&key.layer.to_le_bytes());
    header[34] = u8::from(key.tile.is_some());
    let tile = key.tile.unwrap_or(TileId::new(0, 0));
    header[36..40].copy_from_slice(&tile.tx.to_le_bytes());
    header[40..44].copy_from_slice(&tile.ty.to_le_bytes());
    let header_checksum = hash64(&header[..56]);
    header[56..64].copy_from_slice(&header_checksum.to_le_bytes());
}

/// The number of elements a blob holds, after checking its header.
pub(crate) fn blob_elements(blob: &[u8]) -> Result<u64> {
    let header = blob
        .get(..HEADER)
        .ok_or_else(|| Error::Corrupt(format!("{} bytes is shorter than a header", blob.len())))?;
    if header[0..4] != MAGIC || header[4] != VERSION {
        return Err(Error::Corrupt("bad magic or version".into()));
    }
    if u64::from_le_bytes(header[56..64].try_into().unwrap()) != hash64(&header[..56]) {
        return Err(Error::Corrupt("header checksum mismatch".into()));
    }
    Ok(u64::from_le_bytes(header[8..16].try_into().unwrap()))
}

pub(crate) fn decode_with_key<T: TileElement>(
    blob: &[u8],
    expect: Option<BlobKey>,
    dst: &mut Vec<T>,
) -> Result<()> {
    let elements = blob_elements(blob)?;
    let header = &blob[..HEADER];
    let size = size_of::<T>();
    if header[5] != T::TAG || header[6] as usize != size {
        return Err(Error::Corrupt(format!(
            "blob holds element type {} of {} bytes, not {}",
            header[5],
            header[6],
            std::any::type_name::<T>()
        )));
    }
    let filter = Prefilter::from_code(header[7])
        .ok_or_else(|| Error::Corrupt(format!("unknown prefilter {}", header[7])))?;
    let payload = u64::from_le_bytes(header[16..24].try_into().unwrap());
    if blob.len() as u64 - HEADER as u64 != payload {
        return Err(Error::Corrupt(format!(
            "payload of {} bytes, header says {payload}",
            blob.len() - HEADER
        )));
    }
    if let Some(key) = expect {
        let layer = u16::from_le_bytes(header[32..34].try_into().unwrap());
        let tile = (header[34] == 1).then(|| {
            TileId::new(
                u32::from_le_bytes(header[36..40].try_into().unwrap()),
                u32::from_le_bytes(header[40..44].try_into().unwrap()),
            )
        });
        if layer != key.layer || tile != key.tile {
            return Err(Error::Corrupt(format!(
                "blob of layer {layer} tile {tile:?} where layer {} tile {:?} was expected",
                key.layer, key.tile
            )));
        }
    }
    let body = &blob[HEADER..];
    if u64::from_le_bytes(header[24..32].try_into().unwrap()) != hash64(body) {
        return Err(Error::Corrupt("payload checksum mismatch".into()));
    }
    let elements =
        usize::try_from(elements).map_err(|_| Error::Corrupt("tile too large".into()))?;
    dst.clear();
    dst.resize(elements, T::default());
    let out = T::as_bytes_mut(dst);
    let chunk_bytes = CHUNK_BYTES / size * size;
    let mut rest = body;
    let mut previous = 0u64;
    SCRATCH.with_borrow_mut(|s| -> Result<()> {
        for target in out.chunks_mut(chunk_bytes) {
            let (word, tail) = rest
                .split_first_chunk::<4>()
                .ok_or_else(|| Error::Corrupt("missing chunk".into()))?;
            let word = u32::from_le_bytes(*word);
            let stored = word & STORED != 0;
            let n = (word & !STORED) as usize;
            let (bytes, tail) = tail
                .split_at_checked(n)
                .ok_or_else(|| Error::Corrupt("truncated chunk".into()))?;
            rest = tail;
            let unfiltered = filter == Prefilter::None;
            let sink: &mut [u8] = if unfiltered {
                target
            } else {
                s.filtered.resize(target.len(), 0);
                &mut s.filtered
            };
            if stored {
                if n != sink.len() {
                    return Err(Error::Corrupt("stored chunk of the wrong length".into()));
                }
                sink.copy_from_slice(bytes);
            } else {
                let got =
                    decompress_into(bytes, sink).map_err(|e| Error::Corrupt(e.to_string()))?;
                if got != sink.len() {
                    return Err(Error::Corrupt(format!(
                        "chunk decoded to {got} bytes, not {}",
                        sink.len()
                    )));
                }
            }
            if !unfiltered {
                previous = inverse(&s.filtered, target, size, filter, previous);
            }
        }
        Ok(())
    })?;
    if !rest.is_empty() {
        return Err(Error::Corrupt("bytes after the last chunk".into()));
    }
    Ok(())
}

/// Integer lanes the delta filter subtracts in.
trait Lane: Copy {
    const SIZE: usize;
    fn load(bytes: &[u8]) -> Self;
    fn store(self, bytes: &mut [u8]);
    fn sub(self, other: Self) -> Self;
    fn add(self, other: Self) -> Self;
    fn from_u64(v: u64) -> Self;
    fn to_u64(self) -> u64;
}

macro_rules! lane {
    ($($t:ty),*) => {$(
        impl Lane for $t {
            const SIZE: usize = size_of::<$t>();
            #[inline(always)]
            fn load(bytes: &[u8]) -> Self {
                <$t>::from_ne_bytes(bytes.try_into().unwrap())
            }
            #[inline(always)]
            fn store(self, bytes: &mut [u8]) {
                bytes.copy_from_slice(&self.to_ne_bytes());
            }
            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                self.wrapping_sub(other)
            }
            #[inline(always)]
            fn add(self, other: Self) -> Self {
                self.wrapping_add(other)
            }
            #[inline(always)]
            fn from_u64(v: u64) -> Self {
                v as $t
            }
            #[inline(always)]
            fn to_u64(self) -> u64 {
                self as u64
            }
        }
    )*};
}

lane!(u8, u16, u32, u64);

/// Filters `input` into `output`; `previous` carries the last sample of the
/// previous chunk for delta, and the new last sample is returned.
fn forward(input: &[u8], output: &mut [u8], size: usize, filter: Prefilter, previous: u64) -> u64 {
    match size {
        1 => forward_lanes::<u8>(input, output, filter, previous),
        2 => forward_lanes::<u16>(input, output, filter, previous),
        4 => forward_lanes::<u32>(input, output, filter, previous),
        8 => forward_lanes::<u64>(input, output, filter, previous),
        _ => unreachable!("tile elements are 1, 2, 4 or 8 bytes"),
    }
}

fn inverse(input: &[u8], output: &mut [u8], size: usize, filter: Prefilter, previous: u64) -> u64 {
    match size {
        1 => inverse_lanes::<u8>(input, output, filter, previous),
        2 => inverse_lanes::<u16>(input, output, filter, previous),
        4 => inverse_lanes::<u32>(input, output, filter, previous),
        8 => inverse_lanes::<u64>(input, output, filter, previous),
        _ => unreachable!("tile elements are 1, 2, 4 or 8 bytes"),
    }
}

fn forward_lanes<L: Lane>(
    input: &[u8],
    output: &mut [u8],
    filter: Prefilter,
    previous: u64,
) -> u64 {
    let n = input.len() / L::SIZE;
    let mut last = L::from_u64(previous);
    if !filter.shuffle() {
        for (src, dst) in input
            .chunks_exact(L::SIZE)
            .zip(output.chunks_exact_mut(L::SIZE))
        {
            let v = L::load(src);
            v.sub(last).store(dst);
            last = v;
        }
        return last.to_u64();
    }
    let mut lane = [0u8; 8];
    for (i, src) in input.chunks_exact(L::SIZE).enumerate() {
        let mut v = L::load(src);
        if filter.delta() {
            let d = v.sub(last);
            last = v;
            v = d;
        }
        v.store(&mut lane[..L::SIZE]);
        for (b, &byte) in lane[..L::SIZE].iter().enumerate() {
            output[b * n + i] = byte;
        }
    }
    last.to_u64()
}

fn inverse_lanes<L: Lane>(
    input: &[u8],
    output: &mut [u8],
    filter: Prefilter,
    previous: u64,
) -> u64 {
    let n = input.len() / L::SIZE;
    let mut last = L::from_u64(previous);
    if !filter.shuffle() {
        for (src, dst) in input
            .chunks_exact(L::SIZE)
            .zip(output.chunks_exact_mut(L::SIZE))
        {
            last = last.add(L::load(src));
            last.store(dst);
        }
        return last.to_u64();
    }
    let mut lane = [0u8; 8];
    for (i, dst) in output.chunks_exact_mut(L::SIZE).enumerate() {
        for (b, byte) in lane[..L::SIZE].iter_mut().enumerate() {
            *byte = input[b * n + i];
        }
        let mut v = L::load(&lane[..L::SIZE]);
        if filter.delta() {
            v = last.add(v);
            last = v;
        }
        v.store(dst);
    }
    last.to_u64()
}

/// A 64-bit checksum after xxHash64's structure (four lanes of 8-byte
/// multiply-rotate rounds and an avalanche). It detects corruption; it is
/// not a cryptographic hash and does not reproduce XXH64's digests.
pub(crate) fn hash64(bytes: &[u8]) -> u64 {
    const P1: u64 = 0x9E37_79B1_85EB_CA87;
    const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    const P3: u64 = 0x1656_67B1_9E37_79F9;
    const P4: u64 = 0x85EB_CA77_C2B2_AE63;
    const P5: u64 = 0x27D4_EB2F_1656_67C5;
    #[inline(always)]
    fn round(acc: u64, input: u64) -> u64 {
        acc.wrapping_add(input.wrapping_mul(P2))
            .rotate_left(31)
            .wrapping_mul(P1)
    }
    let mut acc = [P1.wrapping_add(P2), P2, 0, 0u64.wrapping_sub(P1)];
    let (stripes, rest) = bytes.as_chunks::<32>();
    for stripe in stripes {
        for (lane, b) in acc.iter_mut().zip(stripe.as_chunks::<8>().0) {
            *lane = round(*lane, u64::from_le_bytes(*b));
        }
    }
    let mut h = acc[0]
        .rotate_left(1)
        .wrapping_add(acc[1].rotate_left(7))
        .wrapping_add(acc[2].rotate_left(12))
        .wrapping_add(acc[3].rotate_left(18));
    for lane in acc {
        h = (h ^ round(0, lane)).wrapping_mul(P1).wrapping_add(P4);
    }
    h = h.wrapping_add(bytes.len() as u64);
    let (words, tail) = rest.as_chunks::<8>();
    for b in words {
        h = (h ^ round(0, u64::from_le_bytes(*b)))
            .rotate_left(27)
            .wrapping_mul(P1)
            .wrapping_add(P4);
    }
    for &b in tail {
        h = (h ^ u64::from(b).wrapping_mul(P5))
            .rotate_left(11)
            .wrapping_mul(P1);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(P2);
    h ^= h >> 29;
    h = h.wrapping_mul(P3);
    h ^ (h >> 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip<T: TileElement + std::fmt::Debug>(data: &[T]) {
        for filter in Prefilter::ALL {
            let mut blob = Vec::new();
            encode_tile(data, filter, &mut blob);
            let mut back = vec![T::default(); 3];
            decode_tile(&blob, &mut back).unwrap();
            assert!(back == data, "{filter:?} {}", std::any::type_name::<T>());
        }
    }

    #[test]
    fn every_type_and_filter_round_trips_across_chunks() {
        // More than one chunk, and a length that leaves a partial chunk.
        let n = CHUNK_BYTES / 2 + 12_345;
        let f: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01).sin() * 100.0 + 250.0)
            .collect();
        roundtrip(&f);
        let f64s: Vec<f64> = f.iter().map(|&v| f64::from(v) * 1.5).collect();
        roundtrip(&f64s);
        let mut special = f.clone();
        special[7] = f32::NAN;
        special[8] = -0.0;
        special[9] = f32::INFINITY;
        special[10] = -3.0e38;
        let mut blob = Vec::new();
        encode_tile(&special, Prefilter::DeltaShuffle, &mut blob);
        let mut back: Vec<f32> = Vec::new();
        decode_tile(&blob, &mut back).unwrap();
        assert!(
            back.iter()
                .zip(&special)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        roundtrip(&(0..n * 2).map(|i| (i % 9) as u8).collect::<Vec<_>>());
        roundtrip(
            &(0..n)
                .map(|i| (i as i16).wrapping_mul(7))
                .collect::<Vec<_>>(),
        );
        roundtrip(&(0..n).map(|i| (i / 3) as u32).collect::<Vec<_>>());
        roundtrip(&(0..n).map(|i| i as i64 - 99).collect::<Vec<_>>());
        roundtrip::<u16>(&[]);
        roundtrip(&[42u8]);
    }

    #[test]
    fn incompressible_chunks_are_stored() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let noise: Vec<u64> = (0..CHUNK_BYTES / 8 * 2)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state
            })
            .collect();
        let mut blob = Vec::new();
        encode_tile(&noise, Prefilter::None, &mut blob);
        assert!(blob.len() <= noise.len() * 8 + HEADER + 8);
        let mut back: Vec<u64> = Vec::new();
        decode_tile(&blob, &mut back).unwrap();
        assert_eq!(back, noise);
    }

    #[test]
    fn corruption_and_mismatches_are_detected() {
        let data: Vec<f32> = (0..50_000).map(|i| i as f32).collect();
        let mut blob = Vec::new();
        let key = BlobKey {
            layer: 3,
            tile: Some(TileId::new(4, 5)),
        };
        encode_with_key(&data, Prefilter::Shuffle, key, &mut blob);
        let mut back: Vec<f32> = Vec::new();
        decode_with_key(&blob, Some(key), &mut back).unwrap();
        assert_eq!(back, data);
        let other = BlobKey {
            layer: 3,
            tile: Some(TileId::new(5, 4)),
        };
        assert!(matches!(
            decode_with_key(&blob, Some(other), &mut back),
            Err(Error::Corrupt(_))
        ));
        let mut wrong_type: Vec<u32> = Vec::new();
        assert!(decode_with_key(&blob, None, &mut wrong_type).is_err());
        for position in [0, 9, 30, 70, blob.len() / 2, blob.len() - 1] {
            let mut bad = blob.clone();
            bad[position] ^= 0x10;
            assert!(decode_tile(&bad, &mut back).is_err(), "flip at {position}");
        }
        assert!(decode_tile(&blob[..blob.len() - 1], &mut back).is_err());
        assert!(decode_tile(&blob[..10], &mut back).is_err());
    }

    #[test]
    fn prefilters_shrink_typical_grids() {
        // A smooth float surface: shuffle should beat raw LZ4.
        let n = 512 * 512;
        let surface: Vec<f32> = (0..n)
            .map(|i| 800.0 + ((i % 512) as f32 * 0.037).sin() * 3.0 + (i / 512) as f32 * 0.01)
            .collect();
        let size = |filter| {
            let mut blob = Vec::new();
            encode_tile(&surface, filter, &mut blob);
            blob.len()
        };
        assert!(size(Prefilter::Shuffle) < size(Prefilter::None));
        // BFS distances: delta should beat raw LZ4.
        let distances: Vec<u32> = (0..n)
            .map(|i| ((i % 512) as u32).min(511 - (i % 512) as u32) + (i / 512) as u32 / 4)
            .collect();
        let size = |filter| {
            let mut blob = Vec::new();
            encode_tile(&distances, filter, &mut blob);
            blob.len()
        };
        assert!(size(Prefilter::Delta) < size(Prefilter::None));
    }

    #[test]
    fn hash_distinguishes_lengths_and_bytes() {
        let a = hash64(b"");
        let b = hash64(&[0]);
        let c = hash64(&[0, 0]);
        assert!(a != b && b != c && a != c);
        let long: Vec<u8> = (0..1000).map(|i| i as u8).collect();
        let mut flipped = long.clone();
        flipped[999] ^= 1;
        assert_ne!(hash64(&long), hash64(&flipped));
        flipped = long.clone();
        flipped[0] ^= 0x80;
        assert_ne!(hash64(&long), hash64(&flipped));
    }
}
