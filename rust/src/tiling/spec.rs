// SPDX-License-Identifier: GPL-3.0-only
//! Tile geometry: the partition of a raster into tiles and their halos.

use super::{Error, Result};

/// D8 neighbor offsets, `k = 1..=8` from east counterclockwise, as in the kernels.
const DX: [i64; 9] = [0, 1, 1, 0, -1, -1, -1, 0, 1];
const DY: [i64; 9] = [0, 0, -1, -1, -1, 0, 1, 1, 1];

/// A tile's column and row in the grid of tiles. Tiles order row-major:
/// by `ty`, then `tx`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TileId {
    pub tx: u32,
    pub ty: u32,
}

impl TileId {
    pub const fn new(tx: u32, ty: u32) -> Self {
        Self { tx, ty }
    }
}

impl Ord for TileId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.ty, self.tx).cmp(&(other.ty, other.tx))
    }
}

impl PartialOrd for TileId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// A rectangle of raster pixels: first column, first row, width and height.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

impl Rect {
    pub const fn new(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub const fn cells(&self) -> usize {
        self.width * self.height
    }

    /// One past the last column.
    pub const fn right(&self) -> usize {
        self.x + self.width
    }

    /// One past the last row.
    pub const fn bottom(&self) -> usize {
        self.y + self.height
    }

    pub const fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.x && x < self.right() && y >= self.y && y < self.bottom()
    }

    pub fn terrio_window(&self) -> terrio::Window {
        terrio::Window::new(self.x, self.y, self.width, self.height)
    }
}

/// A tile with `radius` cells of halo on every side. The buffer always has
/// the full halo; [`Self::window`] is the part inside the raster, and the
/// pads are the cells of the buffer outside it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HaloWindow {
    /// The tile's own cells.
    pub core: Rect,
    /// The raster pixels the buffer covers: the core and halo, clamped to the raster.
    pub window: Rect,
    pub radius: usize,
    /// Buffer columns left of `window` (outside the raster).
    pub pad_left: usize,
    /// Buffer rows above `window` (outside the raster).
    pub pad_top: usize,
    pub pad_right: usize,
    pub pad_bottom: usize,
}

impl HaloWindow {
    pub const fn buffer_width(&self) -> usize {
        self.core.width + 2 * self.radius
    }

    pub const fn buffer_height(&self) -> usize {
        self.core.height + 2 * self.radius
    }

    pub const fn buffer_cells(&self) -> usize {
        self.buffer_width() * self.buffer_height()
    }

    /// The buffer index of core cell `(lx, ly)`; core cells start at
    /// `(radius, radius)` in the buffer.
    #[inline]
    pub fn index(&self, lx: usize, ly: usize) -> u32 {
        debug_assert!(lx < self.core.width && ly < self.core.height);
        ((ly + self.radius) * self.buffer_width() + lx + self.radius) as u32
    }

    /// Copies `window_data` (the rows of [`Self::window`]) into `buffer`,
    /// sized to the whole halo buffer, with `fill` outside the raster.
    pub fn pad_into<T: Copy>(&self, window_data: &[T], fill: T, buffer: &mut Vec<T>) -> Result<()> {
        if window_data.len() != self.window.cells() {
            return Err(Error::Invalid(format!(
                "{} values for a {} x {} window",
                window_data.len(),
                self.window.width,
                self.window.height
            )));
        }
        let width = self.buffer_width();
        buffer.clear();
        buffer.resize(self.buffer_cells(), fill);
        for (row, values) in window_data.chunks_exact(self.window.width).enumerate() {
            let start = (row + self.pad_top) * width + self.pad_left;
            buffer[start..start + values.len()].copy_from_slice(values);
        }
        Ok(())
    }

    /// Copies the core cells of a halo buffer into `core`, row-major.
    pub fn core_from_buffer<T: Copy>(&self, buffer: &[T], core: &mut Vec<T>) -> Result<()> {
        if buffer.len() != self.buffer_cells() {
            return Err(Error::Invalid(format!(
                "{} values for a halo buffer of {} cells",
                buffer.len(),
                self.buffer_cells()
            )));
        }
        core.clear();
        core.reserve(self.core.cells());
        for ly in 0..self.core.height {
            let start = self.index(0, ly) as usize;
            core.extend_from_slice(&buffer[start..start + self.core.width]);
        }
        Ok(())
    }
}

/// The partition of a `raster_width × raster_height` raster into tiles of
/// `tile_width × tile_height`, the last column and row of tiles cut at the
/// raster's right and bottom edges. Raster dimensions are 64-bit; indices
/// inside a tile, including a 1-cell halo, fit in `u32`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct TileSpec {
    raster_width: usize,
    raster_height: usize,
    tile_width: usize,
    tile_height: usize,
    tiles_x: u32,
    tiles_y: u32,
}

impl TileSpec {
    /// Exact tile dimensions, clamped to the raster. Errors for empty rasters
    /// or tiles, a tile whose 1-cell halo buffer has more than `u32::MAX`
    /// cells, and more than `u32::MAX` tiles across or down.
    pub fn new(
        raster_width: usize,
        raster_height: usize,
        tile_width: usize,
        tile_height: usize,
    ) -> Result<Self> {
        if raster_width == 0 || raster_height == 0 {
            return Err(Error::Invalid("the raster is empty".into()));
        }
        if raster_width.checked_mul(raster_height).is_none() {
            return Err(Error::Invalid(
                "the raster has more cells than usize holds".into(),
            ));
        }
        if tile_width == 0 || tile_height == 0 {
            return Err(Error::Invalid("tile dimensions must be positive".into()));
        }
        let tile_width = tile_width.min(raster_width);
        let tile_height = tile_height.min(raster_height);
        let halo_cells = (tile_width as u128 + 2) * (tile_height as u128 + 2);
        if halo_cells > u128::from(u32::MAX) {
            return Err(Error::Invalid(format!(
                "a {tile_width} x {tile_height} tile with its halo exceeds the 32-bit tile index"
            )));
        }
        let tiles_x = u32::try_from(raster_width.div_ceil(tile_width))
            .map_err(|_| Error::Invalid("too many tiles across".into()))?;
        let tiles_y = u32::try_from(raster_height.div_ceil(tile_height))
            .map_err(|_| Error::Invalid("too many tiles down".into()))?;
        Ok(Self {
            raster_width,
            raster_height,
            tile_width,
            tile_height,
            tiles_x,
            tiles_y,
        })
    }

    /// Square tiles of about `target` cells a side, aligned to the input's
    /// native blocks when they are known: a tiled input gets the nearest
    /// multiple of its block size (at least one block); a striped input
    /// (`block width >= raster width`) keeps `target` columns and gets the
    /// nearest multiple of its strip height.
    pub fn aligned(
        raster_width: usize,
        raster_height: usize,
        target: usize,
        native_block: Option<(usize, usize)>,
    ) -> Result<Self> {
        let target = target.max(1);
        let nearest = |block: usize| (target + block / 2) / block * block;
        let (tile_width, tile_height) = match native_block {
            Some((bw, bh)) if bw > 0 && bh > 0 && bw >= raster_width => {
                (target, nearest(bh).max(bh))
            }
            Some((bw, bh)) if bw > 0 && bh > 0 => (nearest(bw).max(bw), nearest(bh).max(bh)),
            _ => (target, target),
        };
        Self::new(raster_width, raster_height, tile_width, tile_height)
    }

    pub const fn raster_width(&self) -> usize {
        self.raster_width
    }

    pub const fn raster_height(&self) -> usize {
        self.raster_height
    }

    /// Raster cells, as 64 bits.
    pub const fn cells(&self) -> u64 {
        self.raster_width as u64 * self.raster_height as u64
    }

    /// The nominal tile size; edge tiles may be smaller ([`Self::rect`]).
    pub const fn tile_size(&self) -> (usize, usize) {
        (self.tile_width, self.tile_height)
    }

    pub const fn tiles_x(&self) -> u32 {
        self.tiles_x
    }

    pub const fn tiles_y(&self) -> u32 {
        self.tiles_y
    }

    pub const fn tile_count(&self) -> usize {
        self.tiles_x as usize * self.tiles_y as usize
    }

    pub const fn contains(&self, id: TileId) -> bool {
        id.tx < self.tiles_x && id.ty < self.tiles_y
    }

    /// Whether every tile boundary falls on a boundary of `block` (columns
    /// and rows), so tiles can be written as whole output blocks.
    pub const fn is_aligned_to(&self, block_width: usize, block_height: usize) -> bool {
        self.tile_width.is_multiple_of(block_width) && self.tile_height.is_multiple_of(block_height)
    }

    /// The dense, row-major index of a tile, `0..tile_count()`.
    #[inline]
    pub const fn index(&self, id: TileId) -> usize {
        id.ty as usize * self.tiles_x as usize + id.tx as usize
    }

    /// The tile at a dense index ([`Self::index`]).
    #[inline]
    pub const fn id_at(&self, index: usize) -> TileId {
        TileId {
            tx: (index % self.tiles_x as usize) as u32,
            ty: (index / self.tiles_x as usize) as u32,
        }
    }

    /// The tile's own cells, clamped at the raster's right and bottom edges.
    pub fn rect(&self, id: TileId) -> Rect {
        assert!(self.contains(id), "no tile {id:?} in {self:?}");
        let x = id.tx as usize * self.tile_width;
        let y = id.ty as usize * self.tile_height;
        Rect::new(
            x,
            y,
            self.tile_width.min(self.raster_width - x),
            self.tile_height.min(self.raster_height - y),
        )
    }

    /// The tile with `radius` halo cells on each side, the read window
    /// clamped to the raster, and the pads that fall outside it.
    pub fn halo(&self, id: TileId, radius: usize) -> HaloWindow {
        let core = self.rect(id);
        let x0 = core.x.saturating_sub(radius);
        let y0 = core.y.saturating_sub(radius);
        let x1 = (core.right() + radius).min(self.raster_width);
        let y1 = (core.bottom() + radius).min(self.raster_height);
        HaloWindow {
            core,
            window: Rect::new(x0, y0, x1 - x0, y1 - y0),
            radius,
            pad_left: radius - (core.x - x0),
            pad_top: radius - (core.y - y0),
            pad_right: radius - (x1 - core.right()),
            pad_bottom: radius - (y1 - core.bottom()),
        }
    }

    /// The tile with a 1-cell halo ([`Self::halo`]).
    pub fn halo_window(&self, id: TileId) -> HaloWindow {
        self.halo(id, 1)
    }

    /// The tile holding raster cell `(x, y)`.
    #[inline]
    pub fn tile_of(&self, x: usize, y: usize) -> TileId {
        debug_assert!(x < self.raster_width && y < self.raster_height);
        TileId::new((x / self.tile_width) as u32, (y / self.tile_height) as u32)
    }

    /// The tile holding raster cell `(x, y)` and the cell's row-major index
    /// inside that tile's core.
    #[inline]
    pub fn locate(&self, x: usize, y: usize) -> (TileId, u32) {
        let id = self.tile_of(x, y);
        (id, self.local_index(id, x, y))
    }

    /// The row-major index of raster cell `(x, y)` inside tile `id`'s core.
    #[inline]
    pub fn local_index(&self, id: TileId, x: usize, y: usize) -> u32 {
        let rect = self.rect(id);
        debug_assert!(rect.contains(x, y), "({x}, {y}) is not in tile {id:?}");
        ((y - rect.y) * rect.width + (x - rect.x)) as u32
    }

    /// The raster cell `(x, y)` of a tile-local core index.
    #[inline]
    pub fn global_coords(&self, id: TileId, local: u32) -> (usize, usize) {
        let rect = self.rect(id);
        let local = local as usize;
        debug_assert!(local < rect.cells());
        (rect.x + local % rect.width, rect.y + local / rect.width)
    }

    /// The row-major raster index of cell `(x, y)`, as 64 bits.
    #[inline]
    pub const fn global_index(&self, x: usize, y: usize) -> u64 {
        y as u64 * self.raster_width as u64 + x as u64
    }

    /// The neighboring tile in D8 direction `k` (1 east, counterclockwise to 8 southeast).
    pub fn neighbor(&self, id: TileId, k: usize) -> Option<TileId> {
        assert!((1..=8).contains(&k), "D8 direction {k} is not 1..=8");
        let tx = i64::from(id.tx) + DX[k];
        let ty = i64::from(id.ty) + DY[k];
        (tx >= 0 && ty >= 0 && tx < i64::from(self.tiles_x) && ty < i64::from(self.tiles_y))
            .then(|| TileId::new(tx as u32, ty as u32))
    }

    /// Every tile, row-major: the deterministic default order.
    pub fn row_major(&self) -> impl ExactSizeIterator<Item = TileId> + '_ {
        (0..self.tile_count()).map(|index| self.id_at(index))
    }

    /// Every tile in Morton (Z-order) for spatial locality; ties cannot occur.
    pub fn morton_order(&self) -> Vec<TileId> {
        let mut ids: Vec<_> = self.row_major().collect();
        ids.sort_by_key(|id| morton(id.tx, id.ty));
        ids
    }

    /// Anti-diagonals `tx + ty == d` for `d = 0, 1, ...`, each sorted by `ty`.
    /// Every tile's north and west neighbors are in earlier groups.
    pub fn wavefront_groups(&self) -> Vec<Vec<TileId>> {
        let diagonals = self.tiles_x as usize + self.tiles_y as usize - 1;
        let mut groups = vec![Vec::new(); diagonals];
        for id in self.row_major() {
            groups[id.tx as usize + id.ty as usize].push(id);
        }
        groups
    }
}

fn morton(x: u32, y: u32) -> u64 {
    fn spread(v: u32) -> u64 {
        let mut v = u64::from(v);
        v = (v | (v << 16)) & 0x0000_ffff_0000_ffff;
        v = (v | (v << 8)) & 0x00ff_00ff_00ff_00ff;
        v = (v | (v << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
        v = (v | (v << 2)) & 0x3333_3333_3333_3333;
        v = (v | (v << 1)) & 0x5555_5555_5555_5555;
        v
    }
    spread(x) | (spread(y) << 1)
}

/// Runs `visit` on every tile, each tile only after its west (`tx - 1`) and
/// north (`ty - 1`) neighbors finished, so it sees their final results: the
/// dependency of TauDEM's row-major `dontCross` rule, which reads directions
/// already assigned to the north and west. Tiles of a diagonal run in
/// parallel on the current Rayon pool's thread count of plain threads
/// ([`super::workers`]); a tile starts as soon as its two dependencies are
/// done rather than at a barrier.
///
/// After an error no further tile starts; the error of the first tile in
/// row-major order that failed is returned.
pub fn for_each_wavefront<E, F>(spec: &TileSpec, visit: F) -> std::result::Result<(), E>
where
    E: Send,
    F: Fn(TileId) -> std::result::Result<(), E> + Sync,
{
    let mut states = vec![(); super::workers::threads()];
    super::workers::wavefront(spec, &mut states, |_, id| visit(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn non_multiple_rasters_partition_exactly() {
        for (w, h, tw, th) in [
            (100, 67, 32, 32),
            (100, 67, 17, 16),
            (1, 1, 16, 16),
            (1, 50, 16, 16),
            (50, 1, 17, 17),
            (64, 64, 16, 16),
            (65, 33, 32, 32),
            (10, 10, 100, 3),
        ] {
            let spec = TileSpec::new(w, h, tw, th).unwrap();
            let mut seen = vec![0u8; w * h];
            for id in spec.row_major() {
                let rect = spec.rect(id);
                assert!(rect.width > 0 && rect.height > 0);
                assert!(rect.width <= spec.tile_size().0 && rect.height <= spec.tile_size().1);
                for y in rect.y..rect.bottom() {
                    for x in rect.x..rect.right() {
                        seen[y * w + x] += 1;
                        let (tile, local) = spec.locate(x, y);
                        assert_eq!(tile, id);
                        assert_eq!(spec.global_coords(id, local), (x, y));
                    }
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "{w}x{h} tiles {tw}x{th}");
            assert_eq!(spec.cells(), (w * h) as u64);
            // Only the last column and row of tiles are partial.
            let last = spec.rect(TileId::new(spec.tiles_x() - 1, spec.tiles_y() - 1));
            assert_eq!(last.right(), w);
            assert_eq!(last.bottom(), h);
        }
    }

    #[test]
    fn halo_windows_clamp_and_pad() {
        let spec = TileSpec::new(100, 67, 32, 32).unwrap();
        let corner = spec.halo_window(TileId::new(0, 0));
        assert_eq!(corner.window, Rect::new(0, 0, 33, 33));
        assert_eq!((corner.pad_left, corner.pad_top), (1, 1));
        assert_eq!((corner.pad_right, corner.pad_bottom), (0, 0));
        let inner = spec.halo_window(TileId::new(1, 1));
        assert_eq!(inner.window, Rect::new(31, 31, 34, 34));
        assert_eq!(inner.pad_left + inner.pad_top + inner.pad_right, 0);
        // The last tile is 4 x 3 cells; its halo pads right and bottom.
        let last = spec.halo_window(TileId::new(3, 2));
        assert_eq!(last.core, Rect::new(96, 64, 4, 3));
        assert_eq!(last.window, Rect::new(95, 63, 5, 4));
        assert_eq!((last.pad_right, last.pad_bottom), (1, 1));
        assert_eq!((last.buffer_width(), last.buffer_height()), (6, 5));

        // Padding a window and taking its core back round-trips.
        let raster: Vec<u32> = (0..100 * 67).collect();
        for id in spec.row_major() {
            let halo = spec.halo(id, 2);
            let w = halo.window;
            let data: Vec<u32> = (w.y..w.bottom())
                .flat_map(|y| (w.x..w.right()).map(move |x| (y * 100 + x) as u32))
                .collect();
            let mut buffer = Vec::new();
            halo.pad_into(&data, u32::MAX, &mut buffer).unwrap();
            let bw = halo.buffer_width() as i64;
            for (i, &v) in buffer.iter().enumerate() {
                let x = halo.core.x as i64 - 2 + i as i64 % bw;
                let y = halo.core.y as i64 - 2 + i as i64 / bw;
                let inside = (0..100).contains(&x) && (0..67).contains(&y);
                assert_eq!(
                    v,
                    if inside {
                        (y * 100 + x) as u32
                    } else {
                        u32::MAX
                    }
                );
            }
            let mut core = Vec::new();
            halo.core_from_buffer(&buffer, &mut core).unwrap();
            let rect = halo.core;
            let expected: Vec<u32> = (rect.y..rect.bottom())
                .flat_map(|y| raster[y * 100 + rect.x..y * 100 + rect.right()].to_vec())
                .collect();
            assert_eq!(core, expected);
        }
    }

    #[test]
    fn sixty_four_bit_rasters_and_u32_tile_limits() {
        // A raster of more than 2^32 cells is described without overflow.
        let spec = TileSpec::new(100_000, 50_000, 4096, 4096).unwrap();
        assert_eq!(spec.cells(), 5_000_000_000);
        assert_eq!(spec.global_index(99_999, 49_999), 4_999_999_999);
        let (id, local) = spec.locate(99_999, 49_999);
        assert_eq!(spec.global_coords(id, local), (99_999, 49_999));
        // A tile whose halo buffer overflows u32 is refused.
        assert!(TileSpec::new(100_000, 100_000, 65_534, 65_534).is_err());
        assert!(TileSpec::new(100_000, 100_000, 65_533, 65_533).is_ok());
        assert!(TileSpec::new(0, 5, 1, 1).is_err());
        assert!(TileSpec::new(5, 5, 0, 1).is_err());
    }

    #[test]
    fn alignment_follows_native_blocks() {
        let tiled = TileSpec::aligned(36_000, 31_635, 2048, Some((256, 256))).unwrap();
        assert_eq!(tiled.tile_size(), (2048, 2048));
        assert!(tiled.is_aligned_to(256, 256));
        let odd = TileSpec::aligned(36_000, 31_635, 2000, Some((512, 512))).unwrap();
        assert_eq!(odd.tile_size(), (2048, 2048));
        let small = TileSpec::aligned(36_000, 31_635, 100, Some((512, 256))).unwrap();
        assert_eq!(small.tile_size(), (512, 256));
        let striped = TileSpec::aligned(36_000, 31_635, 2048, Some((36_000, 3))).unwrap();
        assert_eq!(striped.tile_size(), (2048, 2049));
        let unknown = TileSpec::aligned(100, 67, 17, None).unwrap();
        assert_eq!(unknown.tile_size(), (17, 17));
        let clamped = TileSpec::aligned(100, 67, 4096, Some((256, 256))).unwrap();
        assert_eq!((clamped.tile_size(), clamped.tile_count()), ((100, 67), 1));
    }

    #[test]
    fn orders_are_deterministic_permutations() {
        let spec = TileSpec::new(100, 67, 17, 16).unwrap();
        let all: HashSet<_> = spec.row_major().collect();
        assert_eq!(all.len(), spec.tile_count());
        let rows: Vec<_> = spec.row_major().collect();
        let mut sorted = rows.clone();
        sorted.sort();
        assert_eq!(rows, sorted);
        let morton = spec.morton_order();
        assert_eq!(morton.iter().copied().collect::<HashSet<_>>(), all);
        assert_eq!(morton, spec.morton_order());
        assert_eq!(
            &morton[..4],
            &[
                TileId::new(0, 0),
                TileId::new(1, 0),
                TileId::new(0, 1),
                TileId::new(1, 1)
            ]
        );
        let groups = spec.wavefront_groups();
        assert_eq!(groups.len(), 6 + 5 - 1);
        let mut position = vec![0; spec.tile_count()];
        for (d, group) in groups.iter().enumerate() {
            for id in group {
                assert_eq!(id.tx as usize + id.ty as usize, d);
                position[spec.index(*id)] = d;
            }
        }
        for id in spec.row_major() {
            for k in [3, 5] {
                if let Some(n) = spec.neighbor(id, k) {
                    assert!(position[spec.index(n)] < position[spec.index(id)]);
                }
            }
        }
        assert_eq!(spec.neighbor(TileId::new(0, 0), 4), None);
        assert_eq!(spec.neighbor(TileId::new(0, 0), 8), Some(TileId::new(1, 1)));
    }

    #[test]
    fn wavefront_respects_north_and_west_dependencies() {
        let spec = TileSpec::new(200, 130, 17, 16).unwrap();
        for threads in [1, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let done: Vec<AtomicBool> = spec.row_major().map(|_| AtomicBool::new(false)).collect();
            let visited = std::sync::atomic::AtomicUsize::new(0);
            pool.install(|| {
                for_each_wavefront(&spec, |id| {
                    for k in [3, 5, 4] {
                        if let Some(n) = spec.neighbor(id, k) {
                            assert!(
                                done[spec.index(n)].load(Ordering::Acquire),
                                "{id:?} before {n:?}"
                            );
                        }
                    }
                    assert!(!done[spec.index(id)].swap(true, Ordering::AcqRel));
                    visited.fetch_add(1, Ordering::Relaxed);
                    Ok::<(), ()>(())
                })
            })
            .unwrap();
            assert_eq!(visited.into_inner(), spec.tile_count());
        }
    }

    #[test]
    fn wavefront_stops_and_reports_the_first_failure() {
        let spec = TileSpec::new(64, 64, 16, 16).unwrap();
        let started = std::sync::atomic::AtomicUsize::new(0);
        let result = for_each_wavefront(&spec, |id| {
            started.fetch_add(1, Ordering::Relaxed);
            if id == TileId::new(1, 1) {
                Err(id)
            } else {
                Ok(())
            }
        });
        assert_eq!(result, Err(TileId::new(1, 1)));
        // (2, 2), (3, 3) and every tile depending on (1, 1) never start.
        assert!(started.into_inner() < spec.tile_count());
    }
}
