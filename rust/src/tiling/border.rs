// SPDX-License-Identifier: GPL-3.0-only
//! Tile perimeters kept in RAM for every tile.

use std::sync::{RwLock, RwLockReadGuard};

use super::spec::{HaloWindow, TileId, TileSpec};
use super::{Error, Result};

/// A corner of a tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Corner {
    Nw,
    Ne,
    Sw,
    Se,
}

/// The perimeter cells of one tile: its top and bottom rows and its left and
/// right columns. Corner cells belong to a row and a column and are kept
/// equal in both.
#[derive(Clone, Debug, PartialEq)]
pub struct Border<T> {
    width: usize,
    height: usize,
    /// `top[width] | bottom[width] | left[height] | right[height]`.
    cells: Box<[T]>,
}

impl<T: Copy> Border<T> {
    fn new(width: usize, height: usize, fill: T) -> Self {
        Self {
            width,
            height,
            cells: vec![fill; 2 * (width + height)].into_boxed_slice(),
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Row 0, west to east.
    pub fn top(&self) -> &[T] {
        &self.cells[..self.width]
    }

    /// The last row, west to east.
    pub fn bottom(&self) -> &[T] {
        &self.cells[self.width..2 * self.width]
    }

    /// Column 0, north to south.
    pub fn left(&self) -> &[T] {
        &self.cells[2 * self.width..2 * self.width + self.height]
    }

    /// The last column, north to south.
    pub fn right(&self) -> &[T] {
        &self.cells[2 * self.width + self.height..]
    }

    pub fn corner(&self, corner: Corner) -> T {
        match corner {
            Corner::Nw => self.top()[0],
            Corner::Ne => self.top()[self.width - 1],
            Corner::Sw => self.bottom()[0],
            Corner::Se => self.bottom()[self.width - 1],
        }
    }

    /// Whether tile-local `(lx, ly)` is on the perimeter.
    pub fn on_perimeter(&self, lx: usize, ly: usize) -> bool {
        lx < self.width
            && ly < self.height
            && (lx == 0 || ly == 0 || lx + 1 == self.width || ly + 1 == self.height)
    }

    /// The perimeter cell at tile-local `(lx, ly)`.
    pub fn get(&self, lx: usize, ly: usize) -> Option<T> {
        if !self.on_perimeter(lx, ly) {
            return None;
        }
        Some(if ly == 0 {
            self.top()[lx]
        } else if ly + 1 == self.height {
            self.bottom()[lx]
        } else if lx == 0 {
            self.left()[ly]
        } else {
            self.right()[ly]
        })
    }

    /// Sets the perimeter cell at tile-local `(lx, ly)` in every strip that
    /// holds it; `false` when the cell is not on the perimeter.
    pub fn set(&mut self, lx: usize, ly: usize, value: T) -> bool {
        if !self.on_perimeter(lx, ly) {
            return false;
        }
        let (w, h) = (self.width, self.height);
        if ly == 0 {
            self.cells[lx] = value;
        }
        if ly + 1 == h {
            self.cells[w + lx] = value;
        }
        if lx == 0 {
            self.cells[2 * w + ly] = value;
        }
        if lx + 1 == w {
            self.cells[2 * w + h + ly] = value;
        }
        true
    }

    /// Copies the perimeter of a tile from `data`, whose rows are `stride`
    /// values apart with the tile's first cell at `origin`.
    fn copy_from(&mut self, data: &[T], stride: usize, origin: usize) {
        let (w, h) = (self.width, self.height);
        let row = |r: usize| &data[origin + r * stride..origin + r * stride + w];
        self.cells[..w].copy_from_slice(row(0));
        self.cells[w..2 * w].copy_from_slice(row(h - 1));
        for r in 0..h {
            self.cells[2 * w + r] = data[origin + r * stride];
            self.cells[2 * w + h + r] = data[origin + r * stride + w - 1];
        }
    }
}

/// The 1-cell ring around a tile, gathered from its neighbors' borders:
/// `north` and `south` run from the column west of the tile to the column
/// east of it (`width + 2` cells), `west` and `east` along the tile's rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Ring<T> {
    pub north: Vec<T>,
    pub south: Vec<T>,
    pub west: Vec<T>,
    pub east: Vec<T>,
}

/// The perimeter of every tile of a [`TileSpec`], always in RAM: about
/// `2 (T_w + T_h)` values per tile. Each tile has its own lock, so tiles
/// publish their borders in parallel while neighbors read theirs.
pub struct BorderTable<T> {
    spec: TileSpec,
    borders: Vec<RwLock<Border<T>>>,
}

impl<T: Copy + Send + Sync> BorderTable<T> {
    /// A table whose every perimeter cell holds `fill`.
    pub fn new(spec: TileSpec, fill: T) -> Self {
        let borders = spec
            .row_major()
            .map(|id| {
                let rect = spec.rect(id);
                RwLock::new(Border::new(rect.width, rect.height, fill))
            })
            .collect();
        Self { spec, borders }
    }

    pub fn spec(&self) -> &TileSpec {
        &self.spec
    }

    /// Bytes of perimeter values held.
    pub fn bytes(&self) -> usize {
        self.spec
            .row_major()
            .map(|id| {
                let r = self.spec.rect(id);
                2 * (r.width + r.height) * size_of::<T>()
            })
            .sum()
    }

    fn slot(&self, id: TileId) -> &RwLock<Border<T>> {
        assert!(self.spec.contains(id), "no tile {id:?}");
        &self.borders[self.spec.index(id)]
    }

    /// Tile `id`'s border, locked for reading.
    pub fn read(&self, id: TileId) -> RwLockReadGuard<'_, Border<T>> {
        self.slot(id).read().unwrap_or_else(|p| p.into_inner())
    }

    /// Changes tile `id`'s border under its write lock.
    pub fn update<R>(&self, id: TileId, f: impl FnOnce(&mut Border<T>) -> R) -> R {
        let mut border = self.slot(id).write().unwrap_or_else(|p| p.into_inner());
        f(&mut border)
    }

    /// Publishes the perimeter of tile `id` from its cells, row-major.
    pub fn set_from_tile(&self, id: TileId, tile: &[T]) -> Result<()> {
        let rect = self.spec.rect(id);
        if tile.len() != rect.cells() {
            return Err(Error::Invalid(format!(
                "{} values for tile {id:?} of {} x {}",
                tile.len(),
                rect.width,
                rect.height
            )));
        }
        self.update(id, |b| b.copy_from(tile, rect.width, 0));
        Ok(())
    }

    /// Publishes the perimeter of tile `id` from a halo buffer of it.
    pub fn set_from_halo(&self, id: TileId, halo: &HaloWindow, buffer: &[T]) -> Result<()> {
        if halo.core != self.spec.rect(id) || buffer.len() != halo.buffer_cells() {
            return Err(Error::Invalid(format!(
                "the halo buffer is not tile {id:?}'s"
            )));
        }
        let origin = halo.index(0, 0) as usize;
        self.update(id, |b| b.copy_from(buffer, halo.buffer_width(), origin));
        Ok(())
    }

    /// The value of raster cell `(x, y)` when it lies on its tile's perimeter.
    pub fn cell(&self, x: usize, y: usize) -> Option<T> {
        if x >= self.spec.raster_width() || y >= self.spec.raster_height() {
            return None;
        }
        let id = self.spec.tile_of(x, y);
        let rect = self.spec.rect(id);
        self.read(id).get(x - rect.x, y - rect.y)
    }

    /// The 1-cell ring around tile `id` from its eight neighbors' borders,
    /// with `outside` for cells outside the raster.
    pub fn ring(&self, id: TileId, outside: T) -> Ring<T> {
        let rect = self.spec.rect(id);
        let (w, h) = (rect.width, rect.height);
        let mut ring = Ring {
            north: vec![outside; w + 2],
            south: vec![outside; w + 2],
            west: vec![outside; h],
            east: vec![outside; h],
        };
        let n = |k| self.spec.neighbor(id, k);
        if let Some(t) = n(3) {
            ring.north[1..=w].copy_from_slice(self.read(t).bottom());
        }
        if let Some(t) = n(4) {
            ring.north[0] = self.read(t).corner(Corner::Se);
        }
        if let Some(t) = n(2) {
            ring.north[w + 1] = self.read(t).corner(Corner::Sw);
        }
        if let Some(t) = n(7) {
            ring.south[1..=w].copy_from_slice(self.read(t).top());
        }
        if let Some(t) = n(6) {
            ring.south[0] = self.read(t).corner(Corner::Ne);
        }
        if let Some(t) = n(8) {
            ring.south[w + 1] = self.read(t).corner(Corner::Nw);
        }
        if let Some(t) = n(5) {
            ring.west.copy_from_slice(self.read(t).right());
        }
        if let Some(t) = n(1) {
            ring.east.copy_from_slice(self.read(t).left());
        }
        ring
    }

    /// Writes the ring around tile `id` ([`Self::ring`]) into the outer
    /// cells of a radius-1 halo buffer of it, leaving the core untouched.
    pub fn fill_halo_ring(
        &self,
        id: TileId,
        halo: &HaloWindow,
        buffer: &mut [T],
        outside: T,
    ) -> Result<()> {
        if halo.radius != 1
            || halo.core != self.spec.rect(id)
            || buffer.len() != halo.buffer_cells()
        {
            return Err(Error::Invalid(format!(
                "the buffer is not a radius-1 halo of tile {id:?}"
            )));
        }
        let ring = self.ring(id, outside);
        let bw = halo.buffer_width();
        let bh = halo.buffer_height();
        buffer[..bw].copy_from_slice(&ring.north);
        buffer[(bh - 1) * bw..].copy_from_slice(&ring.south);
        for r in 0..halo.core.height {
            buffer[(r + 1) * bw] = ring.west[r];
            buffer[(r + 1) * bw + bw - 1] = ring.east[r];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(x: usize, y: usize) -> u32 {
        (y * 1000 + x) as u32
    }

    fn table(w: usize, h: usize, t: usize) -> BorderTable<u32> {
        let spec = TileSpec::new(w, h, t, t).unwrap();
        let table = BorderTable::new(spec, 0);
        for id in spec.row_major() {
            let r = spec.rect(id);
            let tile: Vec<u32> = (r.y..r.bottom())
                .flat_map(|y| (r.x..r.right()).map(move |x| value(x, y)))
                .collect();
            table.set_from_tile(id, &tile).unwrap();
        }
        table
    }

    #[test]
    fn perimeters_hold_edges_and_corners() {
        let t = table(100, 67, 17);
        let spec = *t.spec();
        for id in spec.row_major() {
            let r = spec.rect(id);
            let b = t.read(id);
            assert_eq!(
                b.top(),
                (r.x..r.right()).map(|x| value(x, r.y)).collect::<Vec<_>>()
            );
            assert_eq!(
                b.bottom(),
                (r.x..r.right())
                    .map(|x| value(x, r.bottom() - 1))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                b.left(),
                (r.y..r.bottom()).map(|y| value(r.x, y)).collect::<Vec<_>>()
            );
            assert_eq!(
                b.right(),
                (r.y..r.bottom())
                    .map(|y| value(r.right() - 1, y))
                    .collect::<Vec<_>>()
            );
            assert_eq!(b.corner(Corner::Se), value(r.right() - 1, r.bottom() - 1));
            assert_eq!(b.corner(Corner::Ne), value(r.right() - 1, r.y));
        }
        assert_eq!(t.cell(17, 0), Some(value(17, 0)));
        assert_eq!(t.cell(18, 18), None, "interior cell");
        assert_eq!(t.cell(99, 66), Some(value(99, 66)));
        assert!(t.bytes() > 0);
    }

    #[test]
    fn rings_match_the_raster_around_every_tile() {
        for (w, h, s) in [
            (100, 67, 17),
            (64, 64, 16),
            (33, 5, 32),
            (1, 40, 16),
            (40, 1, 16),
            (5, 5, 16),
        ] {
            let t = table(w, h, s);
            let spec = *t.spec();
            let at = |x: i64, y: i64| {
                if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                    u32::MAX
                } else {
                    value(x as usize, y as usize)
                }
            };
            for id in spec.row_major() {
                let r = spec.rect(id);
                let (x0, y0) = (r.x as i64, r.y as i64);
                let ring = t.ring(id, u32::MAX);
                let north: Vec<_> = (x0 - 1..=r.right() as i64).map(|x| at(x, y0 - 1)).collect();
                let south: Vec<_> = (x0 - 1..=r.right() as i64)
                    .map(|x| at(x, r.bottom() as i64))
                    .collect();
                let west: Vec<_> = (y0..r.bottom() as i64).map(|y| at(x0 - 1, y)).collect();
                let east: Vec<_> = (y0..r.bottom() as i64)
                    .map(|y| at(r.right() as i64, y))
                    .collect();
                assert_eq!(
                    ring,
                    Ring {
                        north,
                        south,
                        west,
                        east
                    },
                    "{w}x{h} tile {s}, {id:?}"
                );

                let halo = spec.halo_window(id);
                let mut buffer = vec![7u32; halo.buffer_cells()];
                t.fill_halo_ring(id, &halo, &mut buffer, u32::MAX).unwrap();
                let bw = halo.buffer_width() as i64;
                for (i, &v) in buffer.iter().enumerate() {
                    let (bx, by) = (i as i64 % bw, i as i64 / bw);
                    let core =
                        bx > 0 && by > 0 && bx < bw - 1 && by < halo.buffer_height() as i64 - 1;
                    assert_eq!(
                        v,
                        if core {
                            7
                        } else {
                            at(x0 - 1 + bx, y0 - 1 + by)
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn halo_buffers_publish_and_cells_update_consistently() {
        let spec = TileSpec::new(40, 40, 16, 16).unwrap();
        let t = BorderTable::new(spec, 0u8);
        let id = TileId::new(1, 1);
        let halo = spec.halo_window(id);
        let buffer: Vec<u8> = (0..halo.buffer_cells()).map(|i| (i % 251) as u8).collect();
        t.set_from_halo(id, &halo, &buffer).unwrap();
        assert_eq!(
            t.read(id).corner(Corner::Nw),
            buffer[halo.index(0, 0) as usize]
        );
        assert_eq!(
            t.read(id).corner(Corner::Se),
            buffer[halo.index(15, 15) as usize]
        );
        t.update(id, |b| {
            assert!(b.set(15, 0, 99));
            assert!(!b.set(5, 5, 1));
        });
        let b = t.read(id);
        assert_eq!(
            (b.top()[15], b.right()[0], b.corner(Corner::Ne)),
            (99, 99, 99)
        );
        assert!(t.set_from_tile(id, &[0; 3]).is_err());
    }
}
