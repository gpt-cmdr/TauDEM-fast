// SPDX-License-Identifier: GPL-3.0-only
//! In-memory PitRemove, D8FlowDir, and AreaD8 kernels for TauDEM.

mod area;
mod d8;
mod flood;
pub mod profile;

pub use area::area_d8;
pub use d8::{D8Result, d8_flow};
pub use flood::{MAX_TILE, pit_remove, pit_remove_with_tiles};

pub const FEL_NODATA: f32 = -3.0e38;
pub const DIRECTION_NODATA: u8 = u8::MAX;
pub const SCALAR_NODATA: f32 = -1.0;
pub const INVALID_DIRECTION: u8 = 254;

const DX: [isize; 9] = [0, 1, 1, 0, -1, -1, -1, 0, 1];
const DY: [isize; 9] = [0, 0, -1, -1, -1, 0, 1, 1, 1];
const ORDER: [usize; 8] = [1, 3, 5, 7, 2, 4, 6, 8];
const OPPOSITE: [u8; 9] = [0, 5, 6, 7, 8, 1, 2, 3, 4];

/// A nonempty, row-major grid with compact 32-bit work indices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Grid {
    width: usize,
    height: usize,
    cells: usize,
    offsets: [isize; 9],
}

impl Grid {
    pub fn width(self) -> usize {
        self.width
    }

    pub fn height(self) -> usize {
        self.height
    }

    pub fn cells(self) -> usize {
        self.cells
    }

    pub fn new(width: usize, height: usize) -> Option<Self> {
        let cells = width.checked_mul(height)?;
        if cells == 0 || cells > u32::MAX as usize || cells > isize::MAX as usize {
            return None;
        }
        Some(Self {
            width,
            height,
            cells,
            offsets: std::array::from_fn(|k| DX[k] + DY[k] * width as isize),
        })
    }

    /// Visits the in-bounds neighbors in D8 order, or every other neighbor for D4.
    #[inline]
    fn for_neighbors(self, i: usize, step: usize, mut visit: impl FnMut(usize, usize)) {
        let col = i % self.width;
        if self.interior(i, col) {
            for k in (1..9).step_by(step) {
                visit(k, self.offset(i, k));
            }
        } else {
            for k in (1..9).step_by(step) {
                if let Some(j) = self.neighbor(i, col, k) {
                    visit(k, j);
                }
            }
        }
    }

    #[inline]
    fn interior(self, i: usize, col: usize) -> bool {
        i >= self.width && i < self.cells - self.width && col > 0 && col + 1 < self.width
    }

    #[inline]
    fn offset(self, i: usize, k: usize) -> usize {
        i.wrapping_add_signed(self.offsets[k])
    }

    #[inline]
    fn neighbor(self, i: usize, col: usize, k: usize) -> Option<usize> {
        let x = col.wrapping_add_signed(DX[k]);
        let j = self.offset(i, k);
        (x < self.width && j < self.cells).then_some(j)
    }
}

#[inline]
pub(crate) fn missing(value: f32, nodata: f32) -> bool {
    !value.is_finite() || (value - nodata).abs() < 1.0e-5
}

/// Optional in-memory products retained alongside contributing area.
#[derive(Clone, Copy, Debug, Default)]
pub struct Retain {
    pub filled: bool,
    pub direction: bool,
    pub slope: bool,
}

#[derive(Debug)]
pub struct PipelineResult {
    pub filled: Option<Vec<f32>>,
    pub direction: Option<Vec<u8>>,
    pub slope: Option<Vec<f32>>,
    pub area: Vec<f32>,
}

/// Runs the three stages without serializing intermediate grids. 'mask' contains
/// one byte per cell, with 1 marking an elevation that must remain fixed.
pub fn pipeline(
    grid: Grid,
    mut elevations: Vec<f32>,
    options: PipelineOptions<'_>,
) -> PipelineResult {
    pit_remove(
        grid,
        &mut elevations,
        options.nodata,
        options.four_way,
        options.mask,
    );
    let flow = d8_flow(
        grid,
        &elevations,
        FEL_NODATA,
        options.spacing,
        options.retain.slope,
    );
    let filled = options.retain.filled.then_some(elevations);
    let area = area_d8(grid, &flow.direction, options.check_contamination);
    PipelineResult {
        filled,
        direction: options.retain.direction.then_some(flow.direction),
        slope: flow.slope,
        area,
    }
}

pub struct PipelineOptions<'a> {
    pub nodata: f32,
    pub four_way: bool,
    pub mask: Option<&'a [u8]>,
    /// One (width, height) pair for a projected grid, or one per row.
    pub spacing: &'a [(f64, f64)],
    pub check_contamination: bool,
    pub retain: Retain,
}
