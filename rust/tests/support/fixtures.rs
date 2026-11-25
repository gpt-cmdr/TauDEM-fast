// SPDX-License-Identifier: GPL-3.0-only
//! The tiling-boundary adversarial fixtures of `STREAMING.md` §6.1, placed
//! on the seams of a given tile size so that every feature straddles tile
//! edges and corners.
//!
//! [`suite`] returns every category for one tile size; [`large_suite`]
//! adds the AreaD8 ramp whose mainstem passes 2^24 cells (17.6 M cells, for
//! release builds). Sizes are deliberately not multiples of the tile size.

use taudem_rs::{DIRECTION_NODATA, Grid};

/// The tile sizes §6.1 requires, including a prime.
pub const TILE_SIZES: [usize; 3] = [16, 17, 32];
/// The thread counts §6.1 requires.
pub const THREADS: [usize; 2] = [1, 8];
/// The nodata value of fixture DEMs.
pub const NODATA: f32 = -9999.0;

/// The §6.1 categories.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Category {
    /// 1. Cross-seam and 4-corner pits, `-4`, `-depmask` outlets.
    CrossSeamPits,
    /// 2. Nodata cells on seams and tile corners.
    NodataAtSeams,
    /// 3. `dontCross` across seams.
    DontCross,
    /// 4. Multi-tile flats and multi-pass (secondary) flats.
    MultiTileFlats,
    /// 5. Meandering rivers, the `k == 4 && other == 0` quirk, `2^24` areas.
    RiversAndQuirks,
    /// 6. Non-multiple and degenerate geometries, sparse nodata tiles.
    Geometry,
}

/// What a fixture gives a stage.
#[derive(Clone, Debug)]
pub enum Input {
    /// A DEM, for PitRemove (and D8FlowDir and AreaD8 after it).
    Dem(Vec<f32>),
    /// D8 codes (`0..=8`, `DIRECTION_NODATA`), for AreaD8 alone, as
    /// `aread8 -p` reads them.
    Directions(Vec<u8>),
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: String,
    pub category: Category,
    pub width: usize,
    pub height: usize,
    pub input: Input,
    pub nodata: f32,
    /// `-depmask`: 1 fixes a cell's elevation.
    pub mask: Option<Vec<u8>>,
    /// `-4`.
    pub four_way: bool,
    /// AreaD8 contamination checks (`false` for `-nc`).
    pub check_contamination: bool,
    /// Projected cell size.
    pub spacing: (f64, f64),
}

impl Fixture {
    fn dem(name: impl Into<String>, category: Category, width: usize, dem: Vec<f32>) -> Self {
        assert_eq!(dem.len() % width, 0);
        Self {
            name: name.into(),
            category,
            width,
            height: dem.len() / width,
            input: Input::Dem(dem),
            nodata: NODATA,
            mask: None,
            four_way: false,
            check_contamination: true,
            spacing: (1.0, 1.0),
        }
    }

    fn directions(name: impl Into<String>, category: Category, width: usize, p: Vec<u8>) -> Self {
        let mut f = Self::dem(name, category, width, vec![0.0; p.len()]);
        f.input = Input::Directions(p);
        f
    }

    fn four_way(mut self) -> Self {
        self.four_way = true;
        self.name += "-4way";
        self
    }

    fn no_contamination(mut self) -> Self {
        self.check_contamination = false;
        self.name += "-nc";
        self
    }

    pub fn grid(&self) -> Grid {
        Grid::new(self.width, self.height).expect("fixture grid")
    }

    pub fn cells(&self) -> usize {
        self.width * self.height
    }

    pub fn dem_values(&self) -> Option<&[f32]> {
        match &self.input {
            Input::Dem(dem) => Some(dem),
            Input::Directions(_) => None,
        }
    }

    pub fn direction_values(&self) -> Option<&[u8]> {
        match &self.input {
            Input::Directions(p) => Some(p),
            Input::Dem(_) => None,
        }
    }
}

/// A mutable DEM.
struct Canvas {
    w: usize,
    h: usize,
    z: Vec<f32>,
}

impl Canvas {
    fn new(w: usize, h: usize, f: impl Fn(usize, usize) -> f32) -> Self {
        Self {
            w,
            h,
            z: (0..w * h).map(|i| f(i % w, i / w)).collect(),
        }
    }

    fn set(&mut self, x: usize, y: usize, v: f32) {
        assert!(
            x < self.w && y < self.h,
            "({x}, {y}) outside {}x{}",
            self.w,
            self.h
        );
        self.z[y * self.w + x] = v;
    }

    fn get(&self, x: usize, y: usize) -> f32 {
        self.z[y * self.w + x]
    }

    fn rect(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, v: f32) {
        for y in y0..y1 {
            for x in x0..x1 {
                self.set(x, y, v);
            }
        }
    }
}

/// Elevations quantized to 1/32, as the ELC LiDAR DEM's are (many ties).
fn quantize(v: f32) -> f32 {
    (v * 32.0).round() / 32.0
}

/// Rolling terrain with many small pits and quantized ties.
fn egg_crate(x: usize, y: usize, base: f32) -> f32 {
    let (x, y) = (x as f32, y as f32);
    quantize(base + 3.0 * (0.7 * x).sin() * (0.6 * y).cos() + 0.02 * (x + y))
}

/// Every fixture of every category for tile size `t` (at least 16).
pub fn suite(t: usize) -> Vec<Fixture> {
    assert!(t >= 16, "fixtures place features at least 16 cells apart");
    [
        cross_seam_pits(t),
        nodata_at_seams(t),
        dont_cross(t),
        multi_tile_flats(t),
        rivers_and_quirks(t),
        geometry(t),
    ]
    .concat()
}

/// The fixtures too large for debug builds: AreaD8's 2^24 boundary.
pub fn large_suite() -> Vec<Fixture> {
    // An odd width makes each row's inflow plus the cell itself odd, so
    // once the mainstem passes 2^24 every f32 addition rounds.
    let ramp = f32_precision_ramp(4201, 4200);
    vec![ramp.clone().no_contamination(), ramp]
}

// --- 1. Cross-seam and 4-corner pits -------------------------------------

/// A deep pit on the corner of four tiles whose only spill path is a
/// winding 1-cell diagonal channel with its saddle three tiles away.
/// Diagonal steps connect only in 8-way flooding, so with `-4` the pit fills
/// to the surrounding terrain instead.
fn corner_pit(t: usize) -> Fixture {
    let (w, h) = (6 * t + 5, 4 * t + 3);
    let mut c = Canvas::new(w, h, |x, y| egg_crate(x, y, 100.0));
    let corner = 2 * t;
    c.rect(corner - 2, corner - 2, corner + 2, corner + 2, 40.0);
    c.rect(corner - 1, corner - 1, corner + 1, corner + 1, 10.0);
    // Zigzag NE / SE legs of t/2 steps from the pit to the east edge.
    let leg = t / 2;
    let (mut x, mut y) = (corner + 2, corner + 2);
    let mut path = vec![(x, y)];
    let mut up = true;
    while x < w - 1 {
        for _ in 0..leg {
            if x == w - 1 {
                break;
            }
            x += 1;
            y = if up { y - 1 } else { y + 1 };
            path.push((x, y));
        }
        up = !up;
    }
    let saddle = path
        .iter()
        .position(|&(x, _)| x >= corner + 3 * t)
        .expect("saddle");
    let after = (path.len() - saddle - 1).max(1) as f32;
    for (n, &(x, y)) in path.iter().enumerate() {
        let z = match n.cmp(&saddle) {
            std::cmp::Ordering::Less => 60.0,
            std::cmp::Ordering::Equal => 70.0,
            std::cmp::Ordering::Greater => 65.0 - 45.0 * (n - saddle) as f32 / after,
        };
        c.set(x, y, z);
    }
    Fixture::dem(
        "c1-corner-pit-winding-spill",
        Category::CrossSeamPits,
        w,
        c.z,
    )
}

/// A pit straddling a horizontal seam drained by a 4-connected channel into
/// a neighboring tile, where a `-depmask` cell in the tile's interior is its
/// only outlet.
fn depmask_drain(t: usize) -> Fixture {
    let (w, h) = (4 * t + 7, 3 * t + 5);
    let mut c = Canvas::new(w, h, |x, y| egg_crate(x, y, 100.0));
    let x0 = t + t / 2;
    let seam = 2 * t;
    c.rect(x0, seam - 1, x0 + 2, seam + 1, 10.0);
    let x1 = 2 * t + t / 2;
    for x in x0 + 2..=x1 {
        c.set(x, seam, 30.0);
    }
    for y in seam + 1..seam + 4 {
        c.set(x1, y, 30.0);
    }
    c.set(x1, seam + 4, 25.0);
    let mut mask = vec![0u8; w * h];
    mask[(seam + 4) * w + x1] = 1;
    let mut f = Fixture::dem("c1-pit-drained-by-depmask", Category::CrossSeamPits, w, c.z);
    f.mask = Some(mask);
    f
}

fn cross_seam_pits(t: usize) -> Vec<Fixture> {
    let pit = corner_pit(t);
    let mask = depmask_drain(t);
    vec![pit.clone(), pit.four_way(), mask.clone(), mask.four_way()]
}

// --- 2. Nodata at tile edges and corners ---------------------------------

/// Rolling terrain with single nodata cells on tile edges, a nodata pair
/// diagonally across a 4-tile corner, and a walled pit whose only outlet is
/// a nodata cell diagonally across a tile corner (an outlet in 8-way
/// flooding only).
fn seam_nodata(t: usize) -> Fixture {
    let (w, h) = (5 * t + 3, 4 * t + 6);
    let mut c = Canvas::new(w, h, |x, y| egg_crate(x, y, 100.0));
    for (x, y) in [
        (t - 1, t / 2),
        (t, t + t / 2),
        (2 * t - 1, 2 * t - 1),
        (2 * t, 2 * t),
        (3 * t + t / 2, 3 * t - 1),
        (4 * t, 3 * t),
        (t + 3, 2 * t),
        (4 * t - 1, t + 5),
    ] {
        c.set(x, y, NODATA);
    }
    // The walled pit in the first cell of tile (3, 1).
    let (px, py) = (3 * t, t);
    for (x, y) in [
        (px + 1, py),
        (px, py + 1),
        (px + 1, py + 1),
        (px - 1, py),
        (px, py - 1),
        (px - 1, py + 1),
        (px + 1, py - 1),
    ] {
        c.set(x, y, 120.0);
    }
    c.set(px, py, 10.0);
    c.set(px - 1, py - 1, NODATA);
    Fixture::dem(
        "c2-nodata-on-seams-and-corners",
        Category::NodataAtSeams,
        w,
        c.z,
    )
}

/// Nodata runs lying exactly along a seam, one cell either side of it, with
/// flats that touch them (D8 nodata masks and AreaD8 contamination across
/// the seam).
fn seam_nodata_runs(t: usize) -> Fixture {
    let (w, h) = (4 * t + 9, 3 * t + 2);
    let mut c = Canvas::new(w, h, |x, y| {
        quantize(80.0 + 0.05 * x as f32 + ((y / 4) % 3) as f32 * 0.03125)
    });
    for y in t / 2..t + t / 2 {
        c.set(2 * t - 1, y, NODATA);
    }
    for x in t / 3..t + t / 3 {
        c.set(x, 2 * t, NODATA);
    }
    for y in 2 * t..3 * t {
        c.set(3 * t, y, NODATA);
    }
    Fixture::dem(
        "c2-nodata-runs-along-seams",
        Category::NodataAtSeams,
        w,
        c.z,
    )
}

fn nodata_at_seams(t: usize) -> Vec<Fixture> {
    let corners = seam_nodata(t);
    vec![corners.clone(), corners.four_way(), seam_nodata_runs(t)]
}

// --- 3. dontCross across seams -------------------------------------------

/// A plateau of diagonal ties across a 3 x 3 block of tiles: a checkerboard
/// of 1/32 bumps on a basin floor. The low cells connect only diagonally,
/// the flat passes assign diagonal directions, and later passes test
/// crossings against them on both sides of every seam.
fn checkerboard_ties(t: usize) -> Fixture {
    let (w, h) = (3 * t + 5, 3 * t + 3);
    let mut c = Canvas::new(w, h, |x, y| {
        if x < 2 || y < 2 || x + 2 >= w || y + 2 >= h {
            60.0
        } else {
            50.0 + 0.03125 * ((x + y) % 2) as f32
        }
    });
    // One outlet in the middle of the west wall.
    c.set(1, h / 2, 40.0);
    c.set(0, h / 2, 30.0);
    Fixture::dem("c3-checkerboard-diagonal-ties", Category::DontCross, w, c.z)
}

/// Diagonal terraces: `floor((x + 2y) / 3)` steps of 1/32 crossing every
/// seam at an angle, with a checkerboard on alternate terraces. Steep
/// diagonals meet flat-pass diagonals on the terraces' edges.
fn diagonal_terraces(t: usize) -> Fixture {
    let (w, h) = (3 * t + 7, 3 * t + 4);
    let c = Canvas::new(w, h, |x, y| {
        let step = (x + 2 * y) / 3;
        let checker = if step % 2 == 0 {
            ((x + y) % 2) as f32 * 0.03125
        } else {
            0.0
        };
        quantize(20.0 + step as f32 * 0.03125 + checker)
    });
    Fixture::dem("c3-diagonal-terraces", Category::DontCross, w, c.z)
}

fn dont_cross(t: usize) -> Vec<Fixture> {
    vec![checkerboard_ties(t), diagonal_terraces(t)]
}

// --- 4. Multi-tile flats --------------------------------------------------

/// Walls at 60 around a flat at 50 of the given shape, with an outlet of
/// two descending cells at `(x, y)` on the west or east edge.
fn walled_flat(
    w: usize,
    h: usize,
    flat: impl Fn(usize, usize) -> bool,
    outlet: (usize, usize),
) -> Canvas {
    let mut c = Canvas::new(w, h, |x, y| if flat(x, y) { 50.0 } else { 60.0 });
    let (x, y) = outlet;
    c.set(x, y, 40.0);
    let edge = if x < w / 2 { x - 1 } else { x + 1 };
    c.set(edge, y, 30.0);
    c
}

/// An L-shaped flat whose arms span three tiles each, drained at the far end.
fn l_flat(t: usize) -> Fixture {
    let (w, h) = (3 * t + 7, 3 * t + 5);
    let arm = t / 2;
    let flat = |x: usize, y: usize| {
        (x >= 2 && x < 2 + arm && y >= 2 && y < 3 * t + 2)
            || (y >= 3 * t + 2 - arm && y < 3 * t + 2 && x >= 2 && x < 3 * t + 5)
    };
    let c = walled_flat(w, h, flat, (3 * t + 5, 3 * t + 1));
    Fixture::dem("c4-l-shaped-flat", Category::MultiTileFlats, w, c.z)
}

/// A square spiral corridor two cells wide, spanning 3 x 3 tiles, drained
/// at its outer end.
fn spiral_flat(t: usize) -> Fixture {
    let (w, h) = (3 * t + 6, 3 * t + 6);
    let mut corridor = vec![false; w * h];
    let (mut x0, mut y0, mut x1, mut y1) = (2usize, 2usize, w - 3, h - 3);
    // Each lap: top row east, right column south, bottom row west, left
    // column north to just below the next lap; walls 2 cells thick.
    while x1 > x0 + 6 && y1 > y0 + 6 {
        for x in x0..=x1 {
            corridor[y0 * w + x] = true;
            corridor[(y0 + 1) * w + x] = true;
        }
        for y in y0..=y1 {
            corridor[y * w + x1] = true;
            corridor[y * w + x1 - 1] = true;
        }
        for x in x0..=x1 {
            corridor[y1 * w + x] = true;
            corridor[(y1 - 1) * w + x] = true;
        }
        for y in y0 + 4..=y1 {
            corridor[y * w + x0] = true;
            corridor[y * w + x0 + 1] = true;
        }
        // Connect the left column's top to the next lap's top row.
        for x in x0..x0 + 4 {
            corridor[(y0 + 4) * w + x] = true;
            corridor[(y0 + 5) * w + x] = true;
        }
        x0 += 4;
        y0 += 4;
        x1 -= 4;
        y1 -= 4;
    }
    let c = walled_flat(w, h, |x, y| corridor[y * w + x], (1, 2));
    Fixture::dem("c4-spiral-flat", Category::MultiTileFlats, w, c.z)
}

/// A flat whose east side borders, across a vertical seam, cells at the
/// same height next to nodata: their nodata direction seeds incfall at
/// level 2 in the neighboring tile.
fn flat_against_nodata(t: usize) -> Fixture {
    let (w, h) = (2 * t + 9, 2 * t + 7);
    let mut c = Canvas::new(w, h, |x, y| {
        if x >= 2 && x <= t && y >= 2 && y + 2 < h {
            50.0
        } else {
            60.0
        }
    });
    for y in 2..h - 2 {
        c.set(t + 1, y, NODATA);
    }
    Fixture::dem(
        "c4-flat-seeded-by-nodata-across-seam",
        Category::MultiTileFlats,
        w,
        c.z,
    )
}

/// A square flat walled on every side with its outlet in the west wall:
/// moving east from the outlet the distance from the outlet and from the
/// west wall grow together, so the combined Garbrecht–Martz surface ties
/// and later passes (on artificial elevations) are needed.
fn secondary_flats(t: usize) -> Fixture {
    let (w, h) = (3 * t + 4, 3 * t + 1);
    let c = walled_flat(
        w,
        h,
        |x, y| x >= 2 && x + 2 < w && y >= 2 && y + 2 < h,
        (1, h / 2),
    );
    Fixture::dem(
        "c4-plateau-with-secondary-flats",
        Category::MultiTileFlats,
        w,
        c.z,
    )
}

/// Two outlets on opposite walls and an off-centre high block: the flat
/// splits and secondary flats form across several seams.
fn twin_outlet_plateau(t: usize) -> Fixture {
    let (w, h) = (4 * t + 3, 3 * t + 5);
    let mut c = walled_flat(
        w,
        h,
        |x, y| x >= 2 && x + 2 < w && y >= 2 && y + 2 < h,
        (1, h / 3),
    );
    c.set(w - 2, 2 * h / 3, 40.0);
    c.set(w - 1, 2 * h / 3, 30.0);
    c.rect(t + 3, t - 2, t + 6, 2 * t + 1, 60.0);
    Fixture::dem("c4-twin-outlet-plateau", Category::MultiTileFlats, w, c.z)
}

fn multi_tile_flats(t: usize) -> Vec<Fixture> {
    vec![
        l_flat(t),
        spiral_flat(t),
        flat_against_nodata(t),
        secondary_flats(t),
        twin_outlet_plateau(t),
    ]
}

// --- 5. Rivers and quirks -------------------------------------------------

/// A river that leaves tile A for B, turns south into C, crosses the
/// A/B/C corner diagonally back into A, returns to B, and leaves for the
/// west edge, so the tile graph has the cycle A → B → C → A → B. Terrain
/// rises away from the channel; a rim two cells wide drains to the edges so
/// edge contamination does not reach the river.
fn meandering_river(t: usize) -> Fixture {
    let (w, h) = (3 * t + 5, 3 * t + 4);
    let half = t / 2;
    let mut path: Vec<(usize, usize)> = Vec::new();
    let mut walk = |to: (usize, usize)| {
        let &(mut x, mut y) = path.last().unwrap_or(&to);
        if path.is_empty() {
            path.push(to);
            return;
        }
        while (x, y) != to {
            x = (x as isize + (to.0 as isize - x as isize).signum()) as usize;
            y = (y as isize + (to.1 as isize - y as isize).signum()) as usize;
            path.push((x, y));
        }
    };
    let (a, b) = (t, 2 * t);
    walk((a + 3, a + half));
    walk((b + half, a + half));
    walk((b + half, b + half));
    walk((b + 2, b + half));
    walk((b + 2, b + 2));
    walk((b - t / 4, b - t / 4));
    walk((b + t / 4, b - t / 4));
    walk((b + t / 4, a + half + 2));
    walk((3, a + half + 2));
    let mut dist = vec![u32::MAX; w * h];
    let mut queue = std::collections::VecDeque::new();
    let n = path.len() as f32;
    let mut channel = vec![f32::NAN; w * h];
    for (k, &(x, y)) in path.iter().enumerate() {
        let i = y * w + x;
        channel[i] = 290.0 - 270.0 * k as f32 / n;
        dist[i] = 0;
        queue.push_back(i);
    }
    while let Some(i) = queue.pop_front() {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for (dx, dy) in [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ] {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            if dist[j] == u32::MAX {
                dist[j] = dist[i] + 1;
                channel[j] = channel[i];
                queue.push_back(j);
            }
        }
    }
    let c = Canvas::new(w, h, |x, y| {
        let rim = x.min(y).min(w - 1 - x).min(h - 1 - y);
        let i = y * w + x;
        if rim < 2 {
            1.0 + rim as f32
        } else {
            quantize(channel[i] + 2.0 * dist[i] as f32)
        }
    });
    Fixture::dem(
        "c5-meandering-river-tile-cycle",
        Category::RiversAndQuirks,
        w,
        c.z,
    )
}

/// D8 codes flowing east in every row with unresolved cells (0) on tile
/// corners: AreaD8 counts a NW neighbor with code 0 as an inflow that never
/// arrives (`k == 4 && other == 0`, `src/area.rs`), so the cell SE of each,
/// in the diagonally adjacent tile, and everything downstream stay nodata.
fn corner_zero_directions(t: usize) -> Fixture {
    let (w, h) = (3 * t + 5, 3 * t + 2);
    let mut p = vec![1u8; w * h];
    for x in 0..w {
        p[x] = DIRECTION_NODATA;
        p[(h - 1) * w + x] = DIRECTION_NODATA;
    }
    for y in 0..h {
        p[y * w] = DIRECTION_NODATA;
        p[y * w + w - 1] = DIRECTION_NODATA;
        // Column 1 drains west, so the rows downstream of it start clean
        // of edge contamination.
        if y > 0 && y + 1 < h {
            p[y * w + 1] = 5;
        }
    }
    for (x, y) in [
        (t - 1, t - 1),
        (2 * t - 1, t - 1),
        (t - 1, 2 * t - 1),
        (2 * t - 1, 2 * t - 1),
        (2 * t, 3 * t - 1),
    ] {
        p[y * w + x] = 0;
    }
    // Southward flow in one column across seams, and a cell flowing into a
    // zero-coded cell.
    for y in 1..h - 1 {
        p[y * w + t + 3] = 7;
    }
    p[(t + 4) * w + 2 * t + 2] = 4;
    Fixture::directions(
        "c5-corner-zero-blocks-diagonal-tile",
        Category::RiversAndQuirks,
        w,
        p,
    )
}

/// D8 codes of a `w x h` basin draining into one mainstem column (column 2,
/// flowing south) whose area exceeds 2^24 cells, where `f32` accumulation
/// stops being associative. The ring one cell inside the edge drains
/// outward so that contamination stays off the mainstem.
fn f32_precision_ramp(w: usize, h: usize) -> Fixture {
    let mut p = vec![5u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            p[i] = if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                DIRECTION_NODATA
            } else if y == 1 {
                3
            } else if y == h - 2 {
                7
            } else if x == 1 {
                5
            } else if x == w - 2 {
                1
            } else if x == 2 {
                7
            } else {
                5
            };
        }
    }
    Fixture::directions(
        format!("c5-f32-precision-ramp-{w}x{h}"),
        Category::RiversAndQuirks,
        w,
        p,
    )
}

fn rivers_and_quirks(t: usize) -> Vec<Fixture> {
    let river = meandering_river(t);
    let zeros = corner_zero_directions(t);
    // A ramp of a few thousand cells keeps the same shape in the default suite.
    let ramp = f32_precision_ramp(3 * t + 5, 2 * t + 7);
    vec![
        river.clone(),
        river.no_contamination(),
        zeros.clone(),
        zeros.no_contamination(),
        ramp,
    ]
}

// --- 6. Geometry ------------------------------------------------------------

fn terrain(name: &str, w: usize, h: usize) -> Fixture {
    let c = Canvas::new(w, h, |x, y| egg_crate(x, y, 100.0));
    Fixture::dem(format!("c6-{name}-{w}x{h}"), Category::Geometry, w, c.z)
}

/// Rolling terrain with some tiles entirely nodata (sparse tiles), a
/// nodata run aligned to a seam, and nodata in the last partial tile.
fn sparse_tiles(t: usize) -> Fixture {
    let (w, h) = (5 * t + 3, 4 * t + 1);
    let mut c = Canvas::new(w, h, |x, y| egg_crate(x, y, 100.0));
    for (tx, ty) in [(1, 1), (3, 0), (2, 2), (4, 3)] {
        c.rect(
            tx * t,
            ty * t,
            ((tx + 1) * t).min(w),
            ((ty + 1) * t).min(h),
            NODATA,
        );
    }
    c.rect(5 * t, 0, w, h, NODATA);
    for x in 0..w {
        if c.get(x, 2 * t) != NODATA && x % 3 != 0 {
            c.set(x, 2 * t, NODATA);
        }
    }
    Fixture::dem("c6-sparse-nodata-tiles", Category::Geometry, w, c.z)
}

fn geometry(t: usize) -> Vec<Fixture> {
    let mut all_nodata = terrain("all-nodata", 3 * t, 2 * t);
    if let Input::Dem(dem) = &mut all_nodata.input {
        dem.fill(NODATA);
    }
    let mut odd = terrain("100-by-67", 100, 67);
    if let Input::Dem(dem) = &mut odd.input {
        dem[33 * 100 + 31] = NODATA;
        dem[64 * 100 + 95] = NODATA;
    }
    vec![
        odd,
        terrain("non-multiple", 3 * t + 1, 2 * t + 15),
        terrain("single-tile", t / 2 + 3, t / 2 + 1),
        terrain("one-row", 5 * t + 3, 1),
        terrain("one-column", 1, 5 * t + 3),
        terrain("two-rows", 5 * t + 3, 2),
        terrain("three-columns", 3, 4 * t + 1),
        sparse_tiles(t),
        all_nodata,
    ]
}
