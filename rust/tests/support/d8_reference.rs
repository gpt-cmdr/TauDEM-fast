// SPDX-License-Identifier: GPL-3.0-only
//! A plain transcription of `src/d8.rs` (steep pass, `dontCross`, and the
//! Garbrecht–Martz flat passes) that can ignore the crossing rule and
//! records where it applied. It exists to show that a fixture exercises
//! `dontCross` near tile seams: with the rule its directions must equal
//! `d8_flow`'s exactly (checked by the fixture tests), and without it they
//! must differ. It is slow and only for small fixtures.

use taudem_rs::DIRECTION_NODATA;

const DX: [isize; 9] = [0, 1, 1, 0, -1, -1, -1, 0, 1];
const DY: [isize; 9] = [0, 0, -1, -1, -1, 0, 1, 1, 1];
const ORDER: [usize; 8] = [1, 3, 5, 7, 2, 4, 6, 8];

/// The result of [`d8_reference`].
pub struct Trace {
    pub direction: Vec<u8>,
    /// Flat passes run (0 without flats).
    pub passes: u32,
    /// Cells where the crossing rule changed or blocked something: a steep
    /// direction re-chosen, an incfall seed refused or a BFS edge refused.
    pub crossing_cells: Vec<usize>,
}

fn missing(value: f32, nodata: f32) -> bool {
    !value.is_finite() || (value - nodata).abs() < 1.0e-5
}

fn scaled(delta: f32, factor: f64) -> f32 {
    (f64::from(delta) * factor) as f32
}

/// D8 directions of a projected grid with cell size `(dx, dy)`.
pub fn d8_reference(
    width: usize,
    height: usize,
    elevations: &[f32],
    nodata: f32,
    (dx, dy): (f64, f64),
    honor_crossing: bool,
) -> Trace {
    let cells = width * height;
    let diag = 1.0 / (dx * dx + dy * dy).sqrt();
    let scale = [
        0.0,
        1.0 / dx,
        diag,
        1.0 / dy,
        diag,
        1.0 / dx,
        diag,
        1.0 / dy,
        diag,
    ];
    let offset = |i: usize, k: usize| (i as isize + DX[k] + DY[k] * width as isize) as usize;
    let steepest = |heights: &[f32; 9], allowed: &dyn Fn(usize) -> bool| {
        let mut best = 0.0;
        let mut direction = 0;
        for k in ORDER {
            let slope = scaled(heights[0] - heights[k], scale[k]);
            if slope > best && allowed(k) {
                best = slope;
                direction = k as u8;
            }
        }
        direction
    };
    let crosses = |dirs: &[u8], i: usize, k: usize, before: usize| {
        if !honor_crossing {
            return false;
        }
        let (a, ac, b, bc) = match k {
            2 => (1, 4, 3, 8),
            4 => (3, 6, 5, 2),
            6 => (7, 4, 5, 8),
            8 => (1, 6, 7, 2),
            _ => return false,
        };
        let a = offset(i, a);
        let b = offset(i, b);
        (a < before && dirs[a] == ac) || (b < before && dirs[b] == bc)
    };
    let mut crossing_cells = Vec::new();
    let mut dirs = vec![DIRECTION_NODATA; cells];
    for y in 1..height.saturating_sub(1) {
        for x in 1..width.saturating_sub(1) {
            let i = y * width + x;
            let heights: [f32; 9] = std::array::from_fn(|k| elevations[offset(i, k)]);
            if heights.iter().any(|&z| missing(z, nodata)) {
                continue;
            }
            dirs[i] = steepest(&heights, &|_| true);
        }
    }
    let mut flats = Vec::new();
    for i in 0..cells {
        let k = dirs[i] as usize;
        if matches!(k, 2 | 4 | 6 | 8) && crosses(&dirs, i, k, i) {
            crossing_cells.push(i);
            let heights: [f32; 9] = std::array::from_fn(|k| elevations[offset(i, k)]);
            let snapshot = dirs.clone();
            dirs[i] = steepest(&heights, &|k| !crosses(&snapshot, i, k, i));
        }
        if dirs[i] == 0 {
            flats.push(i);
        }
    }
    let mut passes = 0;
    if flats.is_empty() {
        return Trace {
            direction: dirs,
            passes,
            crossing_cells,
        };
    }
    let mut artificial: Vec<f32> = Vec::new();
    let mut fall = vec![0u32; cells];
    let mut rise = vec![0u32; cells];
    let mut frontier = std::collections::VecDeque::new();
    loop {
        passes += 1;
        let elev: &[f32] = if artificial.is_empty() {
            elevations
        } else {
            &artificial
        };
        for &i in &flats {
            let mut seeded = false;
            for k in 1..9 {
                let j = offset(i, k);
                if (1..=8).contains(&dirs[j]) && elev[i] >= elev[j] {
                    if crosses(&dirs, i, k, cells) {
                        crossing_cells.push(i);
                    } else {
                        seeded = true;
                        break;
                    }
                }
            }
            if seeded {
                fall[i] = 1;
                frontier.push_back(i);
            }
        }
        for &i in &flats {
            if fall[i] == 0 {
                let mut seeded = false;
                for k in 1..9 {
                    let j = offset(i, k);
                    if dirs[j] != 0 && elev[i] == elev[j] {
                        if crosses(&dirs, i, k, cells) {
                            crossing_cells.push(i);
                        } else {
                            seeded = true;
                            break;
                        }
                    }
                }
                if seeded {
                    fall[i] = 2;
                    frontier.push_back(i);
                }
            }
        }
        let mut last_fall = 0;
        while let Some(i) = frontier.pop_front() {
            last_fall = fall[i];
            for k in 1..9 {
                let j = offset(i, k);
                if dirs[j] == 0 && fall[j] == 0 && elev[i] == elev[j] {
                    if crosses(&dirs, i, k, cells) {
                        crossing_cells.push(i);
                    } else {
                        fall[j] = last_fall + 1;
                        frontier.push_back(j);
                    }
                }
            }
        }
        for &i in &flats {
            if (1..9).any(|k| elev[offset(i, k)] > elev[i]) {
                rise[i] = 1;
                frontier.push_back(i);
            }
        }
        let mut last_rise = 0;
        while let Some(i) = frontier.pop_front() {
            last_rise = rise[i];
            for k in 1..9 {
                let j = offset(i, k);
                if dirs[j] == 0 && rise[j] == 0 {
                    rise[j] = last_rise + 1;
                    frontier.push_back(j);
                }
            }
        }
        for &i in &flats {
            if fall[i] == 0 {
                dirs[i] = DIRECTION_NODATA;
                fall[i] = last_fall.max(1).saturating_add(2);
            }
        }
        let surface = |i: usize| {
            let level = fall[i].max(1) as i64;
            if rise[i] > 0 {
                level + last_rise as i64 + 2 - rise[i] as i64
            } else {
                level
            }
        };
        for &i in &flats {
            let mut best = 0.0;
            for k in ORDER {
                let j = offset(i, k);
                if rise[j] > 0 {
                    let slope = scaled((surface(i) - surface(j)) as f32, scale[k]);
                    if slope > best {
                        best = slope;
                        dirs[i] = k as u8;
                    }
                } else if elev[i] >= elev[j] {
                    dirs[i] = k as u8;
                    break;
                }
            }
        }
        let previous = flats.len();
        flats.retain(|&i| dirs[i] == 0);
        if flats.is_empty() || flats.len() == previous {
            break;
        }
        artificial = (0..cells).map(|i| surface(i) as f32).collect();
        fall.fill(0);
        rise.fill(0);
    }
    crossing_cells.sort_unstable();
    crossing_cells.dedup();
    Trace {
        direction: dirs,
        passes,
        crossing_cells,
    }
}
