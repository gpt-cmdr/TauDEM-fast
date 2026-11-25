// SPDX-License-Identifier: GPL-3.0-only

mod d8flowdir_tiled;
mod io;

use std::{
    collections::HashMap,
    env,
    process::ExitCode,
    time::{Duration, Instant},
};

use io::{Product, Raster, Result, write};
use taudem_rs::pitremove_tiled::PitRemovePlan;
use taudem_rs::profile::{self, CountingAlloc};
use taudem_rs::tiling::{ExecutionMode, ExecutionOptions, TileSpec, auto_tile_size};
use taudem_rs::{FEL_NODATA, SCALAR_NODATA, Surface, area_d8, d8_steep, pit_remove};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const HELP: &str = "\
taudem-rs pitremove  -z DEM -fel FILLED [-4] [-depmask MASK]
taudem-rs d8flowdir  -fel FILLED -p DIRECTIONS -sd8 SLOPE
taudem-rs aread8     -p DIRECTIONS -ad8 AREA [-nc]
taudem-rs pipeline   -z DEM [-fel FILLED] [-p DIRECTIONS] [-sd8 SLOPE] [-ad8 AREA]
                         [-4] [-depmask MASK] [-nc]
Every subcommand also accepts --profile-json PATH, --mode auto|in-memory|tiled,
--max-memory BYTES (such as 4G), --tile-size CELLS and --scratch-dir DIR.

All inputs and outputs are GeoTIFFs. Pipeline computes area in memory even
without output paths. -nc disables downstream edge-contamination checks.
--profile-json writes phase timings, counters and memory use as JSON.
--mode tiled selects out-of-core execution within --max-memory (pitremove
and d8flowdir only so far); auto runs them tiled when their in-memory
estimate exceeds the budget, and in memory otherwise. --max-memory defaults
to the smaller of 65% of usable and 75% of available memory.
Set RAYON_NUM_THREADS to control the number of worker threads.";

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("taudem-rs: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<()> {
    let Some(command) = args.first().map(String::as_str) else {
        return Err(HELP.into());
    };
    if matches!(command, "--help" | "-h" | "help")
        || args
            .get(1)
            .is_some_and(|s| matches!(s.as_str(), "-h" | "--help"))
    {
        println!("{HELP}");
        return Ok(());
    }
    if matches!(command, "--version" | "-V") {
        println!("taudem-rs {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    let (required, optional): (&[&str], &[&str]) = match command {
        "pitremove" => (&["-z", "-fel"], &["-4", "-depmask"]),
        "d8flowdir" => (&["-fel", "-p", "-sd8"], &[]),
        "aread8" => (&["-p", "-ad8"], &["-nc"]),
        "pipeline" => (
            &["-z"],
            &["-fel", "-p", "-sd8", "-ad8", "-4", "-depmask", "-nc"],
        ),
        _ => return Err(format!("unknown subcommand {command}\n{HELP}").into()),
    };
    let mut options = HashMap::new();
    let argv = args.clone();
    let mut args = args[1..].iter();
    while let Some(flag) = args.next().map(String::as_str) {
        let flag = if flag == "-4way" { "-4" } else { flag };
        if !required.contains(&flag)
            && !optional.contains(&flag)
            && flag != "--profile-json"
            && !ExecutionOptions::FLAGS.contains(&flag)
        {
            return Err(if matches!(flag, "-o" | "-wg" | "-sfdr") {
                format!("{flag} is not supported in stage one")
            } else {
                format!("unknown option {flag} for {command}")
            }
            .into());
        }
        let value = if matches!(flag, "-4" | "-nc") {
            ""
        } else {
            args.next()
                .filter(|path| !path.starts_with('-'))
                .ok_or_else(|| {
                    let what = if matches!(flag, "--mode" | "--max-memory" | "--tile-size") {
                        "value"
                    } else {
                        "path"
                    };
                    format!("{flag} requires a {what}")
                })?
        };
        if options.insert(flag, value).is_some() {
            return Err(format!("duplicate option {flag}").into());
        }
    }
    for flag in required {
        if !options.contains_key(flag) {
            return Err(format!("{command} requires {flag}\n{HELP}").into());
        }
    }
    let execution = ExecutionOptions::from_flags(|flag| options.get(flag).copied())?;
    if execution.mode == ExecutionMode::Tiled && !matches!(command, "pitremove" | "d8flowdir") {
        return Err(format!("--mode tiled is not yet implemented for {command}").into());
    }

    profile::start();
    let total = Instant::now();
    let mut timings = Timings::default();
    let cells;
    let check_contamination = !options.contains_key("-nc");
    let output = |timing: &mut Timings, raster: &Raster, flag: &str, product: Product<'_>| {
        if let Some(path) = options.get(flag) {
            let _span = profile::span(format!("write.{}", &flag[1..]));
            let start = Instant::now();
            write(path, &raster.header, product)
                .map_err(|error| format!("writing {path}: {error}"))?;
            timing.write += start.elapsed();
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    };

    let tiled_input = if command == "d8flowdir" && execution.mode != ExecutionMode::InMemory {
        let raster = Raster::open(options["-fel"])?;
        d8flowdir_tiled::wanted(&execution, &raster).then_some(raster)
    } else {
        None
    };
    if let Some(raster) = tiled_input {
        let run = d8flowdir_tiled::run(raster, options["-p"], options["-sd8"], &execution)?;
        cells = run.cells as u64;
        timings.read = run.read;
        timings.write = run.write;
        timings.add("D8FlowDir", run.compute);
    } else if command == "aread8" {
        let reading = profile::span("read");
        let mut raster = Raster::open(options["-p"])?;
        let grid = raster.grid()?;
        cells = grid.cells() as u64;
        let direction = raster.directions()?;
        drop(reading);
        timings.read = total.elapsed();
        let start = Instant::now();
        let span = profile::span("aread8");
        let area = area_d8(grid, &direction, check_contamination);
        drop(span);
        timings.stage("AreaD8", start);
        output(
            &mut timings,
            &raster,
            "-ad8",
            Product::Float(&area, SCALAR_NODATA),
        )?;
    } else {
        let fill = command != "d8flowdir";
        let reading = profile::span("read");
        let mut raster = Raster::open(options[if fill { "-z" } else { "-fel" }])?;
        let (width, height) = raster.dimensions();
        cells = width as u64 * height as u64;
        let has_mask = options.contains_key("-depmask");
        let estimate = cells.saturating_mul(if has_mask { 9 } else { 8 });
        let tiled = command == "pitremove"
            && match execution.mode {
                ExecutionMode::Tiled => true,
                ExecutionMode::Auto => {
                    estimate > execution.memory_budget() || raster.grid().is_err()
                }
                ExecutionMode::InMemory => false,
            };
        if tiled {
            drop(reading);
            timings.read = total.elapsed();
            tiled_pitremove(&mut raster, &options, execution, &mut timings)?;
        } else {
            let grid = raster.grid()?;
            let nodata = raster.nodata();
            let mut elevations = raster.elevations()?;
            let mask = if let Some(path) = options.get("-depmask") {
                let mut mask = Raster::open(path)?;
                raster.check_mask(&mask)?;
                Some(mask.mask()?)
            } else {
                None
            };
            let spacing = if command == "pitremove" {
                Vec::new()
            } else {
                raster.spacing()?
            };
            drop(reading);
            timings.read = total.elapsed();
            if fill {
                let start = Instant::now();
                let span = profile::span("pitremove");
                pit_remove(
                    grid,
                    &mut elevations,
                    nodata,
                    options.contains_key("-4"),
                    mask.as_deref(),
                );
                drop(mask);
                drop(span);
                timings.stage("PitRemove", start);
                output(
                    &mut timings,
                    &raster,
                    "-fel",
                    Product::Float(&elevations, FEL_NODATA),
                )?;
            }
            if command != "pitremove" {
                let start = Instant::now();
                let span = profile::span("d8flowdir.steep");
                let steep = d8_steep(
                    grid,
                    &elevations,
                    if fill { FEL_NODATA } else { nodata },
                    &spacing,
                );
                drop(span);
                let mut compute = start.elapsed();
                // TauDEM writes sd8 before routing flats. Streaming it by rows here
                // keeps a whole-grid slope buffer out of memory.
                output(
                    &mut timings,
                    &raster,
                    "-sd8",
                    Product::FloatRows(
                        &|row, out: &mut [f32]| steep.slope_rows(&elevations, row, out),
                        SCALAR_NODATA,
                    ),
                )?;
                let start = Instant::now();
                let span = profile::span("d8flowdir.flats");
                // The filled grid is no longer needed and becomes flat scratch.
                let direction = steep.resolve_flats(Surface::Owned(elevations));
                drop(span);
                compute += start.elapsed();
                timings.add("D8FlowDir", compute);
                output(&mut timings, &raster, "-p", Product::Direction(&direction))?;
                if command == "pipeline" {
                    let start = Instant::now();
                    let span = profile::span("aread8");
                    let area = area_d8(grid, &direction, check_contamination);
                    drop(span);
                    timings.stage("AreaD8", start);
                    output(
                        &mut timings,
                        &raster,
                        "-ad8",
                        Product::Float(&area, SCALAR_NODATA),
                    )?;
                    // Ensure a compute-only pipeline still consumes its final product.
                    std::hint::black_box(area);
                }
            }
        }
    }
    println!(
        "Read time: {:.6}\nCompute time: {:.6}\nWrite time: {:.6}\nTotal time: {:.6}",
        timings.read.as_secs_f64(),
        timings.compute.as_secs_f64(),
        timings.write.as_secs_f64(),
        total.elapsed().as_secs_f64()
    );
    if let Some(path) = options.get("--profile-json") {
        let json = profile::report().to_json(
            &[
                ("tool", profile::quote("taudem-rs")),
                ("version", profile::quote(env!("CARGO_PKG_VERSION"))),
                ("command", profile::quote(command)),
                (
                    "argv",
                    format!(
                        "[{}]",
                        argv.iter()
                            .map(|a| profile::quote(a))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                ),
                ("threads", rayon::current_num_threads().to_string()),
                ("cells", cells.to_string()),
                ("read_s", format!("{:.6}", timings.read.as_secs_f64())),
                ("compute_s", format!("{:.6}", timings.compute.as_secs_f64())),
                ("write_s", format!("{:.6}", timings.write.as_secs_f64())),
                ("total_s", format!("{:.6}", total.elapsed().as_secs_f64())),
            ],
            cells as usize,
        );
        std::fs::write(path, json).map_err(|error| format!("writing {path}: {error}"))?;
    }
    Ok(())
}

fn tiled_pitremove(
    raster: &mut Raster,
    flags: &HashMap<&str, &str>,
    mut execution: ExecutionOptions,
    timings: &mut Timings,
) -> Result<()> {
    let (width, height) = raster.dimensions();
    let four_way = flags.contains_key("-4");
    let mut mask = flags
        .get("-depmask")
        .map(|path| Raster::open(path))
        .transpose()?;
    if let Some(mask) = &mask {
        raster.check_mask(mask)?;
    }
    let nodata = raster.nodata();
    let budget = execution.memory_budget();
    let mut side = execution
        .tile_size
        .unwrap_or_else(|| auto_tile_size(width as u64 * height as u64, budget));
    let native = raster.block_size();
    let mut fallback: Option<PitRemovePlan> = None;
    let plan = loop {
        let spec = if execution.tile_size.is_some() {
            TileSpec::new(width, height, side, side)?
        } else {
            let aligned = TileSpec::aligned(width, height, side, native)?;
            let (w, h) = aligned.tile_size();
            if w.max(h) > side.saturating_mul(2).min(taudem_rs::MAX_TILE) {
                TileSpec::new(width, height, side, side)?
            } else {
                aligned
            }
        };
        let block = io::block_size_for(&spec);
        let (w, h) = spec.tile_size();
        // Regions with unaligned edges can leave partial TIFF blocks in
        // the writer at the active row and around tiles still in flight.
        let unaligned = (spec.tiles_x() > 1 && !w.is_multiple_of(block))
            || (spec.tiles_y() > 1 && !h.is_multiple_of(block));
        let writer_reserve = if unaligned {
            ((width + (w + h) * 2 * rayon::current_num_threads()) * block * 4) as u64
        } else {
            0
        };
        execution.max_memory = Some(budget.saturating_sub(writer_reserve));
        match PitRemovePlan::new(spec, &execution, nodata, four_way, mask.is_some()) {
            Ok(plan) => {
                if execution.tile_size.is_some()
                    || plan.workers() >= rayon::current_num_threads().min(spec.tile_count())
                {
                    break plan;
                }
                if fallback
                    .as_ref()
                    .is_none_or(|best| plan.workers() > best.workers())
                {
                    fallback = Some(plan);
                }
                if side == 16 {
                    break fallback.expect("the current layout fits");
                }
            }
            Err(error) if execution.tile_size.is_some() || side == 16 => {
                break fallback.ok_or(error)?;
            }
            Err(_) => {}
        }
        side = (side / 2).max(16);
    };
    let spec = plan.spec();
    let (tile_width, tile_height) = spec.tile_size();
    println!(
        "PitRemove mode: tiled ({tile_width}x{tile_height}, {} workers)",
        plan.workers()
    );
    let solved = {
        let dem = raster.tile_reader()?;
        let mask = mask.as_mut().map(Raster::tile_reader).transpose()?;
        plan.solve(|halo, elevations, flags| {
            dem.halo(&halo, elevations)?;
            if let (Some(reader), Some(flags)) = (&mask, flags) {
                reader.mask(halo.core, flags)?;
            }
            Ok(())
        })?
    };
    let output = flags["-fel"];
    let write = Instant::now();
    let writer = io::tile_writer::<f32>(output, &raster.header, &spec, f64::from(FEL_NODATA), 8)?;
    let stats = io::write_fel_tiles(solved, writer)
        .map_err(|error| format!("writing {output}: {error}"))?;
    // Pass 2 includes overlapped compression and writing. Count only the
    // non-overlapping writer setup and drain in the summary write time.
    timings.write += write.elapsed().saturating_sub(stats.pass2);
    timings.add("PitRemove", stats.pass1 + stats.graph + stats.pass2);
    println!(
        "PitRemove Pass 1 time: {:.6}\nPitRemove graph time: {:.6}\nPitRemove Pass 2 time: {:.6}\nPitRemove empty tiles: {}\nPitRemove spilled tiles: {} ({} layer writes, {} bytes)",
        stats.pass1.as_secs_f64(),
        stats.graph.as_secs_f64(),
        stats.pass2.as_secs_f64(),
        stats.empty_tiles,
        stats.spilled_tiles,
        stats.cache.spill_writes,
        stats.cache.spill_write_bytes
    );
    Ok(())
}

#[derive(Default)]
struct Timings {
    read: Duration,
    compute: Duration,
    write: Duration,
}

impl Timings {
    fn stage(&mut self, name: &str, start: Instant) {
        self.add(name, start.elapsed());
    }

    fn add(&mut self, name: &str, elapsed: Duration) {
        self.compute += elapsed;
        println!("{name} compute time: {:.6}", elapsed.as_secs_f64());
    }
}
