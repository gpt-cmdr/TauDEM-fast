// SPDX-License-Identifier: GPL-3.0-only

mod io;
mod tiled_aread8;

use std::{
    collections::HashMap,
    env,
    process::ExitCode,
    time::{Duration, Instant},
};

use io::{Product, Raster, Result, write};
use taudem_rs::area_tiled::{AreaOptions, Arithmetic};
use taudem_rs::profile::{self, CountingAlloc};
use taudem_rs::tiling::{ExecutionMode, ExecutionOptions};
use taudem_rs::{FEL_NODATA, SCALAR_NODATA, Surface, area_d8, d8_steep, pit_remove};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

const HELP: &str = "\
taudem-rs pitremove  -z DEM -fel FILLED [-4] [-depmask MASK]
taudem-rs d8flowdir  -fel FILLED -p DIRECTIONS -sd8 SLOPE
taudem-rs aread8     -p DIRECTIONS -ad8 AREA [-nc] [--exact-area]
taudem-rs pipeline   -z DEM [-fel FILLED] [-p DIRECTIONS] [-sd8 SLOPE] [-ad8 AREA]
                         [-4] [-depmask MASK] [-nc]
Every subcommand also accepts --profile-json PATH, --mode auto|in-memory|tiled,
--max-memory BYTES (such as 4G), --tile-size CELLS and --scratch-dir DIR.

All inputs and outputs are GeoTIFFs. Pipeline computes area in memory even
without output paths. -nc disables downstream edge-contamination checks.
--profile-json writes phase timings, counters and memory use as JSON.
--mode tiled selects out-of-core execution, implemented for aread8 (other
subcommands run in memory); auto tiles aread8 when the in-memory kernel
would exceed --max-memory. --max-memory defaults to the smaller of 65% of
usable and 75% of available memory. Tiled aread8 matches in-memory aread8
bit for bit; --exact-area (tiled only) instead sums exact integer counts,
which differ from TauDEM's float sums above 2^24 cells.
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
        "aread8" => (&["-p", "-ad8"], &["-nc", "--exact-area"]),
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
        let value = if matches!(flag, "-4" | "-nc" | "--exact-area") {
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
    let exact_area = options.contains_key("--exact-area");
    let tiled =
        command == "aread8" && tiled_aread8::use_tiled(options["-p"], &execution, exact_area)?;
    if execution.mode == ExecutionMode::Tiled && command != "aread8" {
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

    if tiled {
        let mut area = AreaOptions::new(check_contamination);
        if exact_area {
            area = area.with_arithmetic(Arithmetic::Exact);
        }
        let outcome = tiled_aread8::run(options["-p"], options["-ad8"], area, &execution)?;
        cells = outcome.cells as usize;
        timings.read = outcome.open;
        timings.add("AreaD8", outcome.compute);
        timings.write = outcome.write;
        report_tiled(&outcome, &execution);
    } else if command == "aread8" {
        let reading = profile::span("read");
        let mut raster = Raster::open(options["-p"])?;
        cells = raster.grid.cells();
        let direction = raster.directions()?;
        drop(reading);
        timings.read = total.elapsed();
        let start = Instant::now();
        let span = profile::span("aread8");
        let area = area_d8(raster.grid, &direction, check_contamination);
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
        cells = raster.grid.cells();
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
                raster.grid,
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
                raster.grid,
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
                let area = area_d8(raster.grid, &direction, check_contamination);
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
            cells,
        );
        std::fs::write(path, json).map_err(|error| format!("writing {path}: {error}"))?;
    }
    Ok(())
}

/// Prints the tiled run's plan and counters and records them for `--profile-json`.
fn report_tiled(outcome: &tiled_aread8::Outcome, execution: &ExecutionOptions) {
    let (plan, stats) = (&outcome.plan, &outcome.stats);
    let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
    let (tw, th) = plan.spec.tile_size();
    println!(
        "Tiled AreaD8: {} tiles of {tw} x {th}, {} workers, budget {:.0} MiB (workers {:.0} MiB each, cache {:.0} MiB)",
        stats.tiles,
        stats.workers,
        mib(plan.budget),
        mib(plan.worker_bytes),
        mib(plan.cache_bytes)
    );
    let g = &stats.graph;
    println!(
        "Perimeter graph: {} exits, {} entries, {} edges, {} with inflow, {} nodata, {} on cycles, {:.1} MiB; max area {}",
        g.nodes,
        g.entries,
        g.edges,
        g.inflow_nodes,
        g.poisoned_nodes,
        g.cycle_nodes,
        mib(g.bytes),
        g.max_area
    );
    println!(
        "f32 replay: {} exits over 2^24 in {} tiles, {} steps, {} operands ({:.3} s)",
        g.replay_nodes, g.replay_tiles, g.replay_steps, g.replay_operands, g.replay_seconds
    );
    println!(
        "Passes: pass 1 {:.3} s, solve {:.3} s, pass 2 {:.3} s",
        stats.pass1_seconds, stats.solve_seconds, stats.pass2_seconds
    );
    if let Some(store) = &stats.store {
        println!(
            "Tile cache: peak resident {:.1} MiB, peak spilled {:.1} MiB ({} spill writes), scratch {}",
            mib(store.peak_resident_bytes),
            mib(store.peak_spilled_bytes),
            store.spill_writes,
            execution.scratch_dir().display()
        );
    }
    stats.record_profile();
    outcome.writer.record_profile("ad8");
    profile::count("aread8.plan.budget_bytes", plan.budget);
    profile::count("aread8.plan.worker_bytes", plan.worker_bytes);
    profile::count("aread8.plan.cache_bytes", plan.cache_bytes);
    profile::count("aread8.plan.graph_bytes", plan.graph_bytes);
    profile::count("aread8.tile_size", tw as u64);
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
