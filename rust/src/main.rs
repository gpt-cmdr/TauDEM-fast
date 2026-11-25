// SPDX-License-Identifier: GPL-3.0-only

mod io;

use std::{
    collections::HashMap,
    env,
    process::ExitCode,
    time::{Duration, Instant},
};

use io::{Product, Raster, Result, write};
use taudem_rs::{FEL_NODATA, SCALAR_NODATA, area_d8, d8_flow, pit_remove};

const HELP: &str = "\
taudem-rs pitremove  -z DEM -fel FILLED [-4] [-depmask MASK]
taudem-rs d8flowdir  -fel FILLED -p DIRECTIONS -sd8 SLOPE
taudem-rs aread8     -p DIRECTIONS -ad8 AREA [-nc]
taudem-rs pipeline   -z DEM [-fel FILLED] [-p DIRECTIONS] [-sd8 SLOPE] [-ad8 AREA]
                         [-4] [-depmask MASK] [-nc]

All inputs and outputs are GeoTIFFs. Pipeline computes area in memory even
without output paths. -nc disables downstream edge-contamination checks.
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
    let mut args = args[1..].iter();
    while let Some(flag) = args.next().map(String::as_str) {
        let flag = if flag == "-4way" { "-4" } else { flag };
        if !required.contains(&flag) && !optional.contains(&flag) {
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
                .ok_or_else(|| format!("{flag} requires a path"))?
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

    let total = Instant::now();
    let mut timings = Timings::default();
    let check_contamination = !options.contains_key("-nc");
    let output = |timing: &mut Timings, raster: &Raster, flag: &str, product: Product<'_>| {
        if let Some(path) = options.get(flag) {
            let start = Instant::now();
            write(path, &raster.header, product)
                .map_err(|error| format!("writing {path}: {error}"))?;
            timing.write += start.elapsed();
        }
        Ok::<(), Box<dyn std::error::Error>>(())
    };

    if command == "aread8" {
        let mut raster = Raster::open(options["-p"])?;
        let direction = raster.directions()?;
        timings.read = total.elapsed();
        let start = Instant::now();
        let area = area_d8(raster.grid, &direction, check_contamination);
        timings.stage("AreaD8", start);
        output(
            &mut timings,
            &raster,
            "-ad8",
            Product::Float(&area, SCALAR_NODATA),
        )?;
    } else {
        let fill = command != "d8flowdir";
        let mut raster = Raster::open(options[if fill { "-z" } else { "-fel" }])?;
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
        timings.read = total.elapsed();
        if fill {
            let start = Instant::now();
            pit_remove(
                raster.grid,
                &mut elevations,
                nodata,
                options.contains_key("-4"),
                mask.as_deref(),
            );
            drop(mask);
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
            let flow = d8_flow(
                raster.grid,
                &elevations,
                if fill { FEL_NODATA } else { nodata },
                &spacing,
                options.contains_key("-sd8"),
            );
            drop(elevations);
            timings.stage("D8FlowDir", start);
            output(
                &mut timings,
                &raster,
                "-p",
                Product::Direction(&flow.direction),
            )?;
            if let Some(slope) = flow.slope {
                output(
                    &mut timings,
                    &raster,
                    "-sd8",
                    Product::Float(&slope, SCALAR_NODATA),
                )?;
            }
            if command == "pipeline" {
                let start = Instant::now();
                let area = area_d8(raster.grid, &flow.direction, check_contamination);
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
        let elapsed = start.elapsed();
        self.compute += elapsed;
        println!("{name} compute time: {:.6}", elapsed.as_secs_f64());
    }
}
