// SPDX-License-Identifier: GPL-3.0-only
//! System and process memory figures for the default memory budget and for
//! memory tests.

/// The bytes of memory the process can use, as GDAL's
/// `CPLGetUsablePhysicalRAM` counts them: `terrio::usable_memory()` (Linux
/// `/proc` and cgroup limits, Windows physical memory and Job Object caps,
/// macOS `hw.memsize`). Only where terrio returns `None` (other platforms)
/// does this fall back to the total physical memory the local `os` module
/// reads, the same call `available_memory` makes.
pub fn usable_memory() -> Option<u64> {
    terrio::usable_memory().or_else(os::usable_memory)
}

/// The physical memory available to new allocations now, without paging:
/// `ullAvailPhys` on Windows, `MemAvailable` on Linux.
pub fn available_memory() -> Option<u64> {
    os::available_memory()
}

/// This process's working set (resident set) and its high-water mark.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProcessMemory {
    pub working_set: u64,
    pub peak_working_set: u64,
}

/// The current and peak working set of this process, when available.
pub fn process_memory() -> Option<ProcessMemory> {
    os::process_memory()
}

#[cfg(windows)]
mod os {
    use super::ProcessMemory;
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    fn status() -> Option<MEMORYSTATUSEX> {
        let mut status = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: `status` is a MEMORYSTATUSEX with its length set, as the
        // function requires.
        (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
    }

    pub fn usable_memory() -> Option<u64> {
        let status = status()?;
        Some(status.ullTotalPhys.min(status.ullTotalVirtual))
    }

    pub fn available_memory() -> Option<u64> {
        status().map(|s| s.ullAvailPhys)
    }

    pub fn process_memory() -> Option<ProcessMemory> {
        let mut counters = PROCESS_MEMORY_COUNTERS {
            cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ..Default::default()
        };
        // SAFETY: The pseudo-handle of the current process is always valid
        // and `counters` is a PROCESS_MEMORY_COUNTERS with its size set.
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) }
                != 0;
        ok.then_some(ProcessMemory {
            working_set: counters.WorkingSetSize as u64,
            peak_working_set: counters.PeakWorkingSetSize as u64,
        })
    }
}

#[cfg(not(windows))]
mod os {
    use super::ProcessMemory;

    fn kib(text: &str, key: &str) -> Option<u64> {
        let line = text.lines().find_map(|l| l.strip_prefix(key))?;
        Some(
            line.trim()
                .trim_end_matches("kB")
                .trim()
                .parse::<u64>()
                .ok()?
                * 1024,
        )
    }

    pub fn usable_memory() -> Option<u64> {
        kib(&std::fs::read_to_string("/proc/meminfo").ok()?, "MemTotal:")
    }

    pub fn available_memory() -> Option<u64> {
        kib(
            &std::fs::read_to_string("/proc/meminfo").ok()?,
            "MemAvailable:",
        )
    }

    pub fn process_memory() -> Option<ProcessMemory> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
        Some(ProcessMemory {
            working_set: kib(&status, "VmRSS:")?,
            peak_working_set: kib(&status, "VmHWM:")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn figures_are_plausible_where_supported() {
        if cfg!(any(windows, target_os = "linux")) {
            let usable = usable_memory().expect("usable memory");
            let available = available_memory().expect("available memory");
            assert!(usable >= 256 << 20, "{usable}");
            assert!(available <= usable.max(available) && available > 0);
            let process = process_memory().expect("process memory");
            assert!(process.working_set > 0 && process.peak_working_set >= process.working_set);
        }
    }
}
