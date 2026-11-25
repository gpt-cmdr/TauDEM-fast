// SPDX-License-Identifier: GPL-3.0-only
//! The worker threads of every tiled stage.
//!
//! Tile loops run on plain scoped threads, never as jobs of Rayon's pool. A
//! Rayon worker that waits on a GeoTIFF read, which terrio parallelizes on
//! its own pool, keeps busy by running other queued jobs of its pool: it
//! starts more tiles while it still holds a tile's buffers, which nests
//! tiles on one thread, unbounds the tiles in flight and starves the waiting
//! read. A plain thread just blocks. WP-S2 and WP-S3 each met this (T-S2 in
//! `docs/rust-port/STREAMING.md` §4); every tiled pass, reading or not, uses
//! these runners so that the fused pipeline has one scheme. Kernels inside a
//! tile may still use Rayon.
//!
//! A runner's thread count is fixed by the caller, normally the current
//! Rayon pool's size ([`threads`]), so `RAYON_NUM_THREADS` and an installed
//! pool size it as before.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Condvar, Mutex};

use super::spec::{TileId, TileSpec};
use super::{Error, Result};

/// Stack of each spawned worker thread: tile kernels recurse little, but the
/// flood queues and the writers' encoders keep some state on the stack.
const STACK_BYTES: usize = 8 << 20;

/// The worker count of the calling context: the current Rayon pool's size.
pub fn threads() -> usize {
    rayon::current_num_threads().max(1)
}

/// The earliest failure by item order, so that errors do not depend on the
/// schedule.
struct Failure<E = Error> {
    failed: AtomicBool,
    first: Mutex<Option<(usize, E)>>,
}

impl<E> Failure<E> {
    fn new() -> Self {
        Self {
            failed: AtomicBool::new(false),
            first: Mutex::new(None),
        }
    }

    fn set(&self, item: usize, error: E) {
        self.failed.store(true, Relaxed);
        let mut slot = self.first.lock().unwrap_or_else(|p| p.into_inner());
        if slot.as_ref().is_none_or(|(j, _)| item < *j) {
            *slot = Some((item, error));
        }
    }

    fn failed(&self) -> bool {
        self.failed.load(Relaxed)
    }

    fn into_result<T>(self, ok: T) -> std::result::Result<T, E> {
        match self.first.into_inner().unwrap_or_else(|p| p.into_inner()) {
            Some((_, error)) => Err(error),
            None => Ok(ok),
        }
    }
}

/// Runs `work` once per state: the first on the calling thread, the others
/// on scoped threads. A thread that cannot be spawned leaves its share to the
/// threads that run.
fn spread<S: Send>(states: &mut [S], work: &(impl Fn(&mut S) + Sync)) {
    let Some((first, rest)) = states.split_first_mut() else {
        return;
    };
    std::thread::scope(|scope| {
        for state in rest {
            let spawned = std::thread::Builder::new()
                .name("taudem-tile".into())
                .stack_size(STACK_BYTES)
                .spawn_scoped(scope, move || work(state));
            if spawned.is_err() {
                break;
            }
        }
        work(first);
    });
}

/// Runs `visit(state, item)` for `item` in `0..count`, one plain thread per
/// element of `states` (at most `count`), items taken in ascending order.
/// Returns the sum of `visit`'s counts, or the error of the earliest failed
/// item; after a failure no new item starts.
pub fn run_with<S: Send>(
    states: &mut [S],
    count: usize,
    visit: impl Fn(&mut S, usize) -> Result<u64> + Sync,
) -> Result<u64> {
    let next = AtomicUsize::new(0);
    let total = AtomicU64::new(0);
    let failure = Failure::new();
    let threads = states.len().min(count);
    if threads == 0 {
        return Ok(0);
    }
    let work = |state: &mut S| {
        while !failure.failed() {
            let item = next.fetch_add(1, Relaxed);
            if item >= count {
                break;
            }
            match visit(state, item) {
                Ok(n) => {
                    total.fetch_add(n, Relaxed);
                }
                Err(error) => failure.set(item, error),
            }
        }
    };
    spread(&mut states[..threads], &work);
    failure.into_result(total.into_inner())
}

/// [`run_with`] with a fresh `S::default()` per thread.
pub fn run<S: Default + Send>(
    threads: usize,
    count: usize,
    visit: impl Fn(&mut S, usize) -> Result<u64> + Sync,
) -> Result<u64> {
    let mut states: Vec<S> = (0..threads.max(1).min(count.max(1)))
        .map(|_| S::default())
        .collect();
    run_with(&mut states, count, visit)
}

/// `f(item)` for every item of `0..count` on `threads` plain threads,
/// collected in item order.
pub fn map<T: Send>(
    threads: usize,
    count: usize,
    f: impl Fn(usize) -> Result<T> + Sync,
) -> Result<Vec<T>> {
    let slots: Vec<Mutex<Option<T>>> = (0..count).map(|_| Mutex::new(None)).collect();
    run::<()>(threads, count, |_, item| {
        let value = f(item)?;
        *slots[item].lock().unwrap_or_else(|p| p.into_inner()) = Some(value);
        Ok(0)
    })?;
    Ok(slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(|p| p.into_inner())
                .expect("every item ran")
        })
        .collect())
}

/// Runs `visit` on every tile of `spec`, each tile after its west and north
/// neighbors (the NW→SE wavefront D8's crossing rule needs), on `threads`
/// plain threads with a ready queue (dependency counting, no barriers).
/// Returns the error of the earliest failed tile in row-major order; after a
/// failure no new tile starts.
pub fn wavefront<S: Send, E: Send>(
    spec: &TileSpec,
    states: &mut [S],
    visit: impl Fn(&mut S, TileId) -> std::result::Result<(), E> + Sync,
) -> std::result::Result<(), E> {
    struct Queue {
        ready: Vec<TileId>,
        done: usize,
    }
    let count = spec.tile_count();
    let pending: Vec<AtomicUsize> = spec
        .row_major()
        .map(|id| AtomicUsize::new(usize::from(id.tx > 0) + usize::from(id.ty > 0)))
        .collect();
    let queue = Mutex::new(Queue {
        ready: vec![TileId::new(0, 0)],
        done: 0,
    });
    let wake = Condvar::new();
    let failure = Failure::<E>::new();
    let work = |state: &mut S| {
        loop {
            let id = {
                let mut q = queue.lock().unwrap_or_else(|p| p.into_inner());
                loop {
                    if failure.failed() || q.done == count {
                        return;
                    }
                    // Lowest anti-diagonal first, then row-major: the
                    // wavefront advances evenly.
                    if let Some(at) = (0..q.ready.len())
                        .min_by_key(|&i| (q.ready[i].tx + q.ready[i].ty, q.ready[i].ty))
                    {
                        break q.ready.swap_remove(at);
                    }
                    q = wake.wait(q).unwrap_or_else(|p| p.into_inner());
                }
            };
            let result = visit(state, id);
            let mut q = queue.lock().unwrap_or_else(|p| p.into_inner());
            q.done += 1;
            match result {
                Ok(()) => {
                    for k in [1, 7] {
                        if let Some(next) = spec.neighbor(id, k)
                            && pending[spec.index(next)].fetch_sub(1, Relaxed) == 1
                        {
                            q.ready.push(next);
                        }
                    }
                }
                Err(error) => failure.set(spec.index(id), error),
            }
            drop(q);
            wake.notify_all();
        }
    };
    let threads = states.len().min(count);
    spread(&mut states[..threads], &work);
    failure.into_result(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_every_item_once_and_reports_the_first_error() {
        for threads in [1, 3, 8] {
            let seen: Vec<AtomicUsize> = (0..100).map(|_| AtomicUsize::new(0)).collect();
            let total = run::<Vec<u8>>(threads, 100, |buffer, i| {
                buffer.push(1);
                seen[i].fetch_add(1, Relaxed);
                Ok(i as u64)
            })
            .unwrap();
            assert_eq!(total, (0..100).sum::<u64>());
            assert!(seen.iter().all(|s| s.load(Relaxed) == 1));
            let error = run::<()>(threads, 100, |_, i| {
                if i % 10 == 7 {
                    Err(Error::Invalid(format!("item {i}")))
                } else {
                    Ok(0)
                }
            })
            .unwrap_err();
            assert_eq!(error.to_string(), "item 7");
            let squares = map(threads, 50, |i| Ok(i * i)).unwrap();
            assert_eq!(squares, (0..50).map(|i| i * i).collect::<Vec<_>>());
        }
    }

    #[test]
    fn wavefront_visits_after_west_and_north() {
        let spec = TileSpec::new(100, 67, 16, 16).unwrap();
        for threads in [1, 4] {
            let done: Vec<AtomicBool> = (0..spec.tile_count())
                .map(|_| AtomicBool::new(false))
                .collect();
            let mut states = vec![(); threads];
            wavefront(&spec, &mut states, |_, id| {
                for k in [5, 3] {
                    if let Some(n) = spec.neighbor(id, k) {
                        assert!(done[spec.index(n)].load(Relaxed), "{id:?} before {n:?}");
                    }
                }
                done[spec.index(id)].store(true, Relaxed);
                Ok::<(), Error>(())
            })
            .unwrap();
            assert!(done.iter().all(|d| d.load(Relaxed)));
            let error: Result<()> = wavefront(&spec, &mut states, |_, id| {
                if id.tx == 2 && id.ty == 1 {
                    Err(Error::Invalid("stop".into()))
                } else {
                    Ok(())
                }
            });
            assert!(error.is_err());
        }
    }
}
