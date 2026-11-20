//! Minimal parallel-for used by the kernels. Work is always split into regions whose
//! per-element arithmetic is independent of the split, so the thread count never changes
//! the output bytes.

#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
// SAFETY: users only ever write disjoint regions through the pointer.
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// The raw pointer. (A method, so closures capture the `Sync` wrapper, not the field.)
    #[inline(always)]
    pub fn get(&self) -> *mut T {
        self.0
    }

    /// # Safety
    /// The range must be in bounds and not aliased by any other live reference.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice(&self, off: usize, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(off), len)
    }
}

/// Number of worker threads available to the current call.
#[inline]
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    {
        rayon::current_num_threads()
    }
    #[cfg(not(feature = "parallel"))]
    {
        1
    }
}

/// Calls `f(start, end)` over a partition of `0..n`. `cost` is the approximate work per
/// item (in multiply-adds); small jobs stay on the calling thread.
#[inline]
pub fn for_ranges<F>(n: usize, cost: usize, align: usize, f: F)
where
    F: Fn(usize, usize) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        const MIN_TASK: usize = 1 << 17; // multiply-adds per task worth a dispatch
        let nt = rayon::current_num_threads();
        let total = n.saturating_mul(cost.max(1));
        if nt > 1 && total >= 2 * MIN_TASK && n > align {
            let align = align.max(1);
            let units = n.div_ceil(align);
            let tasks = (total / MIN_TASK).min(units).min(nt * 3).max(1);
            if tasks > 1 {
                use rayon::prelude::*;
                (0..tasks).into_par_iter().for_each(|t| {
                    let _fp = crate::fpenv::FpGuard::new();
                    let s = (t * units / tasks * align).min(n);
                    let e = ((t + 1) * units / tasks * align).min(n);
                    if s < e {
                        f(s, e);
                    }
                });
                return;
            }
        }
    }
    let _ = align;
    let _ = cost;
    if n > 0 {
        f(0, n);
    }
}
