//! Pins the floating-point environment to IEEE-754 defaults while the codec runs.
//!
//! The encoder and decoder must compute bit-identical floats. A host process that has
//! switched on flush-to-zero / denormals-are-zero (shared libraries built with
//! `-ffast-math` do this process-wide) or changed the rounding mode would silently change
//! results, so every thread that does model arithmetic holds an [`FpGuard`].
//! WebAssembly has no such modes, so the guard is a no-op there.

pub struct FpGuard {
    #[allow(dead_code)]
    saved: imp::Saved,
}

impl FpGuard {
    #[inline]
    pub fn new() -> Self {
        FpGuard { saved: imp::enter() }
    }
}

impl Default for FpGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FpGuard {
    #[inline]
    fn drop(&mut self) {
        imp::leave(self.saved);
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    #[allow(deprecated)]
    use core::arch::x86_64::{_mm_getcsr, _mm_setcsr};
    pub type Saved = u32;
    /// All exceptions masked, round-to-nearest-even, FTZ and DAZ off.
    const IEEE: u32 = 0x1F80;
    #[inline]
    #[allow(deprecated)]
    pub fn enter() -> Saved {
        // SAFETY: SSE is part of the x86_64 baseline; MXCSR access cannot fault.
        unsafe {
            let s = _mm_getcsr();
            if s & !0x3F != IEEE {
                _mm_setcsr(IEEE);
            }
            s
        }
    }
    #[inline]
    #[allow(deprecated)]
    pub fn leave(s: Saved) {
        unsafe {
            if _mm_getcsr() & !0x3F != s & !0x3F {
                _mm_setcsr(s);
            }
        }
    }
}

#[cfg(all(target_arch = "x86", target_feature = "sse"))]
mod imp {
    #[allow(deprecated)]
    use core::arch::x86::{_mm_getcsr, _mm_setcsr};
    pub type Saved = u32;
    const IEEE: u32 = 0x1F80;
    #[inline]
    #[allow(deprecated)]
    pub fn enter() -> Saved {
        unsafe {
            let s = _mm_getcsr();
            if s & !0x3F != IEEE {
                _mm_setcsr(IEEE);
            }
            s
        }
    }
    #[inline]
    #[allow(deprecated)]
    pub fn leave(s: Saved) {
        unsafe {
            if _mm_getcsr() & !0x3F != s & !0x3F {
                _mm_setcsr(s);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod imp {
    use core::arch::asm;
    pub type Saved = u64;
    /// FIZ, AH, FZ16, RMode (2 bits), FZ.
    const CLEAR: u64 = 1 | (1 << 1) | (1 << 19) | (3 << 22) | (1 << 24);
    #[inline]
    pub fn enter() -> Saved {
        let s: u64;
        // SAFETY: FPCR is readable/writable at EL0 on every AArch64 OS we target.
        unsafe {
            asm!("mrs {}, fpcr", out(reg) s, options(nostack, preserves_flags));
            let want = s & !CLEAR;
            if want != s {
                asm!("msr fpcr, {}", in(reg) want, options(nostack, preserves_flags));
            }
        }
        s
    }
    #[inline]
    pub fn leave(s: Saved) {
        unsafe {
            let cur: u64;
            asm!("mrs {}, fpcr", out(reg) cur, options(nostack, preserves_flags));
            if cur != s {
                asm!("msr fpcr, {}", in(reg) s, options(nostack, preserves_flags));
            }
        }
    }
}

// 32-bit ARM: stable Rust never exposes the VFP target features in `cfg`, so gate on
// targets that guarantee a VFP unit (Android armeabi-v7a mandates VFPv3-D16; `eabihf`
// targets pass floats in VFP registers).
#[cfg(all(target_arch = "arm", any(target_os = "android", target_abi = "eabihf")))]
mod imp {
    use core::arch::asm;
    pub type Saved = u32;
    /// RMode (2 bits) and FZ.
    const CLEAR: u32 = (3 << 22) | (1 << 24);
    #[inline]
    pub fn enter() -> Saved {
        let s: u32;
        unsafe {
            asm!("vmrs {}, fpscr", out(reg) s, options(nostack));
            let want = s & !CLEAR;
            if want != s {
                asm!("vmsr fpscr, {}", in(reg) want, options(nostack));
            }
        }
        s
    }
    #[inline]
    pub fn leave(s: Saved) {
        unsafe {
            asm!("vmsr fpscr, {}", in(reg) s, options(nostack));
        }
    }
}

#[cfg(not(any(
    target_arch = "x86_64",
    all(target_arch = "x86", target_feature = "sse"),
    target_arch = "aarch64",
    all(target_arch = "arm", any(target_os = "android", target_abi = "eabihf"))
)))]
mod imp {
    // wasm32 (always IEEE, no modes) and anything else: nothing to do.
    pub type Saved = ();
    #[inline]
    pub fn enter() -> Saved {}
    #[inline]
    pub fn leave(_: Saved) {}
}
