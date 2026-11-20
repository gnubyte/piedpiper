/// SplitMix64 — tiny, fast, and identical on every platform.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[-1, 1)`, exactly representable (24-bit grid).
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as u32;
        (u as f32) * (1.0 / 8_388_608.0) - 1.0
    }
}
