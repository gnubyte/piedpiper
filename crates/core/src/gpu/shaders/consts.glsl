// Exact bit patterns of the core's f32 constants (fpmath.rs, model.rs), so GLSL's decimal
// parsing can never round one differently. Checked by gpu::tests::constants_match_the_core.
#define C_LOG2E uintBitsToFloat(0x3FB8AA3Bu) // 1.442695e0
#define C_LN2_HI uintBitsToFloat(0x3F317200u) // 6.9314575e-1
#define C_LN2_LO uintBitsToFloat(0x35BFBE8Eu) // 1.4286068e-6
#define C_P3 uintBitsToFloat(0x3E2AAAABu) // 1.6666667e-1
#define C_P4 uintBitsToFloat(0x3D2AAAABu) // 4.1666668e-2
#define C_P5 uintBitsToFloat(0x3C088889u) // 8.333334e-3
#define C_P6 uintBitsToFloat(0x3AB60B61u) // 1.3888889e-3
#define C_EXP_FLOOR uintBitsToFloat(0xC28C0000u) // -7e1
#define C_EXP_LO uintBitsToFloat(0xC2A00000u) // -8e1
#define C_EXP_HI uintBitsToFloat(0x42A00000u) // 8e1
#define C_NORM_EPS uintBitsToFloat(0x358637BDu) // 1e-6
#define C_ADAM_ZERO uintBitsToFloat(0x0DA24260u) // 1e-30
