// Shared definitions for the PiedPiper Vulkan kernels.
//
// The CPU codec is bit-exact because every float operation is an IEEE-754 basic operation
// in a fixed order (docs/DESIGN.md, "Determinism"). These kernels follow the same rules:
//
//  * The float environment is pinned like crates/core/src/fpenv.rs does on the CPU:
//    round-to-nearest-even, subnormals preserved, signed zero / Inf / NaN preserved
//    (SPV_KHR_float_controls execution modes; the device must report support).
//  * Every multiply and add is `precise` (SPIR-V NoContraction), so the driver can never
//    fuse a*b+c into an FMA, whose single rounding would give different bits.
//  * Vulkan only guarantees + - * to be correctly rounded; division is 2.5 ulp and sqrt
//    is derived from inversesqrt. So both are done here in integer arithmetic
//    (f32_div, f32_sqrt), exactly rounded by construction.
//  * exp is crates/core/src/fpmath.rs::exp transcribed operation for operation.
//  * Reductions use the core's orders: ascending sequential sums, or the fixed 8-lane
//    pattern ((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7)) followed by the tail in order.
//
// Defining FAST_MATH before including this file drops `precise` (used only to measure what
// FMA contraction does to the bits).

#extension GL_EXT_spirv_intrinsics : require
#extension GL_EXT_control_flow_attributes : enable

#ifndef NO_FLOAT_CONTROLS
spirv_execution_mode(capabilities = [4464], 4459, 32); // DenormPreserve, 32-bit
spirv_execution_mode(capabilities = [4466], 4461, 32); // SignedZeroInfNanPreserve, 32-bit
spirv_execution_mode(capabilities = [4467], 4462, 32); // RoundingModeRTE, 32-bit
#endif

#include "consts.glsl"

#ifdef FAST_MATH
#define PRECISE
#else
#define PRECISE precise
#endif

// One memory pool holds every tensor; kernels address it with float offsets.
layout(std430, binding = 0) buffer Fb { float F[]; };
layout(std430, binding = 1) buffer Ub { uint U[]; };

layout(push_constant, std430) uniform PC { uint p[32]; } pc;
#define P(i) pc.p[i]
#define PF(i) uintBitsToFloat(pc.p[i])

const uint SIGN_BIT = 0x80000000u;
const uint QNAN = 0x7FC00000u;

// Linear invocation id for 1-D work launched on a 2-D grid (large tensors exceed the
// 65535-group limit of one dimension). Workgroups are 64 wide.
uint linear_id() {
    return (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) * 64u + gl_LocalInvocationID.x;
}

// q holds 26 significant bits (q in [2^25, 2^26)); the exact value is
// (q + fraction) * 2^(e - 152) with `sticky` telling whether the fraction is non-zero.
// Rounds to nearest even and packs, handling overflow and subnormal results.
float round_pack(uint sz, int e, uint q, bool sticky) {
    if (e >= 1) {
        uint m = q >> 2;
        bool rnd = (q & 2u) != 0u;
        bool st = sticky || (q & 1u) != 0u;
        if (rnd && (st || (m & 1u) != 0u)) {
            m += 1u;
            if (m == 0x1000000u) {
                m = 0x800000u;
                e += 1;
            }
        }
        if (e >= 255) return uintBitsToFloat(sz | 0x7F800000u);
        return uintBitsToFloat(sz | (uint(e) << 23) | (m & 0x7FFFFFu));
    }
    // Subnormal or zero: m = q >> (3 - e), rounded.
    int s = 3 - e; // >= 3
    uint m = 0u;
    bool rnd = false;
    bool st = sticky;
    if (s - 1 >= 32) {
        st = st || q != 0u;
    } else {
        uint t = q >> uint(s - 1);
        uint lost = q & ((1u << uint(s - 1)) - 1u);
        st = st || lost != 0u;
        rnd = (t & 1u) != 0u;
        m = t >> 1;
    }
    if (rnd && (st || (m & 1u) != 0u)) m += 1u; // 2^23 becomes the smallest normal
    return uintBitsToFloat(sz | m);
}

// IEEE-754 binary32 a / b, round to nearest even, subnormals in and out. Integer only.
float f32_div(float fa, float fb) {
    uint a = floatBitsToUint(fa), b = floatBitsToUint(fb);
    uint sz = (a ^ b) & SIGN_BIT;
    int ea = int((a >> 23) & 0xFFu), eb = int((b >> 23) & 0xFFu);
    uint ma = a & 0x7FFFFFu, mb = b & 0x7FFFFFu;
    if (ea == 255) {
        if (ma != 0u) return uintBitsToFloat(a | 0x400000u);
        if (eb == 255) return uintBitsToFloat(mb != 0u ? (b | 0x400000u) : QNAN);
        return uintBitsToFloat(sz | 0x7F800000u);
    }
    if (eb == 255) {
        if (mb != 0u) return uintBitsToFloat(b | 0x400000u);
        return uintBitsToFloat(sz);
    }
    if (eb == 0 && mb == 0u) {
        if (ea == 0 && ma == 0u) return uintBitsToFloat(QNAN);
        return uintBitsToFloat(sz | 0x7F800000u);
    }
    if (ea == 0 && ma == 0u) return uintBitsToFloat(sz);
    if (ea == 0) {
        int s = 23 - findMSB(ma);
        ma <<= uint(s);
        ea = 1 - s;
    } else {
        ma |= 0x800000u;
    }
    if (eb == 0) {
        int s = 23 - findMSB(mb);
        mb <<= uint(s);
        eb = 1 - s;
    } else {
        mb |= 0x800000u;
    }
    int e = ea - eb + 127;
    if (ma < mb) {
        ma <<= 1;
        e -= 1;
    }
    // Restoring division: q = floor(ma * 2^25 / mb), 26 bits since ma / mb is in [1, 2).
    uint rem = ma, q = 0u;
    for (int i = 0; i < 26; i++) {
        q <<= 1;
        if (rem >= mb) {
            rem -= mb;
            q |= 1u;
        }
        rem <<= 1;
    }
    return round_pack(sz, e, q, rem != 0u);
}

// IEEE-754 binary32 square root, round to nearest even. Integer only.
float f32_sqrt(float fa) {
    uint a = floatBitsToUint(fa);
    int e = int((a >> 23) & 0xFFu);
    uint m = a & 0x7FFFFFu;
    if (e == 255) {
        if (m != 0u) return uintBitsToFloat(a | 0x400000u);
        if ((a & SIGN_BIT) != 0u) return uintBitsToFloat(QNAN);
        return fa;
    }
    if (e == 0 && m == 0u) return fa; // +0 / -0
    if ((a & SIGN_BIT) != 0u) return uintBitsToFloat(QNAN);
    if (e == 0) {
        int s = 23 - findMSB(m);
        m <<= uint(s);
        e = 1 - s;
    } else {
        m |= 0x800000u;
    }
    // value = m * 2^(e-150); scale the significand by 2 or 4 so the exponent is even.
    uint sh = ((e & 1) != 0) ? 1u : 2u;
    uint sig = m << sh; // [2^24, 2^26)
    int j = (e - 150 - int(sh)) / 2;
    // Digit-by-digit root of N = sig * 2^24 (50 bits): r = floor(sqrt(N)), 25 bits.
    uint r = 0u, rem = 0u;
    for (int i = 0; i < 25; i++) {
        int bp = 48 - 2 * i;
        uint pair = (bp >= 24) ? ((sig >> uint(bp - 24)) & 3u) : 0u;
        rem = (rem << 2) | pair;
        uint trial = (r << 2) | 1u;
        r <<= 1;
        if (rem >= trial) {
            rem -= trial;
            r |= 1u;
        }
    }
    uint mm = r >> 1;
    bool rnd = (r & 1u) != 0u;
    bool st = rem != 0u;
    int ee = j + 139;
    if (rnd && (st || (mm & 1u) != 0u)) {
        mm += 1u;
        if (mm == 0x1000000u) {
            mm = 0x800000u;
            ee += 1;
        }
    }
    return uintBitsToFloat((uint(ee) << 23) | (mm & 0x7FFFFFu));
}

// crates/core/src/fpmath.rs::exp, operation for operation.
float pp_exp(float x) {
    float xc = (x >= C_EXP_LO) ? ((x <= C_EXP_HI) ? x : C_EXP_HI) : C_EXP_LO;
    PRECISE float t = xc * C_LOG2E + 0.5;
    int n = int(t);
    if (float(n) > t) n -= 1;
    float nf = float(n);
    PRECISE float r = (xc - nf * C_LN2_HI) - nf * C_LN2_LO;
    PRECISE float pl = 1.0 + r * (1.0 + r * (0.5 + r * (C_P3 + r * (C_P4 + r * (C_P5 + r * C_P6)))));
    PRECISE float res = pl * uintBitsToFloat(uint(n + 127) << 23);
    return res;
}

// fpmath::sigmoid: 1 / (1 + exp(-x)).
float pp_sigmoid(float x) {
    PRECISE float den = 1.0 + pp_exp(-x);
    return f32_div(1.0, den);
}

// Σ F[a+i]·F[b+i] in the fixed 8-lane order (kernels.rs dot_v).
float dot8(uint a, uint b, uint n) {
    uint chunks = n / 8u;
    PRECISE float l0 = 0.0, l1 = 0.0, l2 = 0.0, l3 = 0.0, l4 = 0.0, l5 = 0.0, l6 = 0.0, l7 = 0.0;
    for (uint c = 0u; c < chunks; c++) {
        uint o = c * 8u;
        l0 = l0 + F[a + o] * F[b + o];
        l1 = l1 + F[a + o + 1u] * F[b + o + 1u];
        l2 = l2 + F[a + o + 2u] * F[b + o + 2u];
        l3 = l3 + F[a + o + 3u] * F[b + o + 3u];
        l4 = l4 + F[a + o + 4u] * F[b + o + 4u];
        l5 = l5 + F[a + o + 5u] * F[b + o + 5u];
        l6 = l6 + F[a + o + 6u] * F[b + o + 6u];
        l7 = l7 + F[a + o + 7u] * F[b + o + 7u];
    }
    PRECISE float s = ((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7));
    for (uint i = chunks * 8u; i < n; i++) s = s + F[a + i] * F[b + i];
    return s;
}

// model.rs rms_inv: 1 / sqrt(dot(x, x) / n + 1e-6).
float rms_inv(uint x, uint n) {
    float ms = f32_div(dot8(x, x, n), float(n));
    PRECISE float t = ms + C_NORM_EPS;
    return f32_div(1.0, f32_sqrt(t));
}

// Key/value window of a query at local position t (model.rs AttnCtx::window).
// Keys are visited newest first: n_cur from the current segment, then n_hist from memory.
void attn_window(uint t, uint seg_start, uint win, out uint nk, out uint n_cur, out uint first_slot) {
    uint p = seg_start + t;
    nk = min(p + 1u, win);
    n_cur = min(nk, t + 1u);
    uint n_hist = nk - n_cur;
    first_slot = n_hist > 0u ? (seg_start - 1u) % win : 0u;
}

// Ring slot of the j-th memory key (j = 0 is the newest), counting down from first_slot.
uint hist_slot(uint first_slot, uint j, uint win) {
    return (first_slot + win - j) % win;
}
