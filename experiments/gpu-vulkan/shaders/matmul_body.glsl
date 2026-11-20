// Y[i][j] = Σ_{t<K} A(i,t)·B(t,j), t ascending, from +0.0 — the per-element arithmetic of
// kernels.rs `matmul` and `matmul_tn`. Each thread owns 2x2 outputs and accumulates them in
// ascending t across the K tiles, so tiling never changes an element's operation order.
// A(i,t) = F[a + i*ars + t*acs], B(t,j) = F[b + t*brs + j*bcs], Y[i][j] = F[y + i*yrs + j].
// push: 0 M, 1 N, 2 K, 3 a, 4 ars, 5 acs, 6 b, 7 brs, 8 bcs, 9 y, 10 yrs
layout(local_size_x = 16, local_size_y = 16) in;
const uint TS = 32u;
const uint BK = 16u;
shared float As[BK][TS + 1u];
shared float Bs[BK][TS + 1u];

void main() {
    uint M = P(0), N = P(1), K = P(2);
    uint a = P(3), ars = P(4), acs = P(5), b = P(6), brs = P(7), bcs = P(8), y = P(9), yrs = P(10);
    uint i0 = gl_WorkGroupID.y * TS, j0 = gl_WorkGroupID.x * TS;
    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y;
    uint lid = ty * 16u + tx;
    PRECISE float c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;
    for (uint k0 = 0u; k0 < K; k0 += BK) {
        uint kn = min(BK, K - k0);
        for (uint e = lid; e < TS * BK; e += 256u) {
            uint ii, tt;
            if (acs == 1u) {
                ii = e / BK;
                tt = e % BK;
            } else {
                tt = e / TS;
                ii = e % TS;
            }
            float v = 0.0;
            if (i0 + ii < M && tt < kn) v = F[a + (i0 + ii) * ars + (k0 + tt) * acs];
            As[tt][ii] = v;
        }
        for (uint e = lid; e < TS * BK; e += 256u) {
            uint tt, jj;
            if (bcs == 1u) {
                tt = e / TS;
                jj = e % TS;
            } else {
                jj = e / BK;
                tt = e % BK;
            }
            float v = 0.0;
            if (j0 + jj < N && tt < kn) v = F[b + (k0 + tt) * brs + (j0 + jj) * bcs];
            Bs[tt][jj] = v;
        }
        barrier();
        for (uint tt = 0u; tt < kn; tt++) {
            float a0 = As[tt][ty], a1 = As[tt][ty + 16u];
            float b0 = Bs[tt][tx], b1 = Bs[tt][tx + 16u];
            c00 = c00 + a0 * b0;
            c01 = c01 + a0 * b1;
            c10 = c10 + a1 * b0;
            c11 = c11 + a1 * b1;
        }
        barrier();
    }
    uint ia = i0 + ty, ib = i0 + ty + 16u, ja = j0 + tx, jb = j0 + tx + 16u;
    if (ia < M) {
        if (ja < N) F[y + ia * yrs + ja] = c00;
        if (jb < N) F[y + ia * yrs + jb] = c01;
    }
    if (ib < M) {
        if (ja < N) F[y + ib * yrs + ja] = c10;
        if (jb < N) F[y + ib * yrs + jb] = c11;
    }
}
