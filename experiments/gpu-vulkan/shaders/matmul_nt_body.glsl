// Y[i][j] = dot8(A[i,:], B[j,:]) over K — kernels.rs `matmul_nt`: eight lane accumulators
// over consecutive 8-element chunks, reduced as ((l0+l4)+(l1+l5))+((l2+l6)+(l3+l7)), then
// the tail added in order. Lanes 0-3 and 4-7 live in two vec4s (component-wise IEEE ops).
// A is M x K, B is N x K (both row-major, row stride K), Y is M x N.
// push: 0 M, 1 N, 2 K, 3 a, 4 b, 5 y
layout(local_size_x = 16, local_size_y = 16) in;
const uint TS = 32u;
const uint BK = 16u; // a multiple of 8, so tiles hold whole chunks
shared float As[TS][BK + 1u];
shared float Bs[TS][BK + 1u];

float reduce8(vec4 lo, vec4 hi) {
    PRECISE vec4 u = lo + hi; // (l0+l4, l1+l5, l2+l6, l3+l7)
    PRECISE float s = (u.x + u.y) + (u.z + u.w);
    return s;
}

void main() {
    uint M = P(0), N = P(1), K = P(2), a = P(3), b = P(4), y = P(5);
    uint K8 = K / 8u * 8u;
    uint i0 = gl_WorkGroupID.y * TS, j0 = gl_WorkGroupID.x * TS;
    uint tx = gl_LocalInvocationID.x, ty = gl_LocalInvocationID.y;
    uint lid = ty * 16u + tx;
    PRECISE vec4 l00 = vec4(0.0), h00 = vec4(0.0), l01 = vec4(0.0), h01 = vec4(0.0);
    PRECISE vec4 l10 = vec4(0.0), h10 = vec4(0.0), l11 = vec4(0.0), h11 = vec4(0.0);
    for (uint k0 = 0u; k0 < K8; k0 += BK) {
        uint kn = min(BK, K8 - k0);
        for (uint e = lid; e < TS * BK; e += 256u) {
            uint ii = e / BK, tt = e % BK;
            float v = 0.0, w = 0.0;
            if (tt < kn) {
                if (i0 + ii < M) v = F[a + (i0 + ii) * K + k0 + tt];
                if (j0 + ii < N) w = F[b + (j0 + ii) * K + k0 + tt];
            }
            As[ii][tt] = v;
            Bs[ii][tt] = w;
        }
        barrier();
        for (uint c = 0u; c < kn; c += 8u) {
            vec4 a0l = vec4(As[ty][c], As[ty][c + 1u], As[ty][c + 2u], As[ty][c + 3u]);
            vec4 a0h = vec4(As[ty][c + 4u], As[ty][c + 5u], As[ty][c + 6u], As[ty][c + 7u]);
            vec4 a1l = vec4(As[ty + 16u][c], As[ty + 16u][c + 1u], As[ty + 16u][c + 2u], As[ty + 16u][c + 3u]);
            vec4 a1h = vec4(As[ty + 16u][c + 4u], As[ty + 16u][c + 5u], As[ty + 16u][c + 6u], As[ty + 16u][c + 7u]);
            vec4 b0l = vec4(Bs[tx][c], Bs[tx][c + 1u], Bs[tx][c + 2u], Bs[tx][c + 3u]);
            vec4 b0h = vec4(Bs[tx][c + 4u], Bs[tx][c + 5u], Bs[tx][c + 6u], Bs[tx][c + 7u]);
            vec4 b1l = vec4(Bs[tx + 16u][c], Bs[tx + 16u][c + 1u], Bs[tx + 16u][c + 2u], Bs[tx + 16u][c + 3u]);
            vec4 b1h = vec4(Bs[tx + 16u][c + 4u], Bs[tx + 16u][c + 5u], Bs[tx + 16u][c + 6u], Bs[tx + 16u][c + 7u]);
            l00 = l00 + a0l * b0l;
            h00 = h00 + a0h * b0h;
            l01 = l01 + a0l * b1l;
            h01 = h01 + a0h * b1h;
            l10 = l10 + a1l * b0l;
            h10 = h10 + a1h * b0h;
            l11 = l11 + a1l * b1l;
            h11 = h11 + a1h * b1h;
        }
        barrier();
    }
    uint ia = i0 + ty, ib = i0 + ty + 16u, ja = j0 + tx, jb = j0 + tx + 16u;
    if (ia < M && ja < N) {
        PRECISE float s = reduce8(l00, h00);
        for (uint t = K8; t < K; t++) s = s + F[a + ia * K + t] * F[b + ja * K + t];
        F[y + ia * N + ja] = s;
    }
    if (ia < M && jb < N) {
        PRECISE float s = reduce8(l01, h01);
        for (uint t = K8; t < K; t++) s = s + F[a + ia * K + t] * F[b + jb * K + t];
        F[y + ia * N + jb] = s;
    }
    if (ib < M && ja < N) {
        PRECISE float s = reduce8(l10, h10);
        for (uint t = K8; t < K; t++) s = s + F[a + ib * K + t] * F[b + ja * K + t];
        F[y + ib * N + ja] = s;
    }
    if (ib < M && jb < N) {
        PRECISE float s = reduce8(l11, h11);
        for (uint t = K8; t < K; t++) s = s + F[a + ib * K + t] * F[b + jb * K + t];
        F[y + ib * N + jb] = s;
    }
}
