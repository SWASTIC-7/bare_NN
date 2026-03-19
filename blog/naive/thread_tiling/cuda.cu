#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return 1; \
    } \
} while (0)

constexpr int WARP_SIZE = 32;
constexpr int MAX_TT = 4;

struct Config {
    int BT;
    int TK;
    int WTX;
    int WTY;
    int TTX;
    int TTY;
};

__global__ void thread_tiled_cuda_runtime(
    const float* A,
    const float* B,
    float* C,
    int N,
    int BT,
    int TK,
    int WTX,
    int WTY,
    int TTX,
    int TTY)
{
    extern __shared__ float smem[];
    float* shA = smem;
    float* shB = smem + (BT * TK);

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const int thread_cols = BT / TTX;
    const int thread_rows = BT / TTY;
    const int warp_tiles_per_row = thread_cols / WTX;

    const int warp_row = warp_id / warp_tiles_per_row;
    const int warp_col = warp_id % warp_tiles_per_row;

    const int inner_row = warp_row * WTY + lane_id / WTX;
    const int inner_col = warp_col * WTX + lane_id % WTX;

    const int thread_row_base = inner_row * TTY;
    const int thread_col_base = inner_col * TTX;

    const int row_base = blockIdx.y * BT + thread_row_base;
    const int col_base = blockIdx.x * BT + thread_col_base;

    float value[MAX_TT][MAX_TT];
    for (int r = 0; r < MAX_TT; ++r) {
        for (int c = 0; c < MAX_TT; ++c) {
            value[r][c] = 0.0f;
        }
    }

    for (int kb = 0; kb < N; kb += TK) {
        for (int idx = tid; idx < BT * TK; idx += blockDim.x) {
            const int r = idx / TK;
            const int c = idx % TK;
            const int gr = blockIdx.y * BT + r;
            const int gc = kb + c;
            shA[idx] = (gr < N && gc < N) ? A[gr * N + gc] : 0.0f;
        }

        for (int idx = tid; idx < TK * BT; idx += blockDim.x) {
            const int r = idx / BT;
            const int c = idx % BT;
            const int gr = kb + r;
            const int gc = blockIdx.x * BT + c;
            shB[idx] = (gr < N && gc < N) ? B[gr * N + gc] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TK; ++k) {
            for (int rr = 0; rr < TTY; ++rr) {
                const int a_row = thread_row_base + rr;
                const float a = (a_row < BT) ? shA[a_row * TK + k] : 0.0f;
                for (int cc = 0; cc < TTX; ++cc) {
                    const int b_col = thread_col_base + cc;
                    const float b = (b_col < BT) ? shB[k * BT + b_col] : 0.0f;
                    value[rr][cc] = fmaf(a, b, value[rr][cc]);
                }
            }
        }

        __syncthreads();
    }

    for (int rr = 0; rr < TTY; ++rr) {
        for (int cc = 0; cc < TTX; ++cc) {
            const int gr = row_base + rr;
            const int gc = col_base + cc;
            if (gr < N && gc < N) {
                C[gr * N + gc] = value[rr][cc];
            }
        }
    }
}

int main()
{
    const int N = 1024;

    Config candidates[] = {
        {16, 16, 16, 2, 1, 1},
        {16, 32, 16, 2, 1, 1},
        {16, 16, 16, 2, 2, 1},
        {16, 32, 16, 2, 2, 1},
        {16, 16, 8, 4, 1, 1},
        {16, 32, 8, 4, 1, 1},
        {16, 16, 8, 4, 2, 2},
        {16, 32, 8, 4, 2, 2},
        {32, 16, 16, 2, 1, 1},
        {32, 32, 16, 2, 1, 1},
        {32, 16, 8, 4, 2, 2},
        {32, 32, 8, 4, 2, 2},
    };

    float *A = nullptr, *B = nullptr, *C = nullptr;
    CUDA_CHECK(cudaMalloc(&A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C, N * N * sizeof(float)));

    float* hA = new float[N * N];
    float* hB = new float[N * N];
    float* hC = new float[N * N];

    for (int i = 0; i < N * N; ++i) hA[i] = sinf((float)i);
    for (int i = 0; i < N * N; ++i) hB[i] = cosf((float)i);

    CUDA_CHECK(cudaMemcpy(A, hA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hB, N * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double best_gflops = -1.0;
    float best_ms = 0.0f;
    double best_max_error = 0.0;
    double best_checksum = 0.0;
    Config best = {0, 0, 0, 0, 0, 0};

    int tested = 0;
    int valid = 0;

    std::printf("Autotuning CUDA thread-tiling for N=%d\n", N);

    for (const Config& cfg : candidates) {
        tested++;

        if (cfg.TTX > MAX_TT || cfg.TTY > MAX_TT) continue;
        if (cfg.WTX * cfg.WTY != 32) continue;
        if (cfg.BT % cfg.TTX != 0 || cfg.BT % cfg.TTY != 0) continue;

        const int thread_cols = cfg.BT / cfg.TTX;
        const int thread_rows = cfg.BT / cfg.TTY;

        if (thread_cols % cfg.WTX != 0 || thread_rows % cfg.WTY != 0) continue;

        const int warp_tiles_per_row = thread_cols / cfg.WTX;
        const int warp_tiles_per_col = thread_rows / cfg.WTY;
        const int total_warps = warp_tiles_per_row * warp_tiles_per_col;
        const int threads = total_warps * WARP_SIZE;

        if (threads <= 0 || threads > 1024) continue;

        const size_t sharedMem = (cfg.BT * cfg.TK + cfg.TK * cfg.BT) * sizeof(float);
        dim3 block(threads, 1, 1);
        dim3 grid((N + cfg.BT - 1) / cfg.BT, (N + cfg.BT - 1) / cfg.BT, 1);

        CUDA_CHECK(cudaMemset(C, 0, N * N * sizeof(float)));

        thread_tiled_cuda_runtime<<<grid, block, sharedMem>>>(
            A, B, C, N,
            cfg.BT, cfg.TK, cfg.WTX, cfg.WTY, cfg.TTX, cfg.TTY);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int rep = 0; rep < 5; ++rep) {
            thread_tiled_cuda_runtime<<<grid, block, sharedMem>>>(
                A, B, C, N,
                cfg.BT, cfg.TK, cfg.WTX, cfg.WTY, cfg.TTX, cfg.TTY);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        const float ms = total_ms / 5.0f;

        CUDA_CHECK(cudaMemcpy(hC, C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        double checksum = 0.0;
        for (int idx = 0; idx < N * N; ++idx) checksum += (double)hC[idx];

        double max_error = 0.0;
        bool ok = true;
        for (int row = 0; row < 64 && ok; ++row) {
            for (int col = 0; col < 64; ++col) {
                double cpu = 0.0;
                for (int kk = 0; kk < N; ++kk) {
                    cpu += (double)hA[row * N + kk] * (double)hB[kk * N + col];
                }
                const double diff = std::fabs(cpu - (double)hC[row * N + col]);
                if (diff > max_error) max_error = diff;
                if (diff > 1e-2) {
                    ok = false;
                    break;
                }
            }
        }

        if (!ok) {
            std::printf("Skip invalid: BT=%d TK=%d WTX=%d WTY=%d TTX=%d TTY=%d\n",
                        cfg.BT, cfg.TK, cfg.WTX, cfg.WTY, cfg.TTX, cfg.TTY);
            continue;
        }

        valid++;
        const double gflops = (2.0 * (double)N * (double)N * (double)N) / ((double)ms * 1e6);
        std::printf("BT=%d TK=%d WTX=%d WTY=%d TTX=%d TTY=%d | %.3f ms | %.2f GFLOPS | err=%e\n",
                    cfg.BT, cfg.TK, cfg.WTX, cfg.WTY, cfg.TTX, cfg.TTY, ms, gflops, max_error);

        if (gflops > best_gflops) {
            best_gflops = gflops;
            best_ms = ms;
            best = cfg;
            best_max_error = max_error;
            best_checksum = checksum;
        }
    }

    if (best_gflops < 0.0) {
        std::printf("No valid config found. Tried=%d\n", tested);
        return 1;
    }

    std::printf("\nBest config:\n");
    std::printf("BT=%d TK=%d WTX=%d WTY=%d TTX=%d TTY=%d\n",
                best.BT, best.TK, best.WTX, best.WTY, best.TTX, best.TTY);
    std::printf("Checksum:     %.10e\n", best_checksum);
    std::printf("Max error:    %.10e\n", best_max_error);
    std::printf("Time:         %.3f ms\n", best_ms);
    std::printf("Performance:  %.2f GFLOPS\n", best_gflops);
    std::printf("Tried=%d Valid=%d\n", tested, valid);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    delete[] hA;
    delete[] hB;
    delete[] hC;
    return 0;
}
