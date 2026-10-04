#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <cstddef>

#include <cuda_runtime.h>
#include "block_gemm_cuda.h"

#define CHECK_ERROR(X)                                              \
    do {                                                            \
        cudaError_t err_ = (X);                                     \
        if (err_ != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: '%s' -> %s\n",    \
                    __FILE__, __LINE__, #X,                         \
                    cudaGetErrorString(err_));                      \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

namespace {

constexpr int BLOCK_M = 64;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 16;
constexpr int THREAD_M = 4;
constexpr int THREAD_N = 4;
constexpr int THREADS_X = BLOCK_N / THREAD_N;
constexpr int THREADS_Y = BLOCK_M / THREAD_M;
constexpr int THREADS_TOTAL = THREADS_X * THREADS_Y;

constexpr int DIV_A  = BLOCK_K / 4;  // float4 per row of A tile
constexpr int DIV_B  = BLOCK_N / 4;  // float4 per row of B tile

constexpr int LDS_A  = (BLOCK_M * BLOCK_K / 4) / THREADS_TOTAL;
constexpr int LDS_B  = (BLOCK_K * BLOCK_N / 4) / THREADS_TOTAL;
static_assert(LDS_A == 1,
              "This simplified loader assumes one float4 per thread per A tile");
static_assert(LDS_B == 1,
              "This simplified loader assumes one float4 per thread per B tile");

__host__ __device__ __forceinline__
std::size_t matIdx(int row, int col, int n) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(n)
         + static_cast<std::size_t>(col);
}

__device__ __forceinline__ const float4& vec4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

__device__ __forceinline__ float4& vec4(float* p) {
    return *reinterpret_cast<float4*>(p);
}

__global__ void __launch_bounds__(THREADS_TOTAL)
blockGemmKernel(const float* __restrict__ a,
                const float* __restrict__ b,
                float* __restrict__ c,
                int n) {
    __shared__  __align__(16) float As[BLOCK_M][BLOCK_K];
    __shared__  __align__(16) float Bs[BLOCK_K][BLOCK_N];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int tid = ty * THREADS_X + tx;

    float acc[THREAD_M][THREAD_N] = {};

    const int rowA   = tid / DIV_A;  // 0 .. BLOCK_M-1
    const int chunkA = tid % DIV_A;

    const int rowB   = tid / DIV_B;  // 0 .. BLOCK_K-1
    const int chunkB = tid % DIV_B;

    for (int k0 = 0; k0 < n; k0 += BLOCK_K) {
        // matrix A
        {
            const float* aPtr = a
                + static_cast<std::size_t>(blockIdx.y * BLOCK_M + rowA) * n
                + k0;
            vec4(&As[rowA][chunkA * 4]) = vec4(aPtr + chunkA * 4);
        }

        // matrix B
        {
            const float* bPtr = b
                + static_cast<std::size_t>(k0 + rowB) * n
                + blockIdx.x * BLOCK_N;
            vec4(&Bs[rowB][chunkB * 4]) = vec4(bPtr + chunkB * 4);
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            float af[THREAD_M];
            float bf[THREAD_N];
#pragma unroll
            for (int i = 0; i < THREAD_M; ++i)
                af[i] = As[ty * THREAD_M + i][k];
#pragma unroll
            for (int j = 0; j < THREAD_N; ++j)
                bf[j] = Bs[k][tx * THREAD_N + j];
#pragma unroll
            for (int i = 0; i < THREAD_M; ++i)
#pragma unroll
                for (int j = 0; j < THREAD_N; ++j)
                    acc[i][j] = fmaf(af[i], bf[j], acc[i][j]);
        }

        __syncthreads();
    }

    const int row0 = blockIdx.y * BLOCK_M + ty * THREAD_M;
    const int col0 = blockIdx.x * BLOCK_N + tx * THREAD_N;
    float* crow = c + matIdx(row0, col0, n);
#pragma unroll
    for (int i = 0; i < THREAD_M; ++i) {
        const float4 v = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        vec4(crow + static_cast<std::size_t>(i) * n) = v;
    }
}

// n < BLOCK_M
__global__ void naiveGemmKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ c,
                                int n) {
    const int r   = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n || col >= n) return;

    float s = 0.f;
    for (int k = 0; k < n; ++k)
        s += a[matIdx(r, k, n)] * b[matIdx(k, col, n)];
    c[matIdx(r, col, n)] = s;
}

} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    std::vector<float> c(static_cast<std::size_t>(n) * n, 0.f);
    const std::size_t bytes = static_cast<std::size_t>(n) * n * sizeof(float);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    if (n % BLOCK_M == 0) {
        blockGemmKernel<<<dim3(n / BLOCK_N, n / BLOCK_M), dim3(THREADS_X, THREADS_Y)>>>
            (d_a, d_b, d_c, n);
    } else {
        naiveGemmKernel<<<dim3((n + 15) / 16, (n + 15) / 16), dim3(16, 16)>>>
            (d_a, d_b, d_c, n);
    }

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}