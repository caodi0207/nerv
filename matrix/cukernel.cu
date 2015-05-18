#include <assert.h>
#include "generic/matrix.h"
#include <stdio.h>
#include "cuda.h"
#define CUDA_THREADS_N 16
#define CUDA_THREADS_NN (16 * 16)
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
__global__ void sigmoid(const float *a, float *b,
                        int nrow, int ncol, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    long idx;
    if (i >= nrow || j >= ncol) return;
    idx = j + i * stride;
    b[idx] = 1.0 / (1.0 + exp(-a[idx]));
}

__global__ void block_sum(const float *input, float *output,
                        const int istride, const int ostride,
                        const int n) {
    extern __shared__ float arr[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    arr[threadIdx.x] = j < n ? input[j + istride * blockIdx.y] : 0;
    __syncthreads();
    for (int offset = blockDim.x >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.x < offset)
            arr[threadIdx.x] += arr[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        /* printf("bx: %d by: %d arr: %f\n", blockIdx.x, blockIdx.y, arr[0]); */
        output[blockIdx.x + ostride * blockIdx.y] = arr[0];
    }
}

extern "C" {
    void cuda_sigmoid(const Matrix *a, Matrix *b) {
        dim3 threadsPerBlock(CUDA_THREADS_N,
                CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        sigmoid<<<numBlocks, threadsPerBlock>>>(a->data.f, b->data.f, b->nrow, b->ncol,
                b->stride / sizeof(float));
    }

    void cuda_rowsum(const Matrix *a, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        float *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(float), a->nrow);
        block_sum<<<grid, block, block.x * sizeof(float)>>> \
            (a->data.f, res,
             a->stride / sizeof(float), stride / sizeof(float),
             ncol);
        ncol = blocks_per_row;
        assert(ncol <= block.x);
        grid.x = 1;
        block_sum<<<grid, block, block.x * sizeof(float)>>> \
            (res, b->data.f,
             stride / sizeof(float), b->stride / sizeof(float),
             ncol);
        cudaFree(res);
    }
}
