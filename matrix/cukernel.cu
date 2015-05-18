#include <assert.h>
#include <stdio.h>
#include "generic/matrix.h"
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

__global__ void softmax_final(const float *a, float *b,
                        const float *max, const float *deno,
                        int nrow, int ncol, int stride, int mstride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    long idx;
    if (i >= nrow || j >= ncol) return;
    idx = j + i * stride;
    b[idx] = exp(a[idx] - max[0 + i * mstride]) / deno[0 + i * mstride];
}

__global__ void block_reduce_sum(const float *input, float *output,
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
        output[blockIdx.x + ostride * blockIdx.y] = arr[0];
}

__global__ void block_reduce_softmax_sum(const float *input, float *output,
                                        const float *max,
                                        const int istride, const int ostride,
                                        const int mstride, const int n) {
    extern __shared__ float arr[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    arr[threadIdx.x] = j < n ? exp(input[j + istride * blockIdx.y] - \
                                    max[0 + mstride * blockIdx.y]) : 0;
    __syncthreads();
    for (int offset = blockDim.x >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.x < offset)
            arr[threadIdx.x] += arr[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x + ostride * blockIdx.y] = arr[0];
}

__global__ void block_reduce_max(const float *input, float *output,
                        const int istride, const int ostride,
                        const int n) {
    extern __shared__ float arr[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    arr[threadIdx.x] = j < n ? input[j + istride * blockIdx.y] : 0;
    __syncthreads();
    for (int offset = blockDim.x >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            float l = arr[threadIdx.x],
                  r = arr[threadIdx.x + offset];
            if (r > l) arr[threadIdx.x] = r;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x + ostride * blockIdx.y] = arr[0];
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

    void cuda_colsum(const Matrix *a, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        float *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(float), a->nrow);
        block_reduce_sum<<<grid, block, block.x * sizeof(float)>>> \
            (a->data.f, res,
             a->stride / sizeof(float), stride / sizeof(float),
             ncol);
        ncol = blocks_per_row;
        assert(ncol <= block.x);
        grid.x = 1;
        block_reduce_sum<<<grid, block, block.x * sizeof(float)>>> \
            (res, b->data.f,
             stride / sizeof(float), b->stride / sizeof(float),
             ncol);
        cudaFree(res);
    }

    void cuda_softmax_final(const Matrix *a, const Matrix *max,
                            const Matrix *deno, Matrix *b) {
        dim3 threadsPerBlock(CUDA_THREADS_N,
                CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        softmax_final<<<numBlocks, threadsPerBlock>>>(a->data.f, b->data.f,
                max->data.f, deno->data.f,
                b->nrow, b->ncol,
                b->stride / sizeof(float),
                max->stride / sizeof(float));
    }

    void cuda_softmax_denominator(const Matrix *a, const Matrix *max, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        float *res;
        size_t stride;
        assert(max->ncol == 1);
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(float), a->nrow);
        block_reduce_softmax_sum<<<grid, block, block.x * sizeof(float)>>> \
            (a->data.f, res, max->data.f,
             a->stride / sizeof(float), stride / sizeof(float),
             max->stride / sizeof(float),
             ncol);
        ncol = blocks_per_row;
        assert(ncol <= block.x);
        grid.x = 1;
        block_reduce_sum<<<grid, block, block.x * sizeof(float)>>> \
            (res, b->data.f,
             stride / sizeof(float), b->stride / sizeof(float),
             ncol);
        cudaFree(res);
    }

    void cuda_colmax(const Matrix *a, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        float *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(float), a->nrow);
        block_reduce_max<<<grid, block, block.x * sizeof(float)>>> \
            (a->data.f, res,
             a->stride / sizeof(float), stride / sizeof(float),
             ncol);
        ncol = blocks_per_row;
        assert(ncol <= block.x);
        grid.x = 1;
        block_reduce_max<<<grid, block, block.x * sizeof(float)>>> \
            (res, b->data.f,
             stride / sizeof(float), b->stride / sizeof(float),
             ncol);
        cudaFree(res);
    }
}
