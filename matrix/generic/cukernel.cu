#ifdef NERV_GENERIC_CUKERNEL
#include <assert.h>
#include <stdio.h>
#include "matrix.h"
#include "cuda.h"
#define CUDA_THREADS_N 16
#define CUDA_THREADS_NN (16 * 16)
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
__global__ void cudak_(sigmoid)(const MATRIX_ELEM *a, MATRIX_ELEM *b,
                        int nrow, int ncol, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    long idx;
    if (i >= nrow || j >= ncol) return;
    idx = j + i * stride;
    b[idx] = 1.0 / (1.0 + exp(-a[idx]));
}

__global__ void cudak_(softmax_final)(const MATRIX_ELEM *a, MATRIX_ELEM *b,
                        const MATRIX_ELEM *max, const MATRIX_ELEM *deno,
                        int nrow, int ncol, int stride, int mstride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    long idx;
    if (i >= nrow || j >= ncol) return;
    idx = j + i * stride;
    b[idx] = exp(a[idx] - max[0 + i * mstride]) / deno[0 + i * mstride];
}

__global__ void cudak_(block_reduce_sum)(const MATRIX_ELEM *input,
                                MATRIX_ELEM *output,
                                const int istride, const int ostride,
                                const int n) {
    extern __shared__ MATRIX_ELEM cudak_(arr)[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    cudak_(arr)[threadIdx.x] = j < n ? input[j + istride * blockIdx.y] : 0;
    __syncthreads();
    for (int offset = blockDim.x >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.x < offset)
            cudak_(arr)[threadIdx.x] += cudak_(arr)[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x + ostride * blockIdx.y] = cudak_(arr)[0];
}

__global__ void cudak_(block_reduce_softmax_sum)(const MATRIX_ELEM *input,
                                        MATRIX_ELEM *output,
                                        const MATRIX_ELEM *max,
                                        const int istride, const int ostride,
                                        const int mstride, const int n) {
    extern __shared__ MATRIX_ELEM cudak_(arr)[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    cudak_(arr)[threadIdx.x] = j < n ? exp(input[j + istride * blockIdx.y] - \
                                    max[0 + mstride * blockIdx.y]) : 0;
    __syncthreads();
    for (int offset = blockDim.x >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.x < offset)
            cudak_(arr)[threadIdx.x] += cudak_(arr)[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x + ostride * blockIdx.y] = cudak_(arr)[0];
}

__global__ void cudak_(block_reduce_max)(const MATRIX_ELEM *input,
                                MATRIX_ELEM *output,
                                const int istride, const int ostride,
                                const int n) {
    extern __shared__ MATRIX_ELEM cudak_(arr)[];
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    cudak_(arr)[threadIdx.x] = j < n ? input[j + istride * blockIdx.y] : 0;
    __syncthreads();
    for (int offset = blockDim.x >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            MATRIX_ELEM l = cudak_(arr)[threadIdx.x],
                  r = cudak_(arr)[threadIdx.x + offset];
            if (r > l) cudak_(arr)[threadIdx.x] = r;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x + ostride * blockIdx.y] = cudak_(arr)[0];
}

extern "C" {
#include "../cukernel.h"
    void cudak_(cuda_sigmoid)(const Matrix *a, Matrix *b) {
        dim3 threadsPerBlock(CUDA_THREADS_N,
                CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(sigmoid)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b), b->nrow, b->ncol,
            b->stride / sizeof(MATRIX_ELEM));
    }

    void cudak_(cuda_colsum)(const Matrix *a, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        MATRIX_ELEM *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(MATRIX_ELEM), a->nrow);
        cudak_(block_reduce_sum)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), res,
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             ncol);
        ncol = blocks_per_row;
        assert((unsigned long)ncol <= block.x);
        grid.x = 1;
        cudak_(block_reduce_sum)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (res, MATRIX_ELEM_PTR(b),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             ncol);
        cudaFree(res);
    }

    void cudak_(cuda_softmax_final)(const Matrix *a, const Matrix *max,
                            const Matrix *deno, Matrix *b) {
        dim3 threadsPerBlock(CUDA_THREADS_N,
                CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(softmax_final)<<<numBlocks, threadsPerBlock>>> \
                (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b),
                MATRIX_ELEM_PTR(max), MATRIX_ELEM_PTR(deno),
                b->nrow, b->ncol,
                b->stride / sizeof(MATRIX_ELEM),
                max->stride / sizeof(MATRIX_ELEM));
    }

    void cudak_(cuda_softmax_denominator)(const Matrix *a, const Matrix *max, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        MATRIX_ELEM *res;
        size_t stride;
        assert(max->ncol == 1);
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(MATRIX_ELEM), a->nrow);
        cudak_(block_reduce_softmax_sum)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), res, MATRIX_ELEM_PTR(max),
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             max->stride / sizeof(MATRIX_ELEM),
             ncol);
        ncol = blocks_per_row;
        assert((unsigned long)ncol <= block.x);
        grid.x = 1;
        cudak_(block_reduce_sum)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (res, MATRIX_ELEM_PTR(b),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             ncol);
        cudaFree(res);
    }

    void cudak_(cuda_colmax)(const Matrix *a, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        MATRIX_ELEM *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(MATRIX_ELEM), a->nrow);
        cudak_(block_reduce_max)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), res,
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             ncol);
        ncol = blocks_per_row;
        assert((unsigned long)ncol <= block.x);
        grid.x = 1;
        cudak_(block_reduce_max)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (res, MATRIX_ELEM_PTR(b),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             ncol);
        cudaFree(res);
    }
}
#endif