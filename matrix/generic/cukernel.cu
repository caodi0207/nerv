#ifdef NERV_GENERIC_CUKERNEL
#include <assert.h>
#include <stdio.h>
#include "matrix.h"
#include "cuda.h"
#include "float.h"
#define CUDA_THREADS_N 16
#define CUDA_THREADS_NN ((CUDA_THREADS_N) * (CUDA_THREADS_N))
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
__global__ void cudak_(log_elem)(const MATRIX_ELEM *a, MATRIX_ELEM *b, 
                                int nrow, int ncol, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    long idx;
    MATRIX_ELEM tmp;
    if (i >= nrow || j >= ncol) return;
    idx = j + i * stride;
    tmp = a[idx];
    if(tmp < FLT_MIN) tmp = FLT_MIN;
    b[idx] = log(tmp);
}

__global__ void cudak_(mul_elem)(const MATRIX_ELEM *a, const MATRIX_ELEM *b,
                                MATRIX_ELEM *c, 
                                int nrow, int ncol, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    long idx;
    if (i >= nrow || j >= ncol) return;
    idx = j + i * stride;
    c[idx] = a[idx] * b[idx];
}

__global__ void cudak_(sigmoid)(const MATRIX_ELEM *a, MATRIX_ELEM *b,
                        int nrow, int ncol, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    long idx;
    if (i >= nrow || j >= ncol) return;
    idx = j + i * stride;
    b[idx] = 1.0 / (1.0 + exp(-a[idx]));
}

__global__ void cudak_(sigmoid_grad)(const MATRIX_ELEM *output,
                                    const MATRIX_ELEM *err,
                                    MATRIX_ELEM *nerr,
                                    int nrow, int ncol, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    long idx;
    if (i >= nrow || j >= ncol) return;
    idx = j + i * stride;
    nerr[idx] = output[idx] * (1.0 - output[idx]) * err[idx];
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

__global__ void cudak_(block_reduce_rowsum)(const MATRIX_ELEM *input,
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

__global__ void cudak_(block_reduce_colsum)(const MATRIX_ELEM *input,
                                MATRIX_ELEM *output,
                                const int istride, const int ostride,
                                const int n) {
    extern __shared__ MATRIX_ELEM cudak_(arr)[];
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    cudak_(arr)[threadIdx.y] = i < n ? input[blockIdx.x + istride * i] : 0;
    __syncthreads();
    for (int offset = blockDim.y >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.y < offset)
            cudak_(arr)[threadIdx.y] += cudak_(arr)[threadIdx.y + offset];
        __syncthreads();
    }
    if (threadIdx.y == 0)
        output[blockIdx.x + ostride * blockIdx.y] = cudak_(arr)[0];
}

__global__ void cudak_(block_reduce_colsame)(const MATRIX_ELEM *input,
                                            const MATRIX_ELEM *ref_input,
                                            MATRIX_ELEM *output,
                                            const int istride, const int ostride,
                                            const int n) {
    extern __shared__ MATRIX_ELEM cudak_(arr)[];
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    cudak_(arr)[threadIdx.y] = (i < n && input[blockIdx.x + istride * i] == \
                                        ref_input[blockIdx.x + istride * i]) ? 1.0 : 0;
    __syncthreads();
    for (int offset = blockDim.y >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.y < offset)
            cudak_(arr)[threadIdx.y] += cudak_(arr)[threadIdx.y + offset];
        __syncthreads();
    }
    if (threadIdx.y == 0)
        output[blockIdx.x + ostride * blockIdx.y] = cudak_(arr)[0];
}

__global__ void cudak_(block_reduce_softmax_rowsum)(const MATRIX_ELEM *input,
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

__global__ void cudak_(block_reduce_rowmax)(const MATRIX_ELEM *input,
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
            if (r > l)
                cudak_(arr)[threadIdx.x] = r;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        output[blockIdx.x + ostride * blockIdx.y] = cudak_(arr)[0];
}

__global__ void cudak_(block_reduce_rowmax_idx)(const MATRIX_ELEM *input,
                                                const MATRIX_ELEM *idx_input,
                                                MATRIX_ELEM *output,
                                                MATRIX_ELEM *idx_output,
                                                const int istride, const int ostride,
                                                const int n) {
    extern __shared__ MATRIX_ELEM cudak_(arr)[];
    MATRIX_ELEM *arr_val = cudak_(arr);
    MATRIX_ELEM *arr_idx = arr_val + blockDim.x;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    arr_val[threadIdx.x] = j < n ? input[j + istride * blockIdx.y] : 0;
    arr_idx[threadIdx.x] = j < n ? idx_input[j + istride * blockIdx.y] : 0;
    __syncthreads();
    for (int offset = blockDim.x >> 1;  offset; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            MATRIX_ELEM l = arr_val[threadIdx.x],
                        r = arr_val[threadIdx.x + offset];
            if (r > l)
            {
                arr_val[threadIdx.x] = r;
                arr_idx[threadIdx.x] = arr_idx[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        output[blockIdx.x + ostride * blockIdx.y] = arr_val[0];
        idx_output[blockIdx.x + ostride * blockIdx.y] = arr_idx[0];
    }
}

__global__ void cudak_(add_row)(const MATRIX_ELEM *a, MATRIX_ELEM *b,
                                int nrow, int ncol, int stride, double beta) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nrow || j >= ncol) return;
    b[j + i * stride] += beta * a[j];
}

__global__ void cudak_(fill)(MATRIX_ELEM *a,
                            int nrow, int ncol, int stride, double val) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nrow || j >= ncol) return;
    a[j + i * stride] = val;
}

__global__ void cudak_(expand_frm)(const MATRIX_ELEM *a, MATRIX_ELEM *b,
                                    int nrow, int ncol,
                                    int enrow, int encol,
                                    int stride, int estride,
                                    int context) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int ridx;
    if (i >= enrow || j >= encol) return;
    ridx = i + j / ncol - context;
    if (ridx < 0) ridx = 0;
    else if (ridx >= nrow) ridx = nrow - 1;
    b[j + i * estride] = a[j % ncol + ridx * stride];
}

__global__ void cudak_(rearrange_frm)(const MATRIX_ELEM *a, MATRIX_ELEM *b,
                                    int nrow, int ncol,
                                    int stride, int step, int orig_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nrow || j >= ncol) return;
    b[j + i * stride] = a[j / step + (j % step) * orig_dim + i * stride];
}

__global__ void cudak_(scale_row)(const MATRIX_ELEM *a, MATRIX_ELEM *b,
                                    int nrow, int ncol,
                                    int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nrow || j >= ncol) return;
    b[j + i * stride] *= a[j];
}

__global__ void cudak_(decompress)(const MATRIX_ELEM *a, MATRIX_ELEM *b,
                                    int nrow, int ncol,
                                    int stride_a, int stride_b) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nrow || j >= ncol) return;
    b[lrintf(a[j + i * stride_a]) + i * stride_b] = 1.0;
}

__global__ void cudak_(gen_col_idx)(MATRIX_ELEM *b,
                                    int nrow, int ncol, int stride) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nrow || j >= ncol) return;
    b[j + i * stride] = j;
}

extern "C" {
#include "../cukernel.h"
    void cudak_(cuda_log_elem)(const Matrix *a, Matrix *b) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(log_elem)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b),
             b->nrow, b->ncol, b->stride / sizeof(MATRIX_ELEM));
    }

    void cudak_(cuda_mul_elem)(const Matrix *a, const Matrix *b,
                                Matrix *c) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(mul_elem)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b),
             MATRIX_ELEM_PTR(c),
             b->nrow, b->ncol, b->stride / sizeof(MATRIX_ELEM));
    }

    void cudak_(cuda_sigmoid)(const Matrix *a, Matrix *b) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(sigmoid)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b), b->nrow, b->ncol,
            b->stride / sizeof(MATRIX_ELEM));
    }

    void cudak_(cuda_sigmoid_grad)(const Matrix *output,
                                    const Matrix *err, Matrix *nerr) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(nerr->ncol, threadsPerBlock.x),
                CEIL_DIV(nerr->nrow, threadsPerBlock.y));
        cudak_(sigmoid_grad)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(output), MATRIX_ELEM_PTR(err),
             MATRIX_ELEM_PTR(nerr),
             nerr->nrow, nerr->ncol,
             nerr->stride / sizeof(MATRIX_ELEM));
    }

    void cudak_(cuda_rowsum)(const Matrix *a, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        MATRIX_ELEM *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(MATRIX_ELEM), a->nrow);
        cudak_(block_reduce_rowsum)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), res,
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             ncol);
        ncol = blocks_per_row;
        assert((unsigned long)ncol <= block.x);
        grid.x = 1;
        cudak_(block_reduce_rowsum)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (res, MATRIX_ELEM_PTR(b),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             ncol);
        cudaFree(res);
    }

    void cudak_(cuda_colsame)(const Matrix *a, const Matrix *ref, Matrix *b) {
        dim3 block(1, CUDA_THREADS_NN);
        int nrow = a->nrow;
        int blocks_per_col = CEIL_DIV(nrow, block.y);
        dim3 grid(a->ncol, blocks_per_col);
        MATRIX_ELEM *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, a->ncol * sizeof(MATRIX_ELEM), blocks_per_col);
        cudak_(block_reduce_colsame)<<<grid, block, block.y * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(ref), res,
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             nrow);
        nrow = blocks_per_col;
        assert((unsigned long)nrow <= block.y);
        grid.y = 1;
        cudak_(block_reduce_colsum)<<<grid, block, block.y * sizeof(MATRIX_ELEM)>>> \
            (res, MATRIX_ELEM_PTR(b),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             nrow);
        cudaFree(res);
    }

    void cudak_(cuda_colsum)(const Matrix *a, Matrix *b) {
        dim3 block(1, CUDA_THREADS_NN);
        int nrow = a->nrow;
        int blocks_per_col = CEIL_DIV(nrow, block.y);
        dim3 grid(a->ncol, blocks_per_col);
        MATRIX_ELEM *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, a->ncol * sizeof(MATRIX_ELEM), blocks_per_col);
        cudak_(block_reduce_colsum)<<<grid, block, block.y * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), res,
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             nrow);
        nrow = blocks_per_col;
        assert((unsigned long)nrow <= block.y);
        grid.y = 1;
        cudak_(block_reduce_colsum)<<<grid, block, block.y * sizeof(MATRIX_ELEM)>>> \
            (res, MATRIX_ELEM_PTR(b),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             nrow);
        cudaFree(res);
    }

    void cudak_(cuda_softmax_final)(const Matrix *a, const Matrix *max,
                            const Matrix *deno, Matrix *b) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
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
        cudak_(block_reduce_softmax_rowsum) \
            <<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), res, MATRIX_ELEM_PTR(max),
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             max->stride / sizeof(MATRIX_ELEM),
             ncol);
        ncol = blocks_per_row;
        assert((unsigned long)ncol <= block.x);
        grid.x = 1;
        cudak_(block_reduce_rowsum) \
            <<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (res, MATRIX_ELEM_PTR(b),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             ncol);
        cudaFree(res);
    }

    void cudak_(cuda_rowmax)(const Matrix *a, Matrix *b) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        MATRIX_ELEM *res;
        size_t stride;
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(MATRIX_ELEM), a->nrow);
        cudak_(block_reduce_rowmax)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), res,
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             ncol);
        ncol = blocks_per_row;
        assert((unsigned long)ncol <= block.x);
        grid.x = 1;
        cudak_(block_reduce_rowmax)<<<grid, block, block.x * sizeof(MATRIX_ELEM)>>> \
            (res, MATRIX_ELEM_PTR(b),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             ncol);
        cudaFree(res);
    }

    void cudak_(cuda_rowmax_idx)(const Matrix *a, Matrix *b, Matrix *b_idx) {
        dim3 block(CUDA_THREADS_NN, 1);
        int ncol = a->ncol;
        int blocks_per_row = CEIL_DIV(ncol, block.x);
        dim3 grid(blocks_per_row, a->nrow);
        MATRIX_ELEM *a_idx, *res, *res_idx;
        size_t stride;
        cudaMallocPitch(&a_idx, &stride, a->stride, a->nrow);
        cudak_(gen_col_idx)<<<grid, block>>>(a_idx, a->nrow, ncol, stride / sizeof(MATRIX_ELEM));
        cudaMallocPitch(&res, &stride, blocks_per_row * sizeof(MATRIX_ELEM), a->nrow);
        cudaMallocPitch(&res_idx, &stride, blocks_per_row * sizeof(MATRIX_ELEM), a->nrow);
        cudak_(block_reduce_rowmax_idx)<<<grid, block,
                                        2 * block.x * sizeof(MATRIX_ELEM)>>> \
            (MATRIX_ELEM_PTR(a), a_idx, res, res_idx,
             a->stride / sizeof(MATRIX_ELEM), stride / sizeof(MATRIX_ELEM),
             ncol);
        cudaFree(a_idx);
        ncol = blocks_per_row;
        assert((unsigned long)ncol <= block.x);
        grid.x = 1;
        cudak_(block_reduce_rowmax_idx)<<<grid, block,
                                        2 * block.x * sizeof(MATRIX_ELEM)>>> \
            (res, res_idx, MATRIX_ELEM_PTR(b), MATRIX_ELEM_PTR(b_idx),
             stride / sizeof(MATRIX_ELEM), b->stride / sizeof(MATRIX_ELEM),
             ncol);
        cudaFree(res);
        cudaFree(res_idx);
    }

    /* in-place calc */
    void cudak_(cuda_add_row)(const Matrix *a, Matrix *b, double beta) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(add_row)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b), b->nrow, b->ncol,
            b->stride / sizeof(MATRIX_ELEM), beta);
    }

    void cudak_(cuda_fill)(Matrix *a, double val) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(a->ncol, threadsPerBlock.x),
                CEIL_DIV(a->nrow, threadsPerBlock.y));
        cudak_(fill)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), a->nrow, a->ncol,
            a->stride / sizeof(MATRIX_ELEM), val);
    }

    void cudak_(cuda_expand_frm)(const Matrix *a, Matrix *b, int context) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(expand_frm)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b),
             a->nrow, a->ncol,
             b->nrow, b->ncol,
             a->stride / sizeof(MATRIX_ELEM),
             b->stride / sizeof(MATRIX_ELEM),
             context);
    }

    void cudak_(cuda_rearrange_frm)(const Matrix *a, Matrix *b, int step) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(rearrange_frm)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b),
             b->nrow, b->ncol, b->stride / sizeof(MATRIX_ELEM),
             step, b->ncol / step);
    }

    void cudak_(cuda_scale_row)(const Matrix *a, Matrix *b) {
        dim3 threadsPerBlock(CUDA_THREADS_N, CUDA_THREADS_N);
        dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                CEIL_DIV(b->nrow, threadsPerBlock.y));
        cudak_(scale_row)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b),
             b->nrow, b->ncol, b->stride / sizeof(MATRIX_ELEM));
    }

    void cudak_(cuda_decompress)(const Matrix *a, Matrix *b) {
        dim3 threadsPerBlock(1, CUDA_THREADS_NN);
        dim3 numBlocks(1, CEIL_DIV(a->nrow, threadsPerBlock.y));
        cudak_(decompress)<<<numBlocks, threadsPerBlock>>> \
            (MATRIX_ELEM_PTR(a), MATRIX_ELEM_PTR(b),
             a->nrow, a->ncol,
             a->stride / sizeof(MATRIX_ELEM),
             b->stride / sizeof(MATRIX_ELEM));
    }
}
#endif
