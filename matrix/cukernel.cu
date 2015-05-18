#include "generic/matrix.h"
#define CUDA_THREADS_N 16
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

extern "C" void cuda_sigmoid(const Matrix *a, Matrix *b) {
    dim3 threadsPerBlock(CUDA_THREADS_N,
                         CUDA_THREADS_N);
    dim3 numBlocks(CEIL_DIV(b->ncol, threadsPerBlock.x),
                    CEIL_DIV(b->nrow, threadsPerBlock.y));
    sigmoid<<<numBlocks, threadsPerBlock>>>(a->data.f, b->data.f, b->nrow, b->ncol,
                                            b->stride / sizeof(float));
}
