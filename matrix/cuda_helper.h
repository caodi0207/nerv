#ifndef NERV_CUDA_HELPER_H
#define NERV_CUDA_HELPER_H
#define CUBLAS_SAFE_CALL(call) \
    do { \
        cublasStatus_t  err = (call); \
        if (err != CUBLAS_STATUS_SUCCESS) \
            nerv_error(L, "cumatrix cublas error: %s", cublasGetErrorString(err)); \
    } while (0)

#define CUDA_SAFE_CALL(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) \
            nerv_error(L, "cumatrix CUDA error: %s", cudaGetErrorString(err)); \
    } while (0)

#define CUDA_SAFE_SYNC_CALL(call) \
    do { \
        CUDA_SAFE_CALL(call); \
        cudaDeviceSynchronize(); \
    } while (0)

#define CHECK_SAME_DIMENSION(a, b) \
    do { \
        if (!(a->nrow == b->nrow && a->ncol == b->ncol)) \
            nerv_error(L, "matrices should be of the same dimension"); \
    } while (0)

static const char *cublasGetErrorString(cublasStatus_t err) {
    switch (err)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}
#endif
