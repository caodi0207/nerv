#ifndef NERV_CUDA_HELPER_H
#define NERV_CUDA_HELPER_H
#include "cuda.h"
#include "cuda_runtime.h"
#include "driver_types.h"
#include "cublas_v2.h"

#define CUBLAS_SAFE_SYNC_CALL_RET(call, status) \
    do { \
        cublasStatus_t  err = (call); \
        if (err != CUBLAS_STATUS_SUCCESS) \
        { \
            NERV_SET_STATUS(status, MAT_CUBLAS_ERR, cublasGetErrorString(err)); \
            return 0; \
        } \
        cudaDeviceSynchronize(); \
    } while (0)

#define CUBLAS_SAFE_SYNC_CALL(call, status) \
    do { \
        cublasStatus_t  err = (call); \
        if (err != CUBLAS_STATUS_SUCCESS) \
            NERV_EXIT_STATUS(status, MAT_CUBLAS_ERR, cublasGetErrorString(err)); \
        cudaDeviceSynchronize(); \
    } while (0)

#define CUDA_SAFE_CALL_RET(call, status) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) \
        { \
            NERV_SET_STATUS(status, MAT_CUDA_ERR, cudaGetErrorString(err)); \
            return 0; \
        } \
    } while (0)

#define CUDA_SAFE_CALL(call, status) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) \
            NERV_EXIT_STATUS(status, MAT_CUDA_ERR, cudaGetErrorString(err)); \
    } while (0)

#define CUDA_SAFE_SYNC_CALL(call, status) \
    do { \
        CUDA_SAFE_CALL(call, status); \
        cudaDeviceSynchronize(); \
    } while (0)

#define CUDA_SAFE_SYNC_CALL_RET(call, status) \
    do { \
        CUDA_SAFE_CALL_RET(call, status); \
        cudaDeviceSynchronize(); \
    } while (0)

#define CHECK_SAME_DIMENSION(a, b, status) \
    do { \
        if (!(a->nrow == b->nrow && a->ncol == b->ncol)) \
            NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0); \
    } while (0)

#define CHECK_SAME_DIMENSION_RET(a, b, status) \
    do { \
        if (!(a->nrow == b->nrow && a->ncol == b->ncol)) \
        { \
            NERV_SET_STATUS(status, MAT_MISMATCH_DIM, 0); \
            return 0; \
        } \
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
/*        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR"; */
    }
    return "<unknown>";
}

#define PROFILE_START \
    do { \
        cudaEventRecord(profile_start, 0);
#define PROFILE_STOP \
        cudaEventRecord(profile_stop, 0); \
        cudaEventSynchronize(profile_stop); \
        float milliseconds = 0; \
        cudaEventElapsedTime(&milliseconds, profile_start, profile_stop); \
        accu_profile(__func__, milliseconds / 1000); \
    } while (0);

#define PROFILE_END
#endif
