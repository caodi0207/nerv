#ifdef NERV_GENERIC_CUMATRIX
#include "matrix.h"
#include "elem_type.h"
#define MATRIX_DATA_FREE(ptr, status) cuda_matrix_(free)(ptr, status)
#define MATRIX_DATA_ALLOC(dptr, stride, width, height, status) \
                            cuda_matrix_(alloc)(dptr, stride, width, height, status)

#define NERV_GENERIC_MATRIX
#define NERV_GENERIC_CUKERNEL
#include "../../common.h"
#include "../cukernel.h"
#include "../cuda_helper.h"

void nerv_matrix_(add)(Matrix *c, const Matrix *a, const Matrix *b,
                            MATRIX_ELEM alpha, MATRIX_ELEM beta,
                            Status *status) {
    CHECK_SAME_DIMENSION(a, b, status);
    CHECK_SAME_DIMENSION(a, c, status);
    PROFILE_START
    CUBLAS_SAFE_SYNC_CALL(
            NERV_CUBLAS_(geam)(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                a->ncol, a->nrow,
                &alpha,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM)),
            status);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(mul)(Matrix *c, const Matrix *a, const Matrix *b,
                            MATRIX_ELEM alpha, MATRIX_ELEM beta,
                            int ta, int tb, Status *status) {
#define SWAP(a, b) \
    do { int t = (a); (a) = (b); (b) = t; } while (0)

    int am = a->nrow, an = a->ncol;
    int bm = b->nrow, bn = b->ncol;
    if (ta == CUBLAS_OP_T) SWAP(am, an);
    if (tb == CUBLAS_OP_T) SWAP(bm, bn);
    if (an != bm)
        NERV_EXIT_STATUS(status, MAT_WRONG_MULT_DIM, 0);
    /* Because matrix in Nerv is row-major, here b comes first */
    PROFILE_START
    CUBLAS_SAFE_SYNC_CALL(
            NERV_CUBLAS_(gemm)(cublas_handle, tb, ta,
                bn, am, bm,
                &alpha,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM)),
            status);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(sigmoid)(Matrix *a, const Matrix *b, Status *status) {
    CHECK_SAME_DIMENSION(a, b, status);
    PROFILE_START
    cudak_(cuda_sigmoid)(b, a);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(sigmoid_grad)(Matrix *nerr, const Matrix *err,
                                const Matrix *output, Status *status) {
    CHECK_SAME_DIMENSION(nerr, err, status);
    CHECK_SAME_DIMENSION(nerr, output, status);
    PROFILE_START
    cudak_(cuda_sigmoid_grad)(output, err, nerr);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

Matrix *nerv_matrix_(softmax)(Matrix *b, const Matrix *a, Status *status) {
    Matrix *max, *max_idx;
    Matrix *dno;
    CHECK_SAME_DIMENSION_RET(a, b, status);
    max = nerv_matrix_(create)(a->nrow, 1, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    max_idx = nerv_matrix_(create)(a->nrow, 1, status);
    if (status->err_code != NERV_NORMAL)
    {
        nerv_matrix_(destroy)(max, status);
        return NULL;
    }
    dno = nerv_matrix_(create)(a->nrow, 1, status);
    if (status->err_code != NERV_NORMAL)
    {   /* FIXME: destroy may also fail? */
        nerv_matrix_(destroy)(max, status);
        nerv_matrix_(destroy)(max_idx, status);
        return NULL;
    }
    PROFILE_START
    cudak_(cuda_rowmax_idx)(a, max, max_idx);
    cudak_(cuda_softmax_denominator)(a, max, dno);
    cudak_(cuda_softmax_final)(a, max, dno, b);
    PROFILE_STOP
    nerv_matrix_(destroy)(max, status);
    nerv_matrix_(destroy)(dno, status);
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return max_idx;
}

Matrix *nerv_matrix_(rowsum)(Matrix *a, Status *status) {
    Matrix *b = nerv_matrix_(create)(a->nrow, 1, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    PROFILE_START
    cudak_(cuda_rowsum)(a, b);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return b;
}

Matrix *nerv_matrix_(colsum)(Matrix *a, Status *status) {
    Matrix *b = nerv_matrix_(create)(1, a->ncol, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    PROFILE_START
    cudak_(cuda_colsum)(a, b);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return b;
}

Matrix *nerv_matrix_(colsame)(Matrix *a, const Matrix *ref,
                                Status *status) {
    Matrix *b = nerv_matrix_(create)(1, a->ncol, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    CHECK_SAME_DIMENSION_RET(a, ref, status);
    PROFILE_START
    cudak_(cuda_colsame)(a, ref, b);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return b;
}

Matrix *nerv_matrix_(rowmax)(Matrix *a, Status *status) {
    Matrix *b = nerv_matrix_(create)(a->nrow, 1, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    PROFILE_START
    cudak_(cuda_rowmax)(a, b);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return b;
}

void nerv_matrix_(rowmax_idx)(Matrix *a, Matrix **b, Matrix **idx,
                                Status *status) {
    *b = nerv_matrix_(create)(a->nrow, 1, status);
    if (status->err_code != NERV_NORMAL)
        return;
    *idx = nerv_matrix_(create)(a->nrow, 1, status);
    if (status->err_code != NERV_NORMAL)
    {
        /* FIXME: destroy may also fail? */
        nerv_matrix_(destroy)(*b, status);
        return;
    }
    PROFILE_START
    cudak_(cuda_rowmax_idx)(a, *b, *idx);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(add_row)(Matrix *b, const Matrix *a, double beta,
                            Status *status) {
    if (a->ncol != b->ncol)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    if (a->nrow != 1)
        NERV_EXIT_STATUS(status, MAT_ROW_VECTOR_EXP, 0);
    PROFILE_START
    cudak_(cuda_add_row)(a, b, beta);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(fill)(Matrix *self, double val, Status *status) {
    PROFILE_START
    cudak_(cuda_fill)(self, val);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(copy_fromd)(Matrix *a, const Matrix *b,
                            int a_begin, int b_begin, int b_end,
                            Status *status) {
    if (!(0 <= b_begin && b_begin < b_end && b_end <= b->nrow &&
            a_begin + b_end - b_begin <= a->nrow))
        NERV_EXIT_STATUS(status, MAT_INVALID_COPY_INTERVAL, 0);
    if (a->ncol != b->ncol)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    PROFILE_START
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ROW_PTR(a, a_begin), a->stride,
                MATRIX_ROW_PTR(b, b_begin), b->stride,
                sizeof(MATRIX_ELEM) * b->ncol, b_end - b_begin,
                cudaMemcpyDeviceToDevice),
            status);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(copy_fromh)(Matrix *a, const Matrix *b,
                            int a_begin, int b_begin, int b_end,
                            Status *status) { 
    if (!(0 <= b_begin && b_begin < b_end && b_end <= b->nrow &&
            a_begin + b_end - b_begin <= a->nrow))
        NERV_EXIT_STATUS(status, MAT_INVALID_COPY_INTERVAL, 0);
    if (a->ncol != b->ncol)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    PROFILE_START
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ROW_PTR(a, a_begin), a->stride,
                MATRIX_ROW_PTR(b, b_begin), b->stride,
                sizeof(MATRIX_ELEM) * b->ncol, b_end - b_begin,
                cudaMemcpyHostToDevice),
            status);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(copy_toh)(Matrix *a, const Matrix *b,
                            int a_begin, int a_end, int b_begin,
                            Status *status) {
    if (!(0 <= a_begin && a_begin < a_end && a_end <= a->nrow &&
            b_begin + a_end - a_begin <= b->nrow))
        NERV_EXIT_STATUS(status, MAT_INVALID_COPY_INTERVAL, 0);
    if (b->ncol != a->ncol)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    PROFILE_START
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ROW_PTR(b, b_begin), b->stride,
                MATRIX_ROW_PTR(a, a_begin), a->stride,
                sizeof(MATRIX_ELEM) * a->ncol, a_end - a_begin,
                cudaMemcpyDeviceToHost),
            status);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

Matrix *nerv_matrix_(trans)(Matrix *a, Status *status) {
    MATRIX_ELEM alpha = 1, beta = 0;
    Matrix *b = nerv_matrix_(create)(a->ncol, a->nrow, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    /* FIXME: possible memory leak when lua error is raised */
    PROFILE_START
    CUBLAS_SAFE_SYNC_CALL_RET(
            NERV_CUBLAS_(geam)(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                a->nrow, a->ncol,
                &alpha,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM)),
            status);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return b;
}

void nerv_matrix_(mul_elem)(Matrix *c, const Matrix *a, const Matrix *b,
                            Status *status) {
    CHECK_SAME_DIMENSION(a, b, status);
    CHECK_SAME_DIMENSION(a, c, status);
    PROFILE_START
    cudak_(cuda_mul_elem)(a, b, c);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(log_elem)(Matrix *b, const Matrix *a, Status *status) {
    CHECK_SAME_DIMENSION(a, b, status);
    PROFILE_START
    cudak_(cuda_log_elem)(a, b);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

Matrix *nerv_matrix_(decompress)(const Matrix *a, int orig_col, Status *status) {
    Matrix *b;
    if (a->ncol != 1)
    {
        NERV_SET_STATUS(status, MAT_COL_VECTOR_EXP, 0);
        return NULL;
    }
    b = nerv_matrix_(create)(a->nrow, orig_col, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    PROFILE_START
    cudak_(cuda_fill)(b, 0.0);
    cudak_(cuda_decompress)(a, b);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return b;
}

void nerv_matrix_(copy_rows_fromh_by_idx)(Matrix *a, const Matrix *b,
                            const Matrix *idx, int b_begin, Status *status) {
    long nrow = a->nrow;
    if (!(0 <= b_begin && b_begin + nrow <= idx->ncol))
        NERV_EXIT_STATUS(status, MAT_INVALID_COPY_INTERVAL, 0);
    long *idx_ptr = idx->data.i;
    int i;
    if (idx->nrow != 1)
        NERV_EXIT_STATUS(status, MAT_IDX_VECTOR_EXP, 0);
    if (a->ncol != b->ncol)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    cudaStream_t *streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nrow);
    for (i = 0; i < nrow; i++)
    {
        int src_row = idx_ptr[b_begin + i];
        if (!(0 <= src_row && src_row < b->nrow))
            NERV_EXIT_STATUS(status, MAT_INVALID_IDX, 0);
        CUDA_SAFE_CALL(cudaStreamCreate(streams + i), status);
        CUDA_SAFE_CALL(cudaMemcpyAsync(MATRIX_ROW_PTR(a, i),
                    MATRIX_ROW_PTR(b, src_row),
                    b->stride,
                    cudaMemcpyHostToDevice, streams[i]), status);
    }
    for (i = 0; i < nrow; i++)
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(streams[i]), status);
        CUDA_SAFE_CALL(cudaStreamDestroy(streams[i]), status);
    }
    free(streams);
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(expand_frm)(Matrix *a, const Matrix *b,
                            int context, Status *status) {
    if (a->nrow != b->nrow)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    if (a->ncol != b->ncol * (context * 2 + 1))
        NERV_EXIT_STATUS(status, MAT_GENERAL_ERR,
                        "the width should be 2 * context + 1");
    PROFILE_START
    cudak_(cuda_expand_frm)(b, a, context);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(rearrange_frm)(Matrix *a, const Matrix *b,
                                int step, Status *status) {
    CHECK_SAME_DIMENSION(a, b, status);
    if (b->ncol % step)
        NERV_EXIT_STATUS(status, MAT_GENERAL_ERR,
                        "the dimension of columns is not divisible by step");
    PROFILE_START
    cudak_(cuda_rearrange_frm)(b, a, step);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(scale_rows_by_col)(Matrix *a, const Matrix *b,
                                    Status *status) {
    if (a->nrow != b->nrow)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    if (b->ncol != 1)
        NERV_EXIT_STATUS(status, MAT_COL_VECTOR_EXP, 0);
    PROFILE_START
    cudak_(cuda_scale_rows_by_col)(b, a);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(scale_rows_by_row)(Matrix *a, const Matrix *b,
                                    Status *status) {
    if (a->ncol != b->ncol)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    if (b->nrow != 1)
        NERV_EXIT_STATUS(status, MAT_ROW_VECTOR_EXP, 0);
    PROFILE_START
    cudak_(cuda_scale_rows_by_row)(b, a);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

static void cuda_matrix_(free)(MATRIX_ELEM *ptr, Status *status) {
    CUDA_SAFE_SYNC_CALL(cudaFree(ptr), status);
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

static void cuda_matrix_(alloc)(MATRIX_ELEM **dptr,
                                size_t *stride, long width, long height,
                                Status *status) {
    PROFILE_START
    CUDA_SAFE_SYNC_CALL(cudaMallocPitch((void **)dptr, stride, width, height),
                        status);
    PROFILE_STOP
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

#include "matrix.c"
#endif
