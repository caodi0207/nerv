#ifdef NERV_GENERIC_CUMATRIX
#include "matrix.h"
#include "elem_type.h"

#define MATRIX_DATA_FREE(L, ptr) cuda_matrix_(free)(L, ptr)
#define MATRIX_DATA_ALLOC(L, dptr, stride, width, height) \
                            cuda_matrix_(alloc)(L, dptr, stride, width, height)
#define MATRIX_DATA_WRITE(L, data, idx, val) cuda_matrix_(write)(L, data, idx, val)
#define MATRIX_DATA_READ(L, data, idx) cuda_matrix_(read)(L, data, idx)
#define MATRIX_INIT(L) cuda_matrix_(init)(L)
#define MATRIX_BASE_TNAME nerv_matrix_cuda_tname
#define NERV_GENERIC_MATRIX
#define NERV_GENERIC_CUKERNEL
#include "../../common.h"
#include "../cukernel.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "driver_types.h"
#include "cublas_v2.h"
#include "../cuda_helper.h"

static cublasHandle_t cublas_handle;

Matrix *nerv_matrix_(new_)(lua_State *L, long nrow, long ncol);
void nerv_matrix_(data_free)(lua_State *L, Matrix *self);

static void nerv_matrix_(add_)(lua_State *L, const Matrix *a, const Matrix *b,
                                const Matrix *c,
                                MATRIX_ELEM alpha, MATRIX_ELEM beta) {
    CUBLAS_SAFE_CALL(
            NERV_CUBLAS_(geam)(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                a->ncol, a->nrow,
                &alpha,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM)));
}

static int nerv_matrix_(add)(lua_State *L) {
    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    MATRIX_ELEM alpha = luaL_checknumber(L, 4); /* alpha */
    MATRIX_ELEM beta = luaL_checknumber(L, 5); /* alpha */
    CHECK_SAME_DIMENSION(a, b);
    CHECK_SAME_DIMENSION(a, c);
    nerv_matrix_(add_)(L, a, b, c, alpha, beta);
    return 0;
}

static int nerv_matrix_(get_cublas_op)(char ch) {
    return (ch == 'T' || ch == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
}

static int nerv_matrix_(mul)(lua_State *L) {
#define SWAP(a, b) \
    do { int t = (a); (a) = (b); (b) = t; } while (0)

    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    MATRIX_ELEM alpha = luaL_checknumber(L, 4);
    MATRIX_ELEM beta = luaL_checknumber(L, 5);
    int nargs = lua_gettop(L);
    int ta = nargs > 5 ? nerv_matrix_(get_cublas_op)(*luaL_checkstring(L, 6)) \
                            : CUBLAS_OP_N;
    int tb = nargs > 6 ? nerv_matrix_(get_cublas_op)(*luaL_checkstring(L, 7)) \
                            : CUBLAS_OP_N;
    int am = a->nrow, an = a->ncol;
    int bm = b->nrow, bn = b->ncol;
    if (ta == CUBLAS_OP_T) SWAP(am, an);
    if (tb == CUBLAS_OP_T) SWAP(bm, bn);
    if (an != bm)
        nerv_error(L, "Wrong dimension of multipliers");
/*    MATRIX_ELEM alpha = 1.0f, beta = 0.0f; */
    CUBLAS_SAFE_CALL(
            NERV_CUBLAS_(gemm)(cublas_handle, tb, ta,
                bn, am, bm,
                &alpha,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM)));
    return 0;
}

static int nerv_matrix_(create)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, a->nrow, a->ncol);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(sigmoid)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    cudak_(cuda_sigmoid)(b, a);
    return 0;
}

static int nerv_matrix_(sigmoid_grad)(lua_State *L) {
    Matrix *nerr = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *err = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *output = luaT_checkudata(L, 3, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(nerr, err);
    CHECK_SAME_DIMENSION(nerr, output);
    cudak_(cuda_sigmoid_grad)(output, err, nerr);
    return 0;
}

static int nerv_matrix_(softmax)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *max = nerv_matrix_(new_)(L, a->nrow, 1);
    Matrix *dno = nerv_matrix_(new_)(L, a->nrow, 1);
    CHECK_SAME_DIMENSION(a, b);
    cudak_(cuda_rowmax)(a, max);
    cudak_(cuda_softmax_denominator)(a, max, dno);
    cudak_(cuda_softmax_final)(a, max, dno, b);
    nerv_matrix_(data_free)(L, max);
    nerv_matrix_(data_free)(L, dno);
    return 0;
}

static int nerv_matrix_(rowsum)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, a->nrow, 1);
    cudak_(cuda_rowsum)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(colsum)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, 1, a->ncol);
    cudak_(cuda_colsum)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(rowmax)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, a->nrow, 1);
    cudak_(cuda_rowmax)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}


static int nerv_matrix_(add_row)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    double beta = luaL_checknumber(L, 3);
    if (a->ncol != b->ncol)
        nerv_error(L, "the number of columns is not the same");
    if (a->nrow != 1)
        nerv_error(L, "a row vector is expected");
    cudak_(cuda_add_row)(a, b, beta);
    return 0;
}

static int nerv_matrix_(fill)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    double val = luaL_checknumber(L, 2);
    cudak_(cuda_fill)(self, val);
    return 0;
}

static int nerv_matrix_(copy_fromd)(lua_State *L) { 
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ELEM_PTR(a), a->stride,
                MATRIX_ELEM_PTR(b), b->stride,
                sizeof(MATRIX_ELEM) * b->ncol, b->nrow,
                cudaMemcpyDeviceToDevice));
    return 0;
}

static int nerv_matrix_(copy_tod)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ELEM_PTR(b), b->stride,
                MATRIX_ELEM_PTR(a), a->stride,
                sizeof(MATRIX_ELEM) * a->ncol, a->nrow,
                cudaMemcpyDeviceToDevice));
    return 0;
}

extern const char *MATRIX_CUMATRIX_HOST_TNAME;
static int nerv_matrix_(copy_fromh)(lua_State *L) { 
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    CHECK_SAME_DIMENSION(a, b);
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ELEM_PTR(a), a->stride,
                MATRIX_ELEM_PTR(b), b->stride,
                sizeof(MATRIX_ELEM) * b->ncol, b->nrow,
                cudaMemcpyHostToDevice));
    return 0;
}

static int nerv_matrix_(copy_toh)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    CHECK_SAME_DIMENSION(a, b);
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ELEM_PTR(b), b->stride,
                MATRIX_ELEM_PTR(a), a->stride,
                sizeof(MATRIX_ELEM) * a->ncol, a->nrow,
                cudaMemcpyDeviceToHost));
    return 0;
}

static int nerv_matrix_(trans)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, a->ncol, a->nrow);
    MATRIX_ELEM alpha = 1, beta = 0;
    CUBLAS_SAFE_CALL(
            NERV_CUBLAS_(geam)(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                a->nrow, a->ncol,
                &alpha,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM)));
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(mul_elem)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    CHECK_SAME_DIMENSION(a, c);
    cudak_(cuda_mul_elem)(a, b, c);
    return 0;
}

static int nerv_matrix_(log_elem)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    cudak_(cuda_log_elem)(a, b);
    return 0;
}

extern const char *nerv_matrix_host_int_tname;
static int nerv_matrix_(copy_rows_fromh_by_idx)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    Matrix *idx = luaT_checkudata(L, 3, nerv_matrix_host_int_tname);
    long *idx_ptr = idx->data.i;
    int i;
    long nrow = a->nrow;
    if (idx->nrow != 1)
        nerv_error(L, "index should be a vector");
    if (idx->ncol != nrow)
        nerv_error(L, "index dimension mismatch");
    if (a->ncol != b->ncol)
        nerv_error(L, "source/destination dimension mismatch");
    cudaStream_t *streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nrow);
    for (i = 0; i < nrow; i++)
    {
        CUDA_SAFE_CALL(cudaStreamCreate(streams + i));
        CUDA_SAFE_CALL(cudaMemcpyAsync(MATRIX_ROW_PTR(a, i),
                    MATRIX_ROW_PTR(b, idx_ptr[i]),
                    b->stride,
                    cudaMemcpyHostToDevice, streams[i]));
    }
    for (i = 0; i < nrow; i++)
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(streams[i]));
        CUDA_SAFE_CALL(cudaStreamDestroy(streams[i]));
    }
    return 0;
}

static const luaL_Reg nerv_matrix_(extra_methods)[] = {
    {"create", nerv_matrix_(create)},
    {"colsum", nerv_matrix_(colsum)},
    {"rowsum", nerv_matrix_(rowsum)},
    {"rowmax", nerv_matrix_(rowmax)},
    {"trans", nerv_matrix_(trans)},
    /* in-place calc */
    {"copy_fromh", nerv_matrix_(copy_fromh)},
    {"copy_fromd", nerv_matrix_(copy_fromd)},
    {"copy_toh", nerv_matrix_(copy_toh)},
    {"copy_tod", nerv_matrix_(copy_tod)},
    {"add", nerv_matrix_(add)},
    {"mul", nerv_matrix_(mul)},
    {"add_row", nerv_matrix_(add_row)},
    {"fill", nerv_matrix_(fill)},
    {"sigmoid", nerv_matrix_(sigmoid)},
    {"sigmoid_grad", nerv_matrix_(sigmoid_grad)},
    {"softmax", nerv_matrix_(softmax)},
    {"mul_elem", nerv_matrix_(mul_elem)},
    {"log_elem", nerv_matrix_(log_elem)},
    {"copy_rows_fromh_by_idx", nerv_matrix_(copy_rows_fromh_by_idx)},
    {NULL, NULL}
};

static void cuda_matrix_(init)(lua_State *L) {
    luaN_append_methods(L, nerv_matrix_(extra_methods));
    cublasCreate(&cublas_handle);
}

static void cuda_matrix_(free)(lua_State *L, MATRIX_ELEM *ptr) {
    CUDA_SAFE_SYNC_CALL(cudaFree(ptr));
}

static void cuda_matrix_(alloc)(lua_State *L, MATRIX_ELEM **dptr,
                                size_t *stride, long width, long height) {
    CUDA_SAFE_SYNC_CALL(cudaMallocPitch((void **)dptr, stride, width, height));
}

static MATRIX_ELEM cuda_matrix_(read)(lua_State *L, MATRIX_ELEM *data,
                                        int idx) {
    MATRIX_ELEM res;
    CUDA_SAFE_SYNC_CALL(cudaMemcpy(&res, data + idx,
                sizeof(MATRIX_ELEM), cudaMemcpyDeviceToHost));
    return res;
}

static void cuda_matrix_(write)(lua_State *L, MATRIX_ELEM *data,
                                int idx, MATRIX_ELEM val) {
    CUDA_SAFE_SYNC_CALL(cudaMemcpy(data + idx, &val,
                sizeof(MATRIX_ELEM), cudaMemcpyHostToDevice));
}

int nerv_matrix_(get_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

int nerv_matrix_(set_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

#include "matrix.c"
#endif
