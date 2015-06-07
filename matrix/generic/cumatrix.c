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
#define PROFILE_HASHMAP_SIZE 123457
#include "../../common.h"
#include "../cukernel.h"
#include "../cuda_helper.h"
#include <string.h>

Matrix *nerv_matrix_(new_)(lua_State *L, long nrow, long ncol);
void nerv_matrix_(data_free)(lua_State *L, Matrix *self);

static void nerv_matrix_(add_)(lua_State *L, const Matrix *a, const Matrix *b,
                                const Matrix *c,
                                MATRIX_ELEM alpha, MATRIX_ELEM beta) {
    PROFILE_START
    CUBLAS_SAFE_SYNC_CALL(
            NERV_CUBLAS_(geam)(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                a->ncol, a->nrow,
                &alpha,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM)));
    PROFILE_STOP
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
    /* Because matrix in Nerv is row-major, here b comes first */
    PROFILE_START
    CUBLAS_SAFE_SYNC_CALL(
            NERV_CUBLAS_(gemm)(cublas_handle, tb, ta,
                bn, am, bm,
                &alpha,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM)));
    PROFILE_STOP
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
    PROFILE_START
    cudak_(cuda_sigmoid)(b, a);
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(sigmoid_grad)(lua_State *L) {
    Matrix *nerr = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *err = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *output = luaT_checkudata(L, 3, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(nerr, err);
    CHECK_SAME_DIMENSION(nerr, output);
    PROFILE_START
    cudak_(cuda_sigmoid_grad)(output, err, nerr);
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(softmax)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *max, *max_idx;
    Matrix *dno;
    CHECK_SAME_DIMENSION(a, b);
    max = nerv_matrix_(new_)(L, a->nrow, 1);
    max_idx = nerv_matrix_(new_)(L, a->nrow, 1);
    dno = nerv_matrix_(new_)(L, a->nrow, 1);
    PROFILE_START
    cudak_(cuda_rowmax_idx)(a, max, max_idx);
    cudak_(cuda_softmax_denominator)(a, max, dno);
    cudak_(cuda_softmax_final)(a, max, dno, b);
    PROFILE_STOP
    nerv_matrix_(data_free)(L, max);
    nerv_matrix_(data_free)(L, dno);
    luaT_pushudata(L, max_idx, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(rowsum)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, a->nrow, 1);
    PROFILE_START
    cudak_(cuda_rowsum)(a, b);
    PROFILE_STOP
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(colsum)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, 1, a->ncol);
    PROFILE_START
    cudak_(cuda_colsum)(a, b);
    PROFILE_STOP
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(colsame)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *ref = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, 1, a->ncol);
    CHECK_SAME_DIMENSION(a, ref);
    PROFILE_START
    cudak_(cuda_colsame)(a, ref, b);
    PROFILE_STOP
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(rowmax)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, a->nrow, 1);
    PROFILE_START
    cudak_(cuda_rowmax)(a, b);
    PROFILE_STOP
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(rowmax_idx)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, a->nrow, 1);
    Matrix *idx = nerv_matrix_(new_)(L, a->nrow, 1);
    PROFILE_START
    cudak_(cuda_rowmax_idx)(a, b, idx);
    PROFILE_STOP
    luaT_pushudata(L, b, nerv_matrix_(tname));
    luaT_pushudata(L, idx, nerv_matrix_(tname));
    return 2;
}

static int nerv_matrix_(add_row)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    double beta = luaL_checknumber(L, 3);
    if (a->ncol != b->ncol)
        nerv_error(L, "the number of columns is not the same");
    if (a->nrow != 1)
        nerv_error(L, "a row vector is expected");
    PROFILE_START
    cudak_(cuda_add_row)(a, b, beta);
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(fill)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    double val = luaL_checknumber(L, 2);
    PROFILE_START
    cudak_(cuda_fill)(self, val);
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(copy_fromd)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    int nargs = lua_gettop(L);
    int b_begin = nargs > 2 ? luaL_checkinteger(L, 3) : 0;
    int b_end = nargs > 3 ? luaL_checkinteger(L, 4) : b->nrow;
    int a_begin = nargs > 4 ? luaL_checkinteger(L, 5) : 0;
    if (!(0 <= b_begin && b_begin < b_end && b_end <= b->nrow &&
            a_begin + b_end - b_begin <= a->nrow))
        nerv_error(L, "invalid copy interval");
    if (a->ncol != b->ncol)
        nerv_error(L, "matrices should be of the same dimension");
    PROFILE_START
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ROW_PTR(a, a_begin), a->stride,
                MATRIX_ROW_PTR(b, b_begin), b->stride,
                sizeof(MATRIX_ELEM) * b->ncol, b_end - b_begin,
                cudaMemcpyDeviceToDevice));
    PROFILE_STOP
    return 0;
}

extern const char *MATRIX_CUMATRIX_HOST_TNAME;
static int nerv_matrix_(copy_fromh)(lua_State *L) { 
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    int nargs = lua_gettop(L);
    int b_begin = nargs > 2 ? luaL_checkinteger(L, 3) : 0;
    int b_end = nargs > 3 ? luaL_checkinteger(L, 4) : b->nrow;
    int a_begin = nargs > 4 ? luaL_checkinteger(L, 5) : 0;
    if (!(0 <= b_begin && b_begin < b_end && b_end <= b->nrow &&
            a_begin + b_end - b_begin <= a->nrow))
        nerv_error(L, "invalid copy interval");
    if (a->ncol != b->ncol)
        nerv_error(L, "matrices should be of the same dimension");
    PROFILE_START
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ROW_PTR(a, a_begin), a->stride,
                MATRIX_ROW_PTR(b, b_begin), b->stride,
                sizeof(MATRIX_ELEM) * b->ncol, b_end - b_begin,
                cudaMemcpyHostToDevice));
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(copy_toh)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    int nargs = lua_gettop(L);
    int a_begin = nargs > 2 ? luaL_checkinteger(L, 3) : 0;
    int a_end = nargs > 3 ? luaL_checkinteger(L, 4) : a->nrow;
    int b_begin = nargs > 4 ? luaL_checkinteger(L, 5) : 0;
    if (!(0 <= a_begin && a_begin < a_end && a_end <= a->nrow &&
            b_begin + a_end - a_begin <= b->nrow))
        nerv_error(L, "invalid copy interval");
    if (b->ncol != a->ncol)
        nerv_error(L, "matrices should be of the same dimension");
    PROFILE_START
    CUDA_SAFE_SYNC_CALL(
            cudaMemcpy2D(MATRIX_ROW_PTR(b, b_begin), b->stride,
                MATRIX_ROW_PTR(a, a_begin), a->stride,
                sizeof(MATRIX_ELEM) * a->ncol, a_end - a_begin,
                cudaMemcpyDeviceToHost));
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(trans)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(L, a->ncol, a->nrow);
    MATRIX_ELEM alpha = 1, beta = 0;
    /* FIXME: possible memory leak when lua error is raised */
    PROFILE_START
    CUBLAS_SAFE_SYNC_CALL(
            NERV_CUBLAS_(geam)(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                a->nrow, a->ncol,
                &alpha,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM)));
    PROFILE_STOP
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(mul_elem)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    CHECK_SAME_DIMENSION(a, c);
    PROFILE_START
    cudak_(cuda_mul_elem)(a, b, c);
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(log_elem)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    PROFILE_START
    cudak_(cuda_log_elem)(a, b);
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(decompress)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b;
    int orig_col = luaL_checkinteger(L, 2);
    if (a->ncol != 1)
        nerv_error(L, "the compressed matrix must be a column vector");
    b = nerv_matrix_(new_)(L, a->nrow, orig_col);
    PROFILE_START
    cudak_(cuda_fill)(b, 0.0);
    cudak_(cuda_decompress)(a, b);
    PROFILE_STOP
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

extern const char *nerv_matrix_host_int_tname;
static int nerv_matrix_(copy_rows_fromh_by_idx)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    Matrix *idx = luaT_checkudata(L, 3, nerv_matrix_host_int_tname);
    long nrow = a->nrow;
    int b_begin = lua_gettop(L) > 3 ? luaL_checkinteger(L, 4) : 0;
    if (!(0 <= b_begin && b_begin + nrow <= idx->ncol))
        nerv_error(L, "invalid copy interval");
    long *idx_ptr = idx->data.i;
    int i;
    if (idx->nrow != 1)
        nerv_error(L, "index should be a vector");
    if (a->ncol != b->ncol)
        nerv_error(L, "source/destination dimension mismatch");
    cudaStream_t *streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nrow);
    for (i = 0; i < nrow; i++)
    {
        int src_row = idx_ptr[b_begin + i];
        if (!(0 <= src_row && src_row < b->nrow))
            nerv_error(L, "invalid index");
        CUDA_SAFE_CALL(cudaStreamCreate(streams + i));
        CUDA_SAFE_CALL(cudaMemcpyAsync(MATRIX_ROW_PTR(a, i),
                    MATRIX_ROW_PTR(b, src_row),
                    b->stride,
                    cudaMemcpyHostToDevice, streams[i]));
    }
    for (i = 0; i < nrow; i++)
    {
        CUDA_SAFE_CALL(cudaStreamSynchronize(streams[i]));
        CUDA_SAFE_CALL(cudaStreamDestroy(streams[i]));
    }
    free(streams);
    return 0;
}

static int nerv_matrix_(expand_frm)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    int context = luaL_checkinteger(L, 3);
    if (a->nrow != b->nrow)
        nerv_error(L, "mismatching number of frames");
    if (a->ncol != b->ncol * (context * 2 + 1))
        nerv_error(L, "the width should be 2 * context + 1");
    PROFILE_START
    cudak_(cuda_expand_frm)(b, a, context);
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(rearrange_frm)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    int step = luaL_checkinteger(L, 3);
    CHECK_SAME_DIMENSION(a, b);
    if (b->ncol % step)
        nerv_error(L, "the dimension of columns is not divisible by step");
    PROFILE_START
    cudak_(cuda_rearrange_frm)(b, a, step);
    PROFILE_STOP
    return 0;
}

static int nerv_matrix_(scale_row)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    if (a->ncol != b->ncol)
        nerv_error(L, "the number of columns is not the same");
    if (b->nrow != 1)
        nerv_error(L, "a row vector is expected");
    PROFILE_START
    cudak_(cuda_scale_row)(b, a);
    PROFILE_STOP
    return 0;
}

static const luaL_Reg nerv_matrix_(extra_methods)[] = {
    {"create", nerv_matrix_(create)},
    {"colsum", nerv_matrix_(colsum)},
    {"colsame", nerv_matrix_(colsame)},
    {"rowsum", nerv_matrix_(rowsum)},
    {"rowmax", nerv_matrix_(rowmax)},
    {"rowmax_idx", nerv_matrix_(rowmax_idx)},
    {"trans", nerv_matrix_(trans)},
    {"decompress", nerv_matrix_(decompress)},
    /* in-place calc */
    {"copy_fromh", nerv_matrix_(copy_fromh)},
    {"copy_fromd", nerv_matrix_(copy_fromd)},
    {"copy_toh", nerv_matrix_(copy_toh)},
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
    {"expand_frm", nerv_matrix_(expand_frm)},
    {"rearrange_frm", nerv_matrix_(rearrange_frm)},
    {"scale_row", nerv_matrix_(scale_row)},
    {NULL, NULL}
};

static void cuda_matrix_(init)(lua_State *L) {
    luaN_append_methods(L, nerv_matrix_(extra_methods));
    cublasCreate(&cublas_handle);
    cudaEventCreate(&profile_start);
    cudaEventCreate(&profile_stop);
    profile = hashmap_create(PROFILE_HASHMAP_SIZE, bkdr_hash, strcmp);
}

static void cuda_matrix_(free)(lua_State *L, MATRIX_ELEM *ptr) {
    CUDA_SAFE_SYNC_CALL(cudaFree(ptr));
}

static void cuda_matrix_(alloc)(lua_State *L, MATRIX_ELEM **dptr,
                                size_t *stride, long width, long height) {
    PROFILE_START
    CUDA_SAFE_SYNC_CALL(cudaMallocPitch((void **)dptr, stride, width, height));
    PROFILE_STOP
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
