#ifdef NERV_GENERIC_CUMATRIX
#include "matrix.h"
#include "elem_type.h"

#define MATRIX_DATA_FREE(ptr) cuda_matrix_(free)(ptr)
#define MATRIX_DATA_ALLOC(dptr, stride, width, height) \
                            cuda_matrix_(alloc)(dptr, stride, width, height)
#define MATRIX_DATA_WRITE(data, idx, val) cuda_matrix_(write)(data, idx, val)
#define MATRIX_DATA_READ(data, idx) cuda_matrix_(read)(data, idx)
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

#define CHECK_SAME_DIMENSION(a, b) \
    do { \
        if (!(a->nrow == b->nrow && a->ncol == b->ncol)) \
            nerv_error(L, "Matrices should be of the same dimension"); \
    } while (0)

static cublasHandle_t cublas_handle;

Matrix *nerv_matrix_(new_)(long nrow, long ncol);

static void nerv_matrix_(add_)(const Matrix *a, const Matrix *b,
                                const Matrix *c,
                                MATRIX_ELEM alpha, MATRIX_ELEM beta) {
    NERV_CUBLAS_(geam)(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                a->ncol, a->nrow,
                &alpha,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM));
}

static int nerv_matrix_(add)(lua_State *L) {
    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    MATRIX_ELEM alpha = luaL_checknumber(L, 4); /* alpha */
    MATRIX_ELEM beta = luaL_checknumber(L, 5); /* alpha */
    CHECK_SAME_DIMENSION(a, b);
    nerv_matrix_(add_)(a, b, c, alpha, beta);
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
    NERV_CUBLAS_(gemm)(cublas_handle, tb, ta,
                bn, am, bm,
                &alpha,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM));
    return 0;
}

static int nerv_matrix_(create)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    fprintf(stderr, "create\n");
    Matrix *b = nerv_matrix_(new_)(a->nrow, a->ncol);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(sigmoid)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    cudak_(cuda_sigmoid)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
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
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *max = nerv_matrix_(new_)(a->nrow, 1);
    Matrix *dno = nerv_matrix_(new_)(a->nrow, 1);
    Matrix *b = nerv_matrix_(new_)(a->nrow, a->ncol);
    cudak_(cuda_rowmax)(a, max);
    cudak_(cuda_softmax_denominator)(a, max, dno);
    cudak_(cuda_softmax_final)(a, max, dno, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(rowsum)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(a->nrow, 1);
    cudak_(cuda_rowsum)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(colsum)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(1, a->ncol);
    cudak_(cuda_colsum)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(rowmax)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(a->nrow, 1);
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
    cudaMemcpy2D(MATRIX_ELEM_PTR(a), a->stride,
                MATRIX_ELEM_PTR(b), b->stride,
                sizeof(MATRIX_ELEM) * b->ncol, b->nrow,
                cudaMemcpyDeviceToDevice);
    return 0;
}

static int nerv_matrix_(copy_tod)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    CHECK_SAME_DIMENSION(a, b);
    cudaMemcpy2D(MATRIX_ELEM_PTR(b), b->stride,
                MATRIX_ELEM_PTR(a), a->stride,
                sizeof(MATRIX_ELEM) * a->ncol, a->nrow,
                cudaMemcpyDeviceToDevice);
    return 0;
}

extern const char *MATRIX_CUMATRIX_HOST_TNAME;
static int nerv_matrix_(copy_fromh)(lua_State *L) { 
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    CHECK_SAME_DIMENSION(a, b);
    cudaMemcpy2D(MATRIX_ELEM_PTR(a), a->stride,
                MATRIX_ELEM_PTR(b), b->stride,
                sizeof(MATRIX_ELEM) * b->ncol, b->nrow,
                cudaMemcpyHostToDevice);
    return 0;
}

static int nerv_matrix_(copy_toh)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    CHECK_SAME_DIMENSION(a, b);
    cudaMemcpy2D(MATRIX_ELEM_PTR(b), b->stride,
                MATRIX_ELEM_PTR(a), a->stride,
                sizeof(MATRIX_ELEM) * a->ncol, a->nrow,
                cudaMemcpyDeviceToHost);
    return 0;
}

static int nerv_matrix_(trans)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(a->ncol, a->nrow);
    MATRIX_ELEM alpha = 1, beta = 0;
    NERV_CUBLAS_(geam)(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                a->nrow, a->ncol,
                &alpha,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM));
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}


static const luaL_Reg nerv_matrix_(extra_methods)[] = {
    {"create", nerv_matrix_(create)},
    {"softmax", nerv_matrix_(softmax)},
    {"colsum", nerv_matrix_(colsum)},
    {"rowsum", nerv_matrix_(rowsum)},
    {"rowmax", nerv_matrix_(rowmax)},
    {"copy_fromh", nerv_matrix_(copy_fromh)},
    {"copy_fromd", nerv_matrix_(copy_fromd)},
    {"copy_toh", nerv_matrix_(copy_toh)},
    {"copy_tod", nerv_matrix_(copy_tod)},
    {"trans", nerv_matrix_(trans)},
    /* in-place calc */
    {"add", nerv_matrix_(add)},
    {"mul", nerv_matrix_(mul)},
    {"add_row", nerv_matrix_(add_row)},
    {"fill", nerv_matrix_(fill)},
    {"sigmoid", nerv_matrix_(sigmoid)},
    {"sigmoid_grad", nerv_matrix_(sigmoid_grad)},
    {NULL, NULL}
};

static void cuda_matrix_(init)(lua_State *L) {
    luaN_append_methods(L, nerv_matrix_(extra_methods));
    cublasCreate(&cublas_handle);
}

static void cuda_matrix_(free)(MATRIX_ELEM *ptr) {
    cudaFree(ptr);
}

static void cuda_matrix_(alloc)(MATRIX_ELEM **dptr, size_t *stride,
                                long width, long height) {
    cudaMallocPitch((void **)dptr, stride, width, height);
}

static MATRIX_ELEM cuda_matrix_(read)(MATRIX_ELEM *data, int idx) {
    MATRIX_ELEM res;
    cudaMemcpy(&res, data + idx, sizeof(MATRIX_ELEM), cudaMemcpyDeviceToHost);
    return res;
}

static void cuda_matrix_(write)(MATRIX_ELEM *data, int idx, MATRIX_ELEM val) {
    cudaMemcpy(data + idx, &val, sizeof(MATRIX_ELEM), cudaMemcpyHostToDevice);
}

int nerv_matrix_(get_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

int nerv_matrix_(set_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

#include "matrix.c"
#endif
