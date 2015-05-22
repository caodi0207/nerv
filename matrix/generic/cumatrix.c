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
    if (!(a->nrow == b->nrow && a->ncol == b->ncol))
        nerv_error(L, "Matrices should be of the same dimension");
    nerv_matrix_(add_)(a, b, c, alpha, beta);
    luaT_pushudata(L, c, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(get_cublas_op)(char ch) {
    return (ch == 'T' || ch == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
}

static int nerv_matrix_(mul)(lua_State *L) {
    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    int nargs = lua_gettop(L);
    int ta = nargs > 3 ? nerv_matrix_(get_cublas_op)(*luaL_checkstring(L, 4)) \
                            : CUBLAS_OP_N;
    int tb = nargs > 4 ? nerv_matrix_(get_cublas_op)(*luaL_checkstring(L, 5)) \
                            : CUBLAS_OP_N;
    printf("%d %d\n", ta, tb);
    if (a->ncol != b->nrow)
        nerv_error(L, "Wrong dimension of multipliers");
    MATRIX_ELEM alpha = 1.0f, beta = 0.0f;
    NERV_CUBLAS_(gemm)(cublas_handle, tb, ta,
                b->ncol, a->nrow, b->nrow,
                &alpha,
                MATRIX_ELEM_PTR(b), b->stride / sizeof(MATRIX_ELEM),
                MATRIX_ELEM_PTR(a), a->stride / sizeof(MATRIX_ELEM),
                &beta,
                MATRIX_ELEM_PTR(c), c->stride / sizeof(MATRIX_ELEM));
    luaT_pushudata(L, c, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(create)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(a->nrow, a->ncol);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(sigmoid)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(a->nrow, a->ncol);
    cudak_(cuda_sigmoid)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(softmax)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *max = nerv_matrix_(new_)(a->nrow, 1);
    Matrix *dno = nerv_matrix_(new_)(a->nrow, 1);
    Matrix *b = nerv_matrix_(new_)(a->nrow, a->ncol);
    cudak_(cuda_colmax)(a, max);
    cudak_(cuda_softmax_denominator)(a, max, dno);
    cudak_(cuda_softmax_final)(a, max, dno, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(colsum)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(a->nrow, 1);
    cudak_(cuda_colsum)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(colmax)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(new_)(a->nrow, 1);
    cudak_(cuda_colmax)(a, b);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static const luaL_Reg nerv_matrix_(extra_methods)[] = {
    {"add", nerv_matrix_(add)},
    {"mul", nerv_matrix_(mul)},
    {"create", nerv_matrix_(create)},
    {"sigmoid", nerv_matrix_(sigmoid)},
    {"softmax", nerv_matrix_(softmax)},
    {"colsum", nerv_matrix_(colsum)},
    {"colmax", nerv_matrix_(colmax)},
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