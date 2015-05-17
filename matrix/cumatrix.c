#define MATRIX_DATA_FREE(ptr) cuda_float_array_free(ptr)
#define MATRIX_DATA_ALLOC(dptr, stride, width, height) cuda_float_array_alloc(dptr, stride, width, height)
#define MATRIX_DATA_WRITE(data, idx, val) cuda_float_array_write(data, idx, val)
#define MATRIX_DATA_READ(data, idx) cuda_float_array_read(data, idx)
#define MATRIX_INIT(L) cuda_float_init(L)
#define NERV_GENERIC_MATRIX
#define nerv_float_matrix_(NAME) nerv_float_matrix_cuda_ ## NAME
#include "../common.h"
#include "generic/matrix.h"
#include "cuda.h"
#include "driver_types.h"
#include "cublas_v2.h"

const char *nerv_float_matrix_(tname) = "nerv.FloatCuMatrix";
static cublasHandle_t cublas_handle;

Matrix *nerv_float_matrix_(new_)(long nrow, long ncol);
static int nerv_float_matrix_(add)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_float_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_float_matrix_(tname));
    Matrix *c;
    long nrow, ncol;
    if (!(a->nrow == b->nrow && a->ncol == b->ncol))
        nerv_error(L, "Matrices should be of the same dimension");
    nrow = a->nrow;
    ncol = a->ncol;
    c = nerv_float_matrix_(new_)(nrow, ncol);
    float alpha = 1.0f, beta = 1.0f;
    cublasSgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                ncol, nrow,
                &alpha,
                a->data.f, a->stride / sizeof(float),
                &beta,
                b->data.f, b->stride / sizeof(float),
                c->data.f, c->stride / sizeof(float));
    luaT_pushudata(L, c, nerv_float_matrix_(tname));
    return 1;
}

static int nerv_float_matrix_(mul)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_float_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_float_matrix_(tname));
    Matrix *c;
    if (a->ncol != b->nrow)
        nerv_error(L, "Wrong dimension of multipliers");
    c = nerv_float_matrix_(new_)(a->nrow, b->ncol);
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                b->ncol, a->nrow, b->nrow,
                &alpha,
                b->data.f, b->stride / sizeof(float),
                a->data.f, a->stride / sizeof(float),
                &beta,
                c->data.f, c->stride / sizeof(float));
    luaT_pushudata(L, c, nerv_float_matrix_(tname));
    return 1;
}

static const luaL_Reg nerv_float_matrix_(extra_methods)[] = {
    {"__add__", nerv_float_matrix_(add)},
    {"__mul__", nerv_float_matrix_(mul)},
    {NULL, NULL}
};

static void cuda_float_init(lua_State *L) {
    luaN_append_methods(L, nerv_float_matrix_(extra_methods));
    cublasCreate(&cublas_handle);
}

static cuda_float_array_free(float *ptr) {
    cudaFree(ptr);
}

static cuda_float_array_alloc(float **dptr, long *stride,
                                long width, long height) {
    cudaMallocPitch(dptr, stride, width, height);
}

static float cuda_float_array_read(float *data, int idx) {
    float res;
    cudaMemcpy(&res, data + idx, sizeof(float), cudaMemcpyDeviceToHost);
    return res;
}

static void cuda_float_array_write(float *data, int idx, float val) {
    cudaMemcpy(data + idx, &val, sizeof(float), cudaMemcpyHostToDevice);
}

int nerv_float_matrix_(get_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

int nerv_float_matrix_(set_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

#include "generic/matrix.c"
