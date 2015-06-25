#define NERV_GENERIC_CUMATRIX
#include "../lib/common.h"
#include "../lib/matrix/cumatrix.h"
#include "../lib/matrix/cuda_helper.h"
#include <string.h>
#define PROFILE_HASHMAP_SIZE 123457
static cublasHandle_t cublas_handle;
static cudaEvent_t profile_start, profile_stop;
static HashMap *profile;

static int print_profile(lua_State *L) {
    nerv_cumatrix_print_profile();
    return 0;
}

static int clear_profile(lua_State *L) {
    nerv_cumatrix_clear_profile();
    return 0;
}

static const luaL_Reg cumatrix_methods[] = {
    {"print_profile", print_profile},
    {"clear_profile", clear_profile},
    {NULL, NULL}
};

extern void nerv_matrix_cuda_float_lua_init(lua_State *L);
extern void nerv_matrix_cuda_double_lua_init(lua_State *L);

void nerv_lua_cumatrix_init(lua_State *L) {
    luaL_register(L, NULL, cumatrix_methods);
    nerv_cumatrix_init();
    nerv_matrix_cuda_float_lua_init(L);
    nerv_matrix_cuda_double_lua_init(L);
}

#define MATRIX_USE_FLOAT
#define cuda_matrix_(NAME) cuda_matrix_float_##NAME
#define nerv_matrix_(NAME) nerv_matrix_cuda_float_##NAME
#define cudak_(NAME) cudak_float_ ## NAME
#define NERV_CUBLAS_(NAME) cublasS##NAME
#define MATRIX_CUMATRIX_HOST_TNAME nerv_matrix_host_float_tname
const char *nerv_matrix_(tname) = "nerv.CuMatrixFloat";
#include "generic/cumatrix.c"
#undef NERV_CUBLAS_
#undef cudak_
#undef nerv_matrix_
#undef cuda_matrix_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT
#undef MATRIX_ELEM_WRITE_FMT
#undef MATRIX_CUMATRIX_HOST_TNAME

#define MATRIX_USE_DOUBLE
#define cuda_matrix_(NAME) cuda_matrix_double_##NAME
#define nerv_matrix_(NAME) nerv_matrix_cuda_double_##NAME
#define cudak_(NAME) cudak_double_ ## NAME
#define NERV_CUBLAS_(NAME) cublasD##NAME
#define MATRIX_CUMATRIX_HOST_TNAME nerv_matrix_host_double_tname
const char *nerv_matrix_(tname) = "nerv.CuMatrixDouble";
#include "generic/cumatrix.c"
