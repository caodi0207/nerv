#define NERV_GENERIC_CUMATRIX
#include "../common.h"
#include "cuda_helper.h"
#include <string.h>
#define PROFILE_HASHMAP_SIZE 123457
static cublasHandle_t cublas_handle;
static cudaEvent_t profile_start, profile_stop;
static HashMap *profile;

static int print_profile(lua_State *L) {
    (void)L;
    size_t i;
    fprintf(stderr, "*** [nerv cumatrix profile] **\n");
    for (i = 0; i < profile->size; i++)
    {
        HashNode *ptr;
        for (ptr = profile->bucket[i]; ptr; ptr = ptr->next)
        {
            fprintf(stderr, "%s:\t%.6f\n", ptr->key, *(float *)ptr->val);
        }
    }
    return 0;
}

static int clear_profile(lua_State *L) {
    (void)L;
    hashmap_clear(profile);
    return 0;
}

void accu_profile(const char *name, float delta) {
    float *val = hashmap_getval(profile, name);
    if (!val)
    {
        val = malloc(sizeof(float));
        *val = 0;
        hashmap_setval(profile, name, val);
    }
    *val += delta;
}

static const luaL_Reg cumatrix_methods[] = {
    {"print_profile", print_profile},
    {"clear_profile", clear_profile},
    {NULL, NULL}
};

extern void nerv_matrix_cuda_float_init(lua_State *L);
extern void nerv_matrix_cuda_double_init(lua_State *L);

void nerv_cumatrix_init(lua_State *L) {
    luaL_register(L, NULL, cumatrix_methods);
    cublasCreate(&cublas_handle);
    cudaEventCreate(&profile_start);
    cudaEventCreate(&profile_stop);
    profile = hashmap_create(PROFILE_HASHMAP_SIZE, bkdr_hash, strcmp);
    nerv_matrix_cuda_float_init(L);
    nerv_matrix_cuda_double_init(L);
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
