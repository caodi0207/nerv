#define NERV_GENERIC_CUMATRIX
#include "../common.h"
#include "cuda_helper.h"
static cublasHandle_t cublas_handle;
static cudaEvent_t profile_start, profile_stop;
static HashMap *profile;

int print_profile(lua_State *L) {
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

int clear_profile(lua_State *L) {
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
#undef MATRIX_CUMATRIX_HOST_TNAME

#define MATRIX_USE_DOUBLE
#define cuda_matrix_(NAME) cuda_matrix_double_##NAME
#define nerv_matrix_(NAME) nerv_matrix_cuda_double_##NAME
#define cudak_(NAME) cudak_double_ ## NAME
#define NERV_CUBLAS_(NAME) cublasD##NAME
#define MATRIX_CUMATRIX_HOST_TNAME nerv_matrix_host_double_tname
const char *nerv_matrix_(tname) = "nerv.CuMatrixDouble";
#include "generic/cumatrix.c"
