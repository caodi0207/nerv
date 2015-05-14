#define MATRIX_DATA_FREE(ptr) free(ptr)
#define MATRIX_DATA_ALLOC(size) malloc(size)
#define MATRIX_DATA_STRIDE(ncol) (sizeof(float) * (ncol))
#define MATRIX_GENERIC
#define nerv_float_matrix_(NAME) nerv_float_matrix_cuda_ ## NAME
#include "generic/matrix.c"

const char *nerv_float_matrix_(tname) = "nerv.FloatCuMatrix";
int nerv_float_matrix_(get_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

int nerv_float_matrix_(set_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}
