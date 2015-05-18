#define MATRIX_DATA_FREE(ptr) free(ptr)
#define MATRIX_DATA_ALLOC(dptr, stride, width, height) host_float_array_alloc(dptr, stride, width, height)
#define MATRIX_DATA_STRIDE(ncol) (sizeof(float) * (ncol))
#define MATRIX_DATA_WRITE(data, idx, val) (data[idx] = val)
#define MATRIX_DATA_READ(data, idx) (data[idx])
#define NERV_GENERIC_MATRIX
#define nerv_float_matrix_(NAME) nerv_float_matrix_host_ ## NAME
#include "../common.h"
#include "generic/matrix.h"

const char *nerv_float_matrix_(tname) = "nerv.FloatMatrix";

static void host_float_array_alloc(float **dptr, size_t *stride,
                                    long width, long height) {
    *dptr = (float *)malloc(width * height);
    *stride = width;
}

int nerv_float_matrix_(get_elem)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_float_matrix_(tname));
    int idx = luaL_checkinteger(L, 2);
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    lua_pushnumber(L, self->data.f[idx]);
    return 1;
}

int nerv_float_matrix_(set_elem)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_float_matrix_(tname));
    int idx = luaL_checkinteger(L, 2);
    float v = luaL_checknumber(L, 3);
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    self->data.f[idx] = v;
    return 0;
}

#include "generic/matrix.c"
