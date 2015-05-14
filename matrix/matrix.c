#define MATRIX_DATA_FREE(ptr) free(ptr)
#define MATRIX_DATA_ALLOC(size) malloc(size)
#define MATRIX_DATA_STRIDE(ncol) (sizeof(float) * (ncol))
#define MATRIX_GENERIC
#define nerv_float_matrix_(NAME) nerv_float_matrix_host_ ## NAME
#include "generic/matrix.c"

const char *nerv_float_matrix_(tname) = "nerv.FloatMatrix";
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
    long upper = self->nrow * self->ncol;
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    self->data.f[idx] = v;
    return 0;
}
