#define NERV_GENERIC_MMATRIX
#include <stdlib.h>
#include "../common.h"
void nerv_matrix_host_float_lua_init(lua_State *L);
void nerv_matrix_host_double_lua_init(lua_State *L);
void nerv_matrix_host_int_lua_init(lua_State *L);

void nerv_lua_mmatrix_init(lua_State *L) {
    srand(1);
    nerv_matrix_host_float_lua_init(L);
    nerv_matrix_host_double_lua_init(L);
    nerv_matrix_host_int_lua_init(L);
}

#define MATRIX_USE_FLOAT
#define host_matrix_(NAME) host_matrix_float_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_float_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixFloat";
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT
#undef MATRIX_ELEM_WRITE_FMT

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_DOUBLE
#define host_matrix_(NAME) host_matrix_double_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_double_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixDouble";
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_DOUBLE
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT
#undef MATRIX_ELEM_WRITE_FMT

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_INT
#define host_matrix_(NAME) host_matrix_int_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_int_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixInt";
#define MMATRIX_INIT(L) host_matrix_(init_extra)(L)

static const luaL_Reg nerv_matrix_(extra_methods_int)[];
static void host_matrix_(init_extra)(lua_State *L) {
    luaN_append_methods(L, nerv_matrix_(extra_methods_int));
}

#include "generic/mmatrix.c"
#include "../lib/matrix/mmatrix.h"

static int nerv_matrix_(lua_perm_gen)(lua_State *L) {
    Status status;
    int i, ncol = luaL_checkinteger(L, 1);
    Matrix *self = nerv_matrix_(perm_gen)(ncol, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, self, nerv_matrix_(tname));
    return 1;
}

static const luaL_Reg nerv_matrix_(extra_methods_int)[] = {
    {"perm_gen", nerv_matrix_(lua_perm_gen)},
    {NULL, NULL}
};
