#include "../common.h"

const char *nerv_matrix_tname = "nerv.Matrix";
const char *nerv_matrix_cuda_tname = "nerv.CuMatrix";
const char *nerv_matrix_host_tname = "nerv.MMatrix";

void nerv_lua_cumatrix_init(lua_State *L);
void nerv_lua_mmatrix_init(lua_State *L);

static const luaL_Reg matrix_methods[] = {
    {"__tostring__", nerv_error_method_not_implemented },
    {"__add__", nerv_error_method_not_implemented },
    {"__sub__", nerv_error_method_not_implemented },
    {"__mul__", nerv_error_method_not_implemented },
    {NULL, NULL}
};

void nerv_matrix_init(lua_State *L) {
    /* abstract base class: Matrix */
    luaT_newmetatable(L, nerv_matrix_tname, NULL, NULL, NULL, NULL);
    luaL_register(L, NULL, matrix_methods);
    lua_pop(L, 1);

    /* CuMatrix inherits from Matrix */
    luaT_newmetatable(L, nerv_matrix_cuda_tname, nerv_matrix_tname,
                            NULL, NULL, NULL);
    nerv_lua_cumatrix_init(L);
    lua_pop(L, 1);
    /* MMatrix inherits from Matrix */
    luaT_newmetatable(L, nerv_matrix_host_tname, nerv_matrix_tname,
                            NULL, NULL, NULL);
    nerv_lua_mmatrix_init(L);
    lua_pop(L, 1);
}
