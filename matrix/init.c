#include "../common.h"
#include "generic/matrix.h"

const char *nerv_matrix_tname = "nerv.Matrix";
const char *nerv_matrix_cuda_tname = "nerv.CuMatrix";
const char *nerv_matrix_host_tname = "nerv.MMatrix";

void nerv_matrix_host_float_init(lua_State *L);
void nerv_matrix_cuda_float_init(lua_State *L);
void nerv_matrix_host_double_init(lua_State *L);
void nerv_matrix_cuda_double_init(lua_State *L);
void nerv_matrix_host_int_init(lua_State *L);
int print_profile(lua_State *L);
int clear_profile(lua_State *L);

static const luaL_Reg matrix_methods[] = {
    {"__tostring__", nerv_error_method_not_implemented },
    {"__add__", nerv_error_method_not_implemented },
    {"__sub__", nerv_error_method_not_implemented },
    {"__mul__", nerv_error_method_not_implemented },
    {"print_profile", print_profile},
    {"clear_profile", clear_profile},
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
    nerv_matrix_cuda_float_init(L);
    nerv_matrix_cuda_double_init(L);

    /* MMatrix inherits from Matrix */
    luaT_newmetatable(L, nerv_matrix_host_tname, nerv_matrix_tname,
                            NULL, NULL, NULL);
    nerv_matrix_host_float_init(L);
    nerv_matrix_host_double_init(L);
    nerv_matrix_host_int_init(L);
}
