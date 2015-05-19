#include "../common.h"
#include "generic/matrix.h"

const char *nerv_matrix_tname = "nerv.Matrix";
void nerv_matrix_float_host_init(lua_State *L);
void nerv_matrix_float_cuda_init(lua_State *L);
void nerv_matrix_double_host_init(lua_State *L);
void nerv_matrix_double_cuda_init(lua_State *L);

static const luaL_Reg matrix_methods[] = {
    {"__tostring__", nerv_error_method_not_implemented },
    {"__add__", nerv_error_method_not_implemented },
    {"__sub__", nerv_error_method_not_implemented },
    {"__mul__", nerv_error_method_not_implemented },
    {NULL, NULL}
};

void nerv_matrix_init(lua_State *L) {
    /* abstract class */
    luaT_newmetatable(L, nerv_matrix_tname, NULL, NULL, NULL, NULL);
    luaL_register(L, NULL, matrix_methods);
    lua_pop(L, 1);
    nerv_matrix_float_host_init(L);
    nerv_matrix_float_cuda_init(L);
/*    nerv_matrix_double_host_init(L); */
    nerv_matrix_double_cuda_init(L);
}
