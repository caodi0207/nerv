#include "common.h"

extern void nerv_point_init(lua_State *L);
extern void nerv_matrix_init(lua_State *L);
extern void nerv_param_init(lua_State *L);

static const luaL_Reg nerv_utils_methods[] = {
    {"setmetatable", luaT_lua_setmetatable},
    {"getmetatable", luaT_lua_getmetatable},
    {"newmetatable", luaT_lua_newmetatable},
    {NULL, NULL}
};

void nerv_utils_init(lua_State *L) {
    luaL_register(L, NULL, nerv_utils_methods);
}

int luaopen_libnerv(lua_State *L) {
    lua_newtable(L);
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_GLOBALSINDEX, "nerv");
    nerv_utils_init(L);
    nerv_point_init(L);
    nerv_matrix_init(L);
    nerv_param_init(L);
    return 1;
}
