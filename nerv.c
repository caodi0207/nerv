#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

extern void nerv_point_init(lua_State *L);
extern void nerv_matrix_init(lua_State *L);
extern void nerv_param_init(lua_State *L);

int luaopen_libnerv(lua_State *L) {
    lua_newtable(L);
    lua_setfield(L, LUA_GLOBALSINDEX, "nerv");
    nerv_point_init(L);
    nerv_matrix_init(L);
    nerv_param_init(L);
    return 1;
}
