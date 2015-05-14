#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

extern void nerv_point_init(lua_State *L);
extern void nerv_matrix_init(lua_State *L);

LUALIB_API int luaopen_libnerv(lua_State *L) {
    lua_newtable(L);
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_GLOBALSINDEX, "nerv");
    nerv_point_init(L);
    nerv_matrix_init(L);
    return 1;
}
