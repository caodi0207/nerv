#include "common.h"

extern void nerv_example_init(lua_State *L);
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
    /* duplicate table */
    lua_pushvalue(L, -1);
    /* set table to global index */
    lua_setfield(L, LUA_GLOBALSINDEX, "nerv");
    /* A table reference still remains.
     *
     * The following initialization functions should obey to the rule that they
     * maintain the stack properly to guarantee the stack stays the same before
     * and after invoking the call (i.e. stay balanced).
     *
     * Also note that they can make use of the value at top of the stack which
     * references to the `nerv` global table. */
    nerv_utils_init(L);
    nerv_example_init(L);
    nerv_matrix_init(L);
    nerv_param_init(L);
    return 1;
}
