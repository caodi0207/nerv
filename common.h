#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "luaT/luaT.h"
#include <stdio.h>
#include <stdlib.h>

int nerv_error(lua_State *L, const char *err_mesg_fmt, ...); 
int nerv_error_method_not_implemented(lua_State *L);
