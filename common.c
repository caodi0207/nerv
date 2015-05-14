#ifndef NERV_COMMON_H
#define NERV_COMMON_H
#include "common.h"
#include <stdarg.h>
int nerv_error(lua_State *L, const char *err_mesg_fmt, ...) {
    va_list ap;
    va_start(ap, err_mesg_fmt);
    lua_pushstring(L, "Nerv internal error: ");
    lua_pushvfstring(L, err_mesg_fmt, ap); 
    lua_concat(L, 2);
    lua_error(L);
    va_end(ap);
    return 0;
}

int nerv_error_method_not_implemented(lua_State *L) {
    return nerv_error(L, "method not implemented"); 
}
#endif
