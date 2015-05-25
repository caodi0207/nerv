#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common.h"

const char *point_tname = "nerv.Point";

typedef struct {
    double x, y;
    int arr[100];
} Point;

static int point_get_sinx (lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    lua_pushnumber(L, sin(p->x));
    return 1;
}

static int point_set_x (lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    p->x = luaL_checknumber(L, 2);
    return 0;
}


static int point_get_y(lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    lua_pushnumber(L, sin(p->x));
    return 1;
}

static int point_newindex(lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    if (lua_isnumber(L, 2))
    {
        int d = luaL_checkinteger(L, 2);
        double v = luaL_checknumber(L, 3);
        if (0 <= d && d < 100)
            p->arr[d] = v;
        lua_pushboolean(L, 1);
        return 1;
    }
    else
    {
        lua_pushboolean(L, 0);
        return 1;
    }
}

static int point_index(lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    if (lua_isnumber(L, 2))
    {
        int d = luaL_checkinteger(L, 2);
        if (0 <= d && d < 100)
            lua_pushnumber(L, p->arr[d]);
        lua_pushboolean(L, 1);
        return 2;
    }
    else
    {
        lua_pushboolean(L, 0);
        return 1;
    }
}

int point_new(lua_State *L) {
    Point *self = (Point *)malloc(sizeof(Point));
    self->x = 0;
    self->y = 0;
    luaT_pushudata(L, self, point_tname);
    return 1;
}

static const luaL_Reg point[] = {
    {"get_sinx", point_get_sinx},
    {"set_x", point_set_x},
    {"get_y", point_get_y},
    {"__index__", point_index},
    {"__newindex__", point_newindex},
    {NULL, NULL}
};

void nerv_point_init(lua_State *L) {
    luaT_newmetatable(L, "nerv.Point", NULL, point_new, NULL, NULL);
    luaL_register(L, NULL, point);
    lua_pop(L, 1);
}
