#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common.h"

#define SQR(x) ((x) * (x))

const char *point_tname = "nerv.Point";
const char *better_point_tname = "nerv.BetterPoint";

typedef struct {
    double x, y;
} Point;

static int point_norm (lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    lua_pushnumber(L, sqrt(SQR(p->x) + SQR(p->y)));
    return 1;
}

static int point_set_x (lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    p->x = luaL_checknumber(L, 2);
    return 0;
}

static int point_set_y (lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    p->y = luaL_checknumber(L, 2);
    return 0;
}

/* generic constructor */
void point_new_(Point *self, double x, double y) {
    self->x = x;
    self->y = y;
}

int point_new(lua_State *L) {
    /* `_new` function should create the object itself */
    Point *self = (Point *)malloc(sizeof(Point));
    point_new_(self, luaL_checknumber(L, 1), luaL_checknumber(L, 2));
    luaT_pushudata(L, self, point_tname);
    fprintf(stderr, "[example] %s constructor is invoked\n",
            point_tname);
    return 1;
}

int point___init(lua_State *L) {
    /* The difference between this function and `_new` function is that this
     * one is called by subclass of Point implemented in Lua, although it
     * basically does the same thing as `_new`. Also, it can read the empty
     * object (table) from the stack. (In this example, the table is ignored.) */
    Point *self = (Point *)malloc(sizeof(Point));
    point_new_(self, luaL_checknumber(L, 2), luaL_checknumber(L, 3));
    luaT_pushudata(L, self, point_tname);
    fprintf(stderr, "[example] A subclass has invoked `__init`\n");
    return 1;
}

static const luaL_Reg point_methods[] = {
    {"set_x", point_set_x},
    {"set_y", point_set_y},
    {"__init", point___init},
    {"norm", point_norm},
    {NULL, NULL}
};


/* the subclass method overrides the one from baseclass */
static int better_point_norm (lua_State *L) {
    Point *p = luaT_checkudata(L, 1, point_tname);
    lua_pushnumber(L, fabs(p->x) + fabs(p->y));
    return 1;
}

int better_point_new(lua_State *L) {
    /* `_new` function should create the object itself */
    Point *self = (Point *)malloc(sizeof(Point));
    point_new_(self, luaL_checknumber(L, 1), luaL_checknumber(L, 2));
    luaT_pushudata(L, self, better_point_tname);
    fprintf(stderr, "[example] %s constructor is invoked\n",
            better_point_tname);
    return 1;
}

int better_point___init(lua_State *L) {
    /* The difference between this function and `_new` function is that this
     * one is called by subclass of Point implemented in Lua, although it
     * basically does the same thing as `_new`. Also, it can read the empty
     * object (table) from the stack. (In this example, the table is ignored.) */
    Point *self = (Point *)malloc(sizeof(Point));
    point_new_(self, luaL_checknumber(L, 2), luaL_checknumber(L, 3));
    luaT_pushudata(L, self, better_point_tname);
    fprintf(stderr, "[example] A subclass has invoked `__init`\n");
    return 1;
}

static const luaL_Reg better_point_methods[] = {
    {"norm", better_point_norm},
    {"__init", better_point___init},
    {NULL, NULL}
};

void nerv_point_init(lua_State *L) {
    /* create a class and let luaT know */
    luaT_newmetatable(L, point_tname, NULL, point_new, NULL, NULL);
    /* register member functions */
    luaL_register(L, NULL, point_methods);
    /* keep the stack balanced, see `nerv.c` */
    lua_pop(L, 1);
}

void nerv_better_point_init(lua_State *L) {
    /* create a class and let luaT know */
    luaT_newmetatable(L, better_point_tname, point_tname,
                        better_point_new, NULL, NULL);
    /* register member functions */
    luaL_register(L, NULL, better_point_methods);
    /* keep the stack balanced, see `nerv.c` */
    lua_pop(L, 1);
}

void nerv_example_init(lua_State *L) {
    nerv_point_init(L);
    nerv_better_point_init(L);
}
