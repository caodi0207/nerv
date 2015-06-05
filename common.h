#ifndef NERV_COMMON_H
#define NERV_COMMON_H
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "luaT/luaT.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct HashNode {
    const char *key;
    void *val;
    struct HashNode *next;
} HashNode;

typedef int (*HashMapCmp_t)(const char *a, const char *b);
typedef size_t (*HashKey_t)(const char *key);

typedef struct HashMap {
    HashNode **bucket;
    HashMapCmp_t cmp;
    HashKey_t hfunc;
    size_t size;
} HashMap;

HashMap *hashmap_create(size_t size, HashKey_t hfunc, HashMapCmp_t cmp);
void *hashmap_getval(HashMap *h, const char *key);
void hashmap_setval(HashMap *h, const char *key, void *val);

size_t bkdr_hash(const char *key);

int nerv_error(lua_State *L, const char *err_mesg_fmt, ...); 
int nerv_error_method_not_implemented(lua_State *L);
void luaN_append_methods(lua_State *L, const luaL_Reg *mlist);
#endif
