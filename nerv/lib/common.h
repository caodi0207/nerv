#ifndef NERV_COMMON_H
#define NERV_COMMON_H
#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "luaT/luaT.h"
#include <stdio.h>
#include <stdlib.h>

typedef enum ErrCode {
    NERV_NORMAL,
    /* matrix err */
    MAT_GENERAL_ERR,
    MAT_INSUF_MEM,
    MAT_INVALID_FORMAT,
    MAT_WRITE_ERROR,
    MAT_INVALID_COPY_INTERVAL,
    MAT_MISMATCH_DIM,
    MAT_WRONG_MULT_DIM,
    MAT_ROW_VECTOR_EXP,
    MAT_COL_VECTOR_EXP,
    MAT_IDX_VECTOR_EXP,
    MAT_INVALID_IDX,
    MAT_CUDA_ERR,
    MAT_CUBLAS_ERR,
    /* chunk file err */
    CF_INVALID_FORMAT,
    CF_END_OF_FILE,
    CF_SECTION_OVERFLOW,
    CF_WRITE_ERROR,
    CF_ERR_OPEN_FILE,
    CF_INVALID_OP,
} ErrCode;

typedef struct Status {
    ErrCode err_code;
    const char *file;
    int lineno;
    const char *msg;
} Status;

#define NERV_SET_STATUS(status, code, m) \
    do { \
        (status)->err_code = (code); \
        (status)->msg = (m); \
        (status)->file = __FILE__; \
        (status)->lineno = __LINE__; \
    } while (0)

#define NERV_EXIT_STATUS(status, code, msg) \
    do { \
        NERV_SET_STATUS(status, code, msg); \
        return; \
    } while (0)

#define NERV_LUA_CHECK_STATUS(L, status) \
    do { \
        if (status.err_code != NERV_NORMAL) \
            nerv_error_status(L, &status); \
    } while (0)

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
void hashmap_clear(HashMap *h);

size_t bkdr_hash(const char *key);

int nerv_error(lua_State *L, const char *err_mesg_fmt, ...); 
int nerv_error_status(lua_State *L, Status *status); 
int nerv_error_method_not_implemented(lua_State *L);
void luaN_append_methods(lua_State *L, const luaL_Reg *mlist);
#endif
