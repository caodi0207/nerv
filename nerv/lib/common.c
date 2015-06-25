#include "common.h"
#include <stdarg.h>
int nerv_error(lua_State *L, const char *err_mesg_fmt, ...) {
    va_list ap;
    va_start(ap, err_mesg_fmt);
    lua_pushstring(L, "[nerv] internal error: ");
    lua_pushvfstring(L, err_mesg_fmt, ap); 
    lua_concat(L, 2);
    lua_error(L);
    va_end(ap);
    return 0;
}

int nerv_error_status(lua_State *L, Status *status) {
    const char *mmesg = NULL;
    switch (status->err_code)
    {
        case MAT_GENERAL_ERR: mmesg = "general error"; break;
        case MAT_INSUF_MEM: mmesg = "insufficient memory"; break;
        case MAT_INVALID_FORMAT: mmesg = "invalid matrix format"; break;
        case MAT_WRITE_ERROR: mmesg = "error while writing matrix"; break;
        case MAT_INVALID_COPY_INTERVAL: mmesg = "invalid copy interval"; break;
        case MAT_MISMATCH_DIM: mmesg = "mismatching matrix dimension"; break;
        case MAT_WRONG_MULT_DIM: mmesg = "wrong multipier dimension"; break;
        case MAT_ROW_VECTOR_EXP: mmesg = "row vector expected"; break;
        case MAT_COL_VECTOR_EXP: mmesg = "column vector expected"; break;
        case MAT_IDX_VECTOR_EXP: mmesg = "index vector expected"; break;
        case MAT_INVALID_IDX: mmesg = "invalid index"; break;
        case MAT_CUDA_ERR: mmesg = "cuda error"; break;
        case MAT_CUBLAS_ERR: mmesg = "cublas error"; break;
        case CF_INVALID_FORMAT: mmesg = "invalid format"; break;
        case CF_END_OF_FILE: mmesg = "unexpected end of file"; break;
        case CF_SECTION_OVERFLOW: mmesg = "section overflow"; break;
        case CF_WRITE_ERROR: mmesg = "error while writing"; break;
        case CF_ERR_OPEN_FILE: mmesg = "error while opening file"; break;
        case CF_INVALID_OP: mmesg = "invalid operation"; break;
        default: mmesg = "unknown"; break;
    }
    if (status->msg)
        nerv_error(L, "%s: %s @%s:%d", mmesg, status->msg,
                                        status->file, status->lineno);
    else
        nerv_error(L, "%s @%s:%d", mmesg, status->file, status->lineno);
}

int nerv_error_method_not_implemented(lua_State *L) {
    return nerv_error(L, "method not implemented"); 
}

void luaN_append_methods(lua_State *L, const luaL_Reg *mlist) {
    for (; mlist->func; mlist++)
    {
        lua_pushcfunction(L, mlist->func);
        lua_setfield(L, -2, mlist->name);
    }
}

HashMap *hashmap_create(size_t size, HashKey_t hfunc, HashMapCmp_t cmp) {
    HashMap *res = (HashMap *)malloc(sizeof(HashMap));
    res->bucket = calloc(size, sizeof(HashNode));
    res->cmp = cmp;
    res->hfunc = hfunc;
    res->size = size;
    return res;
}

void *hashmap_getval(HashMap *h, const char *key) {
    size_t idx = h->hfunc(key) % h->size;
    HashNode *ptr;
    for (ptr = h->bucket[idx]; ptr; ptr = ptr->next)
    {
        if (!h->cmp(ptr->key, key))
            return ptr->val;
    }
    return NULL;
}

void hashmap_setval(HashMap *h, const char *key, void *val) {
    size_t idx = h->hfunc(key) % h->size;
    HashNode *ptr = malloc(sizeof(HashNode));
    ptr->next = h->bucket[idx];
    h->bucket[idx] = ptr;
    ptr->key = key;
    ptr->val = val;
}

void hashmap_clear(HashMap *h) {
    size_t i;
    for (i = 0; i < h->size; i++)
    {
        HashNode *ptr, *nptr;
        for (ptr = h->bucket[i]; ptr; ptr = nptr)
        {
            nptr = ptr->next;
            free(ptr->val);
            free(ptr);
        }
        h->bucket[i] = NULL;
    }
}

size_t bkdr_hash(const char *key) {
    unsigned int seed = 131;
    unsigned int res = 0;
    while (*key)
        res = res * seed + *key++;
    return res;
}
