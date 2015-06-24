#ifdef NERV_GENERIC_MMATRIX
#include "../../lib/matrix/generic/matrix.h"
#include "elem_type.h"
#define MATRIX_DATA_WRITE(L, data, idx, val) (data[idx] = val)
#define MATRIX_DATA_READ(L, data, idx) (data[idx])
#define MATRIX_INIT(L) host_matrix_(init)(L)
#define MATRIX_BASE_TNAME nerv_matrix_host_tname
#define NERV_GENERIC_MATRIX
#include "../../common.h"
#include "../../io/chunk_file.h"
#include "../../lib/matrix/generic/mmatrix.h"
#include "string.h"

int nerv_matrix_(lua_get_elem)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    int idx = luaL_checkinteger(L, 2);
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    lua_pushnumber(L, MATRIX_ELEM_PTR(self)[idx]);
    return 1;
}

int nerv_matrix_(lua_set_elem)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    int idx = luaL_checkinteger(L, 2);
    MATRIX_ELEM v = luaL_checknumber(L, 3);
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    MATRIX_ELEM_PTR(self)[idx] = v;
    return 0;
}

static const luaL_Reg nerv_matrix_(extra_methods)[];
static void host_matrix_(init)(lua_State *L) {
    luaN_append_methods(L, nerv_matrix_(extra_methods));
#ifdef MMATRIX_INIT
    MMATRIX_INIT(L);
#endif
}

#include "matrix.c"

int nerv_matrix_(lua_load)(lua_State *L) {
    Status status;
    ChunkData *cdp = luaT_checkudata(L, 1, nerv_chunk_data_tname);
    Matrix *self = nerv_matrix_(load)(cdp, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, self, nerv_matrix_(tname));
    return 1;
}

int nerv_matrix_(lua_save)(lua_State *L) {
    Status status;
    ChunkFile *cfp = luaT_checkudata(L, 2,
                            nerv_chunk_file_handle_tname);
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    nerv_matrix_(save)(self, cfp, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

int nerv_matrix_(lua_copy_from)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    int nargs = lua_gettop(L);
    int b_begin = nargs > 2 ? luaL_checkinteger(L, 3) : 0;
    int b_end = nargs > 3 ? luaL_checkinteger(L, 4) : b->nrow;
    int a_begin = nargs > 4 ? luaL_checkinteger(L, 5) : 0;
    nerv_matrix_(copy_from)(a, b, a_begin, b_begin, b_end, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static const luaL_Reg nerv_matrix_(extra_methods)[] = {
    {"load", nerv_matrix_(lua_load)},
    {"save", nerv_matrix_(lua_save)},
    {"copy_from", nerv_matrix_(lua_copy_from)},
    {NULL, NULL}
};

#endif
