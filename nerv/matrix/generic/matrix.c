#ifdef NERV_GENERIC_MATRIX
#include "../../common.h"
#include "../../lib/matrix/generic/matrix.h"

extern const char *nerv_matrix_(tname);
extern const char *MATRIX_BASE_TNAME;


int nerv_matrix_(lua_new)(lua_State *L) {
    Status status;
    Matrix *self = nerv_matrix_(create)(luaL_checkinteger(L, 1),
                                        luaL_checkinteger(L, 2), &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, self, nerv_matrix_(tname));
    return 1;
}

int nerv_matrix_(lua_destroy)(lua_State *L) {
    Status status;
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    nerv_matrix_(destroy)(self, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 1;
}

int nerv_matrix_(lua_get_elem)(lua_State *L);
int nerv_matrix_(lua_set_elem)(lua_State *L);

static int nerv_matrix_(lua_newindex)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    if (lua_isnumber(L, 2))
    {
        int idx = luaL_checkinteger(L, 2);
        if (self->nrow == 1)
        {
            if (idx < 0 || idx >= self->ncol)
                nerv_error(L, "index must be within range [0, %d)", self->ncol);
            MATRIX_DATA_WRITE(L, MATRIX_ELEM_PTR(self), idx,
                                luaL_checknumber(L, 3));
        }
        else
            nerv_error(L, "cannot assign to row vector");
        lua_pushboolean(L, 1);
        return 1;
    }
    else
    {
        lua_pushboolean(L, 0);
        return 1;
    }
}


static int nerv_matrix_(lua_index)(lua_State *L) {
    Status status;
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    if (lua_isnumber(L, 2))
    {
        int idx = luaL_checkinteger(L, 2);
        if (self->nrow == 1)
        {
            if (idx < 0 || idx >= self->ncol)
                nerv_error(L, "index must be within range [0, %d)", self->ncol);
            lua_pushnumber(L, MATRIX_DATA_READ(L, MATRIX_ELEM_PTR(self), idx));
        }
        else
        {
            if (idx < 0 || idx >= self->nrow)
                nerv_error(L, "index must be within range [0, %d)", self->nrow);
            luaT_pushudata(L, nerv_matrix_(getrow)(self, idx),
                                nerv_matrix_(tname));
        }
        lua_pushboolean(L, 1);
        return 2;
    }
    else
    {
        lua_pushboolean(L, 0);
        return 1;
    }
}

static int nerv_matrix_(lua_ncol)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    lua_pushinteger(L, self->ncol);
    return 1;
}

static int nerv_matrix_(lua_nrow)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    lua_pushinteger(L, self->nrow);
    return 1;
}

static int nerv_matrix_(lua_get_dataref_value)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    lua_pushinteger(L, *(self->data_ref));
    return 1;
}

static const luaL_Reg nerv_matrix_(methods)[] = {
    {"get_elem", nerv_matrix_(lua_get_elem)},
    {"set_elem", nerv_matrix_(lua_set_elem)},
    {"ncol", nerv_matrix_(lua_ncol)},
    {"nrow", nerv_matrix_(lua_nrow)},
    {"get_dataref_value", nerv_matrix_(lua_get_dataref_value)},
    {"__index__", nerv_matrix_(lua_index)},
    {"__newindex__", nerv_matrix_(lua_newindex)},
    {NULL, NULL}
};

void nerv_matrix_(lua_init)(lua_State *L) {
    luaT_newmetatable(L, nerv_matrix_(tname), MATRIX_BASE_TNAME,
                        nerv_matrix_(lua_new), nerv_matrix_(lua_destroy), NULL);
    luaL_register(L, NULL, nerv_matrix_(methods));
#ifdef MATRIX_INIT
    MATRIX_INIT(L);
#endif
    lua_pop(L, 1);
}
#endif
