#ifdef NERV_GENERIC_MATRIX
#include "../../common.h"
#include "matrix.h"

extern const char *nerv_matrix_(tname);
extern const char *MATRIX_BASE_TNAME;

void nerv_matrix_(data_free)(lua_State *L, Matrix *self) {
    (void)L;
    assert(*self->data_ref > 0);
    if (--(*self->data_ref) == 0)
    {
        /* free matrix data */
        MATRIX_DATA_FREE(L, MATRIX_ELEM_PTR(self));
        free(self->data_ref);
        free(self);
    }
}

void nerv_matrix_(data_retain)(Matrix *self) {
    (*self->data_ref)++;
}

Matrix *nerv_matrix_(new_)(lua_State *L, long nrow, long ncol) {
    Matrix *self = (Matrix *)malloc(sizeof(Matrix));
    self->nrow = nrow;
    self->ncol = ncol;
    self->nmax = self->nrow * self->ncol;
    MATRIX_DATA_ALLOC(L, &MATRIX_ELEM_PTR(self), &self->stride,
                        sizeof(MATRIX_ELEM) * self->ncol, self->nrow);
    self->data_ref = (long *)malloc(sizeof(long));
    *self->data_ref = 0;
    nerv_matrix_(data_retain)(self);
    return self;
}

int nerv_matrix_(new)(lua_State *L) {
    luaT_pushudata(L, nerv_matrix_(new_)(L, luaL_checkinteger(L, 1),
                                        luaL_checkinteger(L, 2)),
                    nerv_matrix_(tname));
    return 1;
}

int nerv_matrix_(destroy)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    nerv_matrix_(data_free)(L, self);
    return 1;
}

int nerv_matrix_(get_elem)(lua_State *L); 
int nerv_matrix_(set_elem)(lua_State *L);

static Matrix *nerv_matrix_(getrow)(Matrix *self, int row) {
    Matrix *prow = (Matrix *)malloc(sizeof(Matrix));
    prow->ncol = self->ncol;
    prow->nrow = 1;
    prow->stride = self->stride;
    prow->nmax = prow->ncol;
    MATRIX_ELEM_PTR(prow) = MATRIX_ROW_PTR(self, row);
    prow->data_ref = self->data_ref;
    nerv_matrix_(data_retain)(prow);
    return prow;
}

static int nerv_matrix_(newindex)(lua_State *L) {
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


static int nerv_matrix_(index)(lua_State *L) {
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
            luaT_pushudata(L, nerv_matrix_(getrow)(self, idx), nerv_matrix_(tname));
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

static int nerv_matrix_(ncol)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    lua_pushinteger(L, self->ncol);
    return 1;
}

static int nerv_matrix_(nrow)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    lua_pushinteger(L, self->nrow);
    return 1;
}

static int nerv_matrix_(get_dataref_value)(lua_State *L) {                                                                                                                                               
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));                                                                                                                                           
    lua_pushinteger(L, *(self->data_ref));                                                                                                                                                               
    return 1;                                                                                                                                                                                            
}      

static const luaL_Reg nerv_matrix_(methods)[] = {
    {"get_elem", nerv_matrix_(get_elem)},
    {"set_elem", nerv_matrix_(set_elem)},
    {"ncol", nerv_matrix_(ncol)},
    {"nrow", nerv_matrix_(nrow)},
    {"get_dataref_value", nerv_matrix_(get_dataref_value)},
    {"__index__", nerv_matrix_(index)},
    {"__newindex__", nerv_matrix_(newindex)},
    {NULL, NULL}
};

void nerv_matrix_(init)(lua_State *L) {
    luaT_newmetatable(L, nerv_matrix_(tname), MATRIX_BASE_TNAME,
                        nerv_matrix_(new), nerv_matrix_(destroy), NULL);
    luaL_register(L, NULL, nerv_matrix_(methods));
#ifdef MATRIX_INIT
    MATRIX_INIT(L);
#endif
    lua_pop(L, 1);
}
#endif