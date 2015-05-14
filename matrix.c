#include "common.h"

typedef struct Matrix {
    long stride;              /* size of a row */
    long ncol, nrow, nmax;    /* dimension of the matrix */
    union {
        float *f;
        double *d;
    } data;                   /* pointer to actual storage */
    long *data_ref;
} Matrix;

const char *float_matrix_tname = "nerv.FloatMatrix";
const char *matrix_tname = "nerv.Matrix";

void float_matrix_data_free(Matrix *self) {
    if (--(*self->data_ref) == 0)
        free(self->data.f);
}

void float_matrix_data_retain(Matrix *self) {
    (*self->data_ref)++;
}

int float_matrix_new(lua_State *L) {
    Matrix *self = (Matrix *)malloc(sizeof(Matrix));
    self->nrow = luaL_checkinteger(L, 1);
    self->ncol = luaL_checkinteger(L, 2);
    self->nmax = self->nrow * self->ncol;
    self->stride = sizeof(float) * self->ncol;
    self->data.f = (float *)malloc(self->stride * self->nrow);
    self->data_ref = (long *)malloc(sizeof(long));
    *self->data_ref = 0;
    float_matrix_data_retain(self);
    luaT_pushudata(L, self, float_matrix_tname);
    return 1;
}

int float_matrix_destroy(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    float_matrix_data_free(self);
    return 0;
}

int nerv_float_matrix_get_elem(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    int idx = luaL_checkinteger(L, 2);
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    lua_pushnumber(L, self->data.f[idx]);
    return 1;
}

int nerv_float_matrix_set_elem(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    int idx = luaL_checkinteger(L, 2);
    float v = luaL_checknumber(L, 3);
    long upper = self->nrow * self->ncol;
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    self->data.f[idx] = v;
    return 0;
}

static Matrix *nerv_float_matrix_getrow(Matrix *self, int row) {
    Matrix *prow = (Matrix *)malloc(sizeof(Matrix));
    prow->ncol = self->ncol;
    prow->nrow = 1;
    prow->stride = self->stride;
    prow->nmax = prow->ncol;
    prow->data.f = (float *)((char *)self->data.f + row * self->stride);
    prow->data_ref = self->data_ref;
    float_matrix_data_retain(self);
    return prow;
}

static int nerv_float_matrix_newindex(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    if (lua_isnumber(L, 2))
    {
        int idx = luaL_checkinteger(L, 2);
        if (self->nrow == 1)
        {
            if (idx < 0 || idx >= self->ncol)
                nerv_error(L, "index must be within range [0, %d)", self->ncol);
            self->data.f[idx] = luaL_checknumber(L, 3);
        }
        else
            nerv_error(L, "cannot assign a scalar to row vector");
        lua_pushboolean(L, 1);
        return 2;
    }
    else
    {
        lua_pushboolean(L, 0);
        return 1;
    }
}


static int nerv_float_matrix_index(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    if (lua_isnumber(L, 2))
    {
        int idx = luaL_checkinteger(L, 2);
        if (self->nrow == 1)
        {
            if (idx < 0 || idx >= self->ncol)
                nerv_error(L, "index must be within range [0, %d)", self->ncol);
            lua_pushnumber(L, self->data.f[idx]);
        }
        else
        {
            if (idx < 0 || idx >= self->nrow)
                nerv_error(L, "index must be within range [0, %d)", self->nrow);
            luaT_pushudata(L, nerv_float_matrix_getrow(self, idx), float_matrix_tname);
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

static int nerv_float_matrix_ncol(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    lua_pushinteger(L, self->ncol);
    return 1;
}

static int nerv_float_matrix_nrow(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    lua_pushinteger(L, self->nrow);
    return 1;
}


static const luaL_Reg float_matrix_methods[] = {
    {"get_elem", nerv_float_matrix_get_elem},
    {"set_elem", nerv_float_matrix_set_elem},
    {"ncol", nerv_float_matrix_ncol},
    {"nrow", nerv_float_matrix_nrow},
    {"__index__", nerv_float_matrix_index},
    {"__newindex__", nerv_float_matrix_newindex},
    {NULL, NULL}
};

void nerv_float_matrix_init(lua_State *L) {
    luaT_newmetatable(L, float_matrix_tname, matrix_tname,
                        float_matrix_new, float_matrix_destroy, NULL);
    luaL_register(L, NULL, float_matrix_methods);
    lua_pop(L, 1);
}

static const luaL_Reg matrix_methods[] = {
    {"__tostring__", nerv_error_method_not_implemented },
    {"__add__", nerv_error_method_not_implemented },
    {"__sub__", nerv_error_method_not_implemented },
    {"__mul__", nerv_error_method_not_implemented },
    {NULL, NULL}
};

void nerv_matrix_init(lua_State *L) {
    /* abstract class */
    luaT_newmetatable(L, matrix_tname, NULL, NULL, NULL, NULL);
    luaL_register(L, NULL, matrix_methods);
    lua_pop(L, 1);
    nerv_float_matrix_init(L);
}
