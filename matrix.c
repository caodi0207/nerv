#include "common.h"

typedef struct Matrix {
    long stride;       /* size of a row */
    long ncol, nrow, nmax;    /* dimension of the matrix */
    union {
        float *f;
        double *d;
    } storage;          /* pointer to actual storage */
} Matrix;

const char *float_matrix_tname = "nerv.FloatMatrix";
const char *matrix_tname = "nerv.Matrix";

int float_matrix_new(lua_State *L) {
    Matrix *self = (Matrix *)malloc(sizeof(Matrix));
    self->nrow = luaL_checkinteger(L, 1);
    self->ncol = luaL_checkinteger(L, 2);
    self->nmax = self->nrow * self->ncol;
    self->stride = sizeof(float) * self->nrow;
    self->storage.f = (float *)malloc(self->stride * self->ncol);
    luaT_pushudata(L, self, float_matrix_tname);
    return 1;
}

int float_matrix_destroy(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    free(self->storage.f);
    fprintf(stderr, "[debug] destroyted\n");
    return 0;
}

int nerv_float_matrix_get_elem(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    int idx = luaL_checkinteger(L, 2);
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    lua_pushnumber(L, self->storage.f[idx]);
    return 1;
}

int nerv_float_matrix_set_elem(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, float_matrix_tname);
    int idx = luaL_checkinteger(L, 2);
    float v = luaL_checknumber(L, 3);
    long upper = self->nrow * self->ncol;
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    self->storage.f[idx] = v;
    return 0;
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
