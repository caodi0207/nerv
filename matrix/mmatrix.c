#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_FLOAT
#define host_matrix_(NAME) host_matrix_float_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_float_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixFloat";
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_DOUBLE
#define host_matrix_(NAME) host_matrix_double_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_double_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixDouble";
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_DOUBLE
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_INT
#define host_matrix_(NAME) host_matrix_int_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_int_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixInt";
#define MMATRIX_INIT(L) host_matrix_(init_extra)(L)

static const luaL_Reg nerv_matrix_(extra_methods_int)[];
static void host_matrix_(init_extra)(lua_State *L) {
    luaN_append_methods(L, nerv_matrix_(extra_methods_int));
}

#include "generic/mmatrix.c"

static int nerv_matrix_(perm_gen)(lua_State *L) {
    int i, ncol = luaL_checkinteger(L, 1);
    Matrix *self = nerv_matrix_(new_)(L, 1, ncol);
    long *prow = self->data.i;
    for (i = 0; i < ncol; i++)
        prow[i] = i;
    for (i = ncol - 1; i >= 0; i--)
    {
        size_t j = rand() % (i + 1);
        long tmp = prow[i];
        prow[i] = prow[j];
        prow[j] = tmp;
    }
    luaT_pushudata(L, self, nerv_matrix_(tname));
    return 1;
}

static const luaL_Reg nerv_matrix_(extra_methods_int)[] = {
    {"perm_gen", nerv_matrix_(perm_gen)},
    {NULL, NULL}
};

