#ifdef NERV_GENERIC_MMATRIX
#include "matrix.h"
#include "elem_type.h"
#define MATRIX_DATA_FREE(ptr) free(ptr)
#define MATRIX_DATA_ALLOC(dptr, stride, width, height) \
                            host_matrix_(alloc)(dptr, stride, width, height)
#define MATRIX_DATA_STRIDE(ncol) (sizeof(MATRIX_ELEM) * (ncol))
#define MATRIX_DATA_WRITE(data, idx, val) (data[idx] = val)
#define MATRIX_DATA_READ(data, idx) (data[idx])
#define MATRIX_INIT(L) host_matrix_(init)(L)
#define MATRIX_BASE_TNAME nerv_matrix_host_tname
#define NERV_GENERIC_MATRIX
#include "../../common.h"
#include "../../io/param.h"

static void host_matrix_(alloc)(MATRIX_ELEM **dptr, size_t *stride,
                                    long width, long height) {
    *dptr = (MATRIX_ELEM *)malloc(width * height);
    *stride = width;
}

int nerv_matrix_(get_elem)(lua_State *L) {
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    int idx = luaL_checkinteger(L, 2);
    if (idx < 0 || idx >= self->nmax)
        nerv_error(L, "index must be within range [0, %d)", self->nmax);
    lua_pushnumber(L, MATRIX_ELEM_PTR(self)[idx]);
    return 1;
}

int nerv_matrix_(set_elem)(lua_State *L) {
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
}

#include "matrix.c"

int nerv_matrix_(load)(lua_State *L) {
    ParamChunkData *chunk = luaT_checkudata(L, 1, nerv_param_chunk_data_tname);
    Matrix *self;
    int i, j;
    long nrow, ncol;
    FILE *fp = chunk->fp;
    if (fscanf(fp, "%ld %ld", &nrow, &ncol) != 2)
        return 0;
    self = nerv_matrix_(new_)(nrow, ncol);
    for (i = 0; i < nrow; i++)
    {
        MATRIX_ELEM *row = MATRIX_ROW_PTR(self, i);
        for (j = 0; j < ncol; j++)
            if (fscanf(fp, MATRIX_ELEM_FMT, row + j) != 1)
            {
                free(self);
                return 0;
            }
    }
    luaT_pushudata(L, self, nerv_matrix_(tname));
    return 1;
}

int nerv_matrix_(save)(lua_State *L) {
    ParamFileHandle *chunk = luaT_checkudata(L, 2,
                                nerv_param_file_handle_tname);
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    int i, j;
    long nrow = self->nrow, ncol = self->ncol;
    FILE *fp = chunk->fp;
    if (fprintf(fp, "%ld %ld\n", nrow, ncol) < 0)
        return 0;
    for (i = 0; i < nrow; i++)
    {
        MATRIX_ELEM *row = MATRIX_ROW_PTR(self, i);
        for (j = 0; j < ncol; j++)
            if (fprintf(fp, MATRIX_ELEM_FMT " ", row[j]) < 0)
            {
                free(self);
                return 0;
            }
        if (fprintf(fp, "\n") < 0)
        {
             free(self);
             return 0;
        }
    }
    return 0;
}


static const luaL_Reg nerv_matrix_(extra_methods)[] = {
    {"load", nerv_matrix_(load)},
    {"save", nerv_matrix_(save)},
    {NULL, NULL}
};

#endif
