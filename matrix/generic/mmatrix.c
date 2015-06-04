#ifdef NERV_GENERIC_MMATRIX
#include "matrix.h"
#include "elem_type.h"
#define MATRIX_DATA_FREE(L, ptr) free(ptr)
#define MATRIX_DATA_ALLOC(L, dptr, stride, width, height) \
                            host_matrix_(alloc)(L, dptr, stride, width, height)
#define MATRIX_DATA_WRITE(L, data, idx, val) (data[idx] = val)
#define MATRIX_DATA_READ(L, data, idx) (data[idx])
#define MATRIX_INIT(L) host_matrix_(init)(L)
#define MATRIX_BASE_TNAME nerv_matrix_host_tname
#define NERV_GENERIC_MATRIX
#include "../../common.h"
#include "../../io/chunk_file.h"
#include "string.h"

static void host_matrix_(alloc)(lua_State *L,
                                MATRIX_ELEM **dptr, size_t *stride,
                                long width, long height) {
    if ((*dptr = (MATRIX_ELEM *)malloc(width * height)) == NULL)
        nerv_error(L, "mmatrix insufficient memory");
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
    ChunkData *chunk = luaT_checkudata(L, 1, nerv_chunk_data_tname);
    Matrix *self;
    int i, j;
    long nrow, ncol;
    FILE *fp = chunk->fp;
    if (fscanf(fp, "%ld %ld", &nrow, &ncol) != 2)
        return 0;
    self = nerv_matrix_(new_)(L, nrow, ncol);
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
    ChunkFileHandle *chunk = luaT_checkudata(L, 2,
                                nerv_chunk_file_handle_tname);
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

static int nerv_matrix_(copy_from)(lua_State *L) {
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    int nargs = lua_gettop(L);
    int b_begin = nargs > 2 ? luaL_checkinteger(L, 3) : 0;
    int b_end = nargs > 3 ? luaL_checkinteger(L, 4) : b->nrow;
    int a_begin = nargs > 4 ? luaL_checkinteger(L, 5) : 0;
    if (!(0 <= b_begin && b_begin < b_end && b_end <= b->nrow &&
            a_begin + b_end - b_begin <= a->nrow))
        nerv_error(L, "invalid copy interval");
    if (a->ncol != b->ncol)
        nerv_error(L, "matrices should be of the same dimension");
    memmove(MATRIX_ROW_PTR(a, a_begin),
            MATRIX_ROW_PTR(b, b_begin),
            sizeof(MATRIX_ELEM) * b->ncol * (b_end - b_begin));
    return 0;
}
static const luaL_Reg nerv_matrix_(extra_methods)[] = {
    {"load", nerv_matrix_(load)},
    {"save", nerv_matrix_(save)},
    {"copy_from", nerv_matrix_(copy_from)},
    {NULL, NULL}
};

#endif
