#ifdef NERV_GENERIC_CUMATRIX
#include "../../lib/matrix/generic/matrix.h"
#include "elem_type.h"
#define MATRIX_DATA_WRITE(L, data, idx, val) cuda_matrix_(write)(L, data, idx, val)
#define MATRIX_DATA_READ(L, data, idx) cuda_matrix_(read)(L, data, idx)
#define MATRIX_INIT(L) cuda_matrix_(init)(L)
#define MATRIX_BASE_TNAME nerv_matrix_cuda_tname
#define NERV_GENERIC_MATRIX
#define NERV_GENERIC_CUKERNEL
#include "../../common.h"
#include "../../lib/matrix/generic/cumatrix.h"

static int nerv_matrix_(lua_add)(lua_State *L) {
    Status status;
    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    MATRIX_ELEM alpha = luaL_checknumber(L, 4);
    MATRIX_ELEM beta = luaL_checknumber(L, 5);
    nerv_matrix_(add)(c, a, b, alpha, beta, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_get_cublas_op)(char ch) {
    return (ch == 'T' || ch == 't') ? CUBLAS_OP_T : CUBLAS_OP_N;
}

static int nerv_matrix_(lua_mul)(lua_State *L) {
    Status status;
    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    MATRIX_ELEM alpha = luaL_checknumber(L, 4);
    MATRIX_ELEM beta = luaL_checknumber(L, 5);
    int nargs = lua_gettop(L);
    int ta = nargs > 5 ? nerv_matrix_(lua_get_cublas_op)(*luaL_checkstring(L, 6)) \
                            : CUBLAS_OP_N;
    int tb = nargs > 6 ? nerv_matrix_(lua_get_cublas_op)(*luaL_checkstring(L, 7)) \
                            : CUBLAS_OP_N;
    nerv_matrix_(mul)(c, a, b, alpha, beta, ta, tb, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_create)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(create)(a->nrow, a->ncol, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(lua_sigmoid)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    nerv_matrix_(sigmoid)(a, b, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_sigmoid_grad)(lua_State *L) {
    Status status;
    Matrix *nerr = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *err = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *output = luaT_checkudata(L, 3, nerv_matrix_(tname));
    nerv_matrix_(sigmoid_grad)(nerr, err, output, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_softmax)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *max_idx = nerv_matrix_(softmax)(b, a, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, max_idx, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(lua_rowsum)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(rowsum)(a, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(lua_colsum)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(colsum)(a, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(lua_colsame)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *ref = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(colsame)(a, ref, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(lua_rowmax)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(rowmax)(a, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(lua_rowmax_idx)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b;
    Matrix *idx;
    nerv_matrix_(rowmax_idx)(a, &b, &idx, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    luaT_pushudata(L, idx, nerv_matrix_(tname));
    return 2;
}

static int nerv_matrix_(lua_add_row)(lua_State *L) {
    Status status;
    const Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    double beta = luaL_checknumber(L, 3);
    nerv_matrix_(add_row)(b, a, beta, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_fill)(lua_State *L) {
    Status status;
    Matrix *self = luaT_checkudata(L, 1, nerv_matrix_(tname));
    double val = luaL_checknumber(L, 2);
    nerv_matrix_(fill)(self, val, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_copy_fromd)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    int nargs = lua_gettop(L);
    int b_begin = nargs > 2 ? luaL_checkinteger(L, 3) : 0;
    int b_end = nargs > 3 ? luaL_checkinteger(L, 4) : b->nrow;
    int a_begin = nargs > 4 ? luaL_checkinteger(L, 5) : 0;
    nerv_matrix_(copy_fromd)(a, b, a_begin, b_begin, b_end, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

extern const char *MATRIX_CUMATRIX_HOST_TNAME;
static int nerv_matrix_(lua_copy_fromh)(lua_State *L) { 
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    int nargs = lua_gettop(L);
    int b_begin = nargs > 2 ? luaL_checkinteger(L, 3) : 0;
    int b_end = nargs > 3 ? luaL_checkinteger(L, 4) : b->nrow;
    int a_begin = nargs > 4 ? luaL_checkinteger(L, 5) : 0;
    nerv_matrix_(copy_fromh)(a, b, a_begin, b_begin, b_end, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_copy_toh)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    int nargs = lua_gettop(L);
    int a_begin = nargs > 2 ? luaL_checkinteger(L, 3) : 0;
    int a_end = nargs > 3 ? luaL_checkinteger(L, 4) : a->nrow;
    int b_begin = nargs > 4 ? luaL_checkinteger(L, 5) : 0;
    nerv_matrix_(copy_toh)(a, b, a_begin, a_end, b_begin, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_trans)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    Matrix *b = nerv_matrix_(trans)(a, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

static int nerv_matrix_(lua_mul_elem)(lua_State *L) {
    Status status;
    const Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 3, nerv_matrix_(tname));
    Matrix *c = luaT_checkudata(L, 1, nerv_matrix_(tname));
    nerv_matrix_(mul_elem)(c, a, b, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_log_elem)(lua_State *L) {
    Status status;
    const Matrix *a = luaT_checkudata(L, 2, nerv_matrix_(tname));
    Matrix *b = luaT_checkudata(L, 1, nerv_matrix_(tname));
    nerv_matrix_(log_elem)(b, a, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_decompress)(lua_State *L) {
    Status status;
    const Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    int orig_col = luaL_checkinteger(L, 2);
    Matrix *b = nerv_matrix_(decompress)(a, orig_col, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    luaT_pushudata(L, b, nerv_matrix_(tname));
    return 1;
}

extern const char *nerv_matrix_host_int_tname;
static int nerv_matrix_(lua_copy_rows_fromh_by_idx)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, MATRIX_CUMATRIX_HOST_TNAME);
    const Matrix *idx = luaT_checkudata(L, 3, nerv_matrix_host_int_tname);
    long nrow = a->nrow;
    int b_begin = lua_gettop(L) > 3 ? luaL_checkinteger(L, 4) : 0;
    nerv_matrix_(copy_rows_fromh_by_idx)(a, b, idx, b_begin, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_expand_frm)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    int context = luaL_checkinteger(L, 3);
    nerv_matrix_(expand_frm)(a, b, context, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_rearrange_frm)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    int step = luaL_checkinteger(L, 3);
    nerv_matrix_(rearrange_frm)(a, b, step, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_scale_rows_by_col)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    nerv_matrix_(scale_rows_by_col)(a, b, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static int nerv_matrix_(lua_scale_rows_by_row)(lua_State *L) {
    Status status;
    Matrix *a = luaT_checkudata(L, 1, nerv_matrix_(tname));
    const Matrix *b = luaT_checkudata(L, 2, nerv_matrix_(tname));
    nerv_matrix_(scale_rows_by_row)(a, b, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    return 0;
}

static const luaL_Reg nerv_matrix_(extra_methods)[] = {
    {"create", nerv_matrix_(lua_create)},
    {"colsum", nerv_matrix_(lua_colsum)},
    {"colsame", nerv_matrix_(lua_colsame)},
    {"rowsum", nerv_matrix_(lua_rowsum)},
    {"rowmax", nerv_matrix_(lua_rowmax)},
    {"rowmax_idx", nerv_matrix_(lua_rowmax_idx)},
    {"trans", nerv_matrix_(lua_trans)},
    {"decompress", nerv_matrix_(lua_decompress)},
    /* in-place calc */
    {"copy_fromh", nerv_matrix_(lua_copy_fromh)},
    {"copy_fromd", nerv_matrix_(lua_copy_fromd)},
    {"copy_toh", nerv_matrix_(lua_copy_toh)},
    {"add", nerv_matrix_(lua_add)},
    {"mul", nerv_matrix_(lua_mul)},
    {"add_row", nerv_matrix_(lua_add_row)},
    {"fill", nerv_matrix_(lua_fill)},
    {"sigmoid", nerv_matrix_(lua_sigmoid)},
    {"sigmoid_grad", nerv_matrix_(lua_sigmoid_grad)},
    {"softmax", nerv_matrix_(lua_softmax)},
    {"mul_elem", nerv_matrix_(lua_mul_elem)},
    {"log_elem", nerv_matrix_(lua_log_elem)},
    {"copy_rows_fromh_by_idx", nerv_matrix_(lua_copy_rows_fromh_by_idx)},
    {"expand_frm", nerv_matrix_(lua_expand_frm)},
    {"rearrange_frm", nerv_matrix_(lua_rearrange_frm)},
    {"scale_rows_by_row", nerv_matrix_(lua_scale_rows_by_row)},
    {"scale_rows_by_col", nerv_matrix_(lua_scale_rows_by_col)},
    {NULL, NULL}
};

static void cuda_matrix_(init)(lua_State *L) {
    luaN_append_methods(L, nerv_matrix_(extra_methods));
}

int nerv_matrix_(lua_get_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

int nerv_matrix_(lua_set_elem)(lua_State *L) {
    return nerv_error_method_not_implemented(L);
}

static MATRIX_ELEM cuda_matrix_(read)(lua_State *L, MATRIX_ELEM *data,
                                    int idx) {
    cudaError_t err;
    MATRIX_ELEM res;
    err = cudaMemcpy(&res, data + idx,
                sizeof(MATRIX_ELEM), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        nerv_error(L, "cuda error: error while reading element");
    cudaDeviceSynchronize();
    return res;
}

static void cuda_matrix_(write)(lua_State *L, MATRIX_ELEM *data,
                                int idx, MATRIX_ELEM val) {
    cudaError_t err;
    err = cudaMemcpy(data + idx, &val,
                sizeof(MATRIX_ELEM), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        nerv_error(L, "cuda error: error while writing element");
    cudaDeviceSynchronize();
}

#include "matrix.c"
#endif
