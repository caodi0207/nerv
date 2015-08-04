#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "matrix/matrix.h"
#include "common.h"
#include "luaT/luaT.h"
#include <stdio.h>

const char *nerv_matrix_host_float_tname = "nerv.MMatrixFloat";
extern Matrix *nerv_matrix_host_float_create(long nrow, long ncol, Status *status);
extern void nerv_matrix_host_float_data_retain(Matrix *self);
extern void nerv_matrix_host_float_data_free(Matrix *self, Status *status);

lua_State *L;
Matrix *input, *output;
Status status;

void setup_nerv() {
    L = lua_open();
    luaL_openlibs(L);
    luaL_loadfile(L, "setup_nerv.lua");
    /* network configuration */
    lua_pushstring(L, "swb_baseline_decode.lua");
    if (lua_pcall(L, 1, LUA_MULTRET, 0))
    {
        printf("%s\n", luaL_checkstring(L, 1));
        exit(1);
    }
    /* lua stack now: input width, output width, propagator */
    input = nerv_matrix_host_float_create(1, luaL_checkinteger(L, 1), &status);
    NERV_LUA_CHECK_STATUS(L, status);
    output = nerv_matrix_host_float_create(1, luaL_checkinteger(L, 2), &status);
    NERV_LUA_CHECK_STATUS(L, status);
}


void propagate(float for_fun) {
    int i, j;
    printf("ok: %d\n", lua_gettop(L));
    lua_pushvalue(L, 3);
    /* lua stack now: input width, output width, propagator, propagator */
    for (i = 0; i < input->nrow; i++) /* nrow is actually 1 */
    {
        float *nerv_row = (float *)((char *)input->data.f + i * input->stride);
        for (j = 0; j < input->ncol; j++)
        {
            nerv_row[j] = j * for_fun;
        }
    }
    /* avoid gc */
    nerv_matrix_host_float_data_retain(input);
    nerv_matrix_host_float_data_retain(output);

    luaT_pushudata(L, input, nerv_matrix_host_float_tname);
    luaT_pushudata(L, output, nerv_matrix_host_float_tname);
    /* lua stack now: input width, output width, propagator, propagator, input, output */
    if (lua_pcall(L, 2, 0, 0)) /* call propagator with two parameters */
    {
        printf("%s\n", luaL_checkstring(L, -1));
        exit(-1);
    }
    /* lua stack now: input width, output width, propagator */
    printf("## caller ##\n");
    for (i = 0; i < output->nrow; i++) /* nrow is actually 1 */
    {
        float *nerv_row = (float *)((char *)output->data.f + i * output->stride);
        for (j = 0; j < output->ncol; j++)
        {
            printf("%.8f ", nerv_row[j]);
        }
        printf("\n");
    }
}

void teardown_nerv() {
    nerv_matrix_host_float_data_free(input, &status);
    NERV_LUA_CHECK_STATUS(L, status);
    nerv_matrix_host_float_data_free(output, &status);
    NERV_LUA_CHECK_STATUS(L, status);
}

int main() {
    setup_nerv();
    propagate(1.0);
    propagate(2.0);
    propagate(3.0);
    teardown_nerv();
    return 0;
}
