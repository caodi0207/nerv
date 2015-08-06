#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "matrix/matrix.h"
#include "common.h"
#include "luaT/luaT.h"
#include <stdio.h>

const char *nerv_matrix_host_float_tname = "nerv.MMatrixFloat";
const char *input_name = "_nerv_embed_input";
const char *output_name = "_nerv_embed_output";
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
    lua_pushstring(L, "../nerv/examples/swb_baseline.lua");
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
    /* add reference to avoid gc */
    luaT_pushudata(L, output, nerv_matrix_host_float_tname);
    luaT_pushudata(L, input, nerv_matrix_host_float_tname);
    lua_setfield(L, LUA_GLOBALSINDEX, input_name);
    lua_setfield(L, LUA_GLOBALSINDEX, output_name);
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
    lua_getfield(L, LUA_GLOBALSINDEX, input_name);
    lua_getfield(L, LUA_GLOBALSINDEX, output_name);
    /* lua stack now: input width, output width, propagator, propagator, input, output */
    if (lua_pcall(L, 2, 0, 0)) /* call propagator with two parameters */
    {
        printf("%s\n", luaL_checkstring(L, -1));
        exit(-1);
    }
    /* lua stack now: input width, output width, propagator */
    printf("## output: %ld %ld ##\n", output->nrow, output->ncol);
    for (i = 0; i < output->nrow; i++) /* nrow is actually 1 */
    {
        float *nerv_row = (float *)((char *)output->data.f + i * output->stride);
        for (j = 0; j < output->ncol; j++)
        {
            printf("%.8f ", nerv_row[j]);
        }
    }
}

void teardown_nerv() {
    lua_pushnil(L);
    lua_pushnil(L);
    lua_setfield(L, LUA_GLOBALSINDEX, input_name);
    lua_setfield(L, LUA_GLOBALSINDEX, output_name);
    lua_gc(L, LUA_GCCOLLECT, 0);
}

int main() {
    setup_nerv();
    propagate(1.0);
    propagate(2.0);
    propagate(2.0);
    propagate(3.0);
    teardown_nerv();
    return 0;
}
