#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "../common.h"
#include "chunk_file.h"

#define INVALID_FORMAT_ERROR(fn) \
    nerv_error(L, "invalid chunk file: %s", fn)
#define CHECK_FORMAT(exp, ret, fname) \
    do { \
        if ((exp) != (ret)) INVALID_FORMAT_ERROR(fn); \
    } while (0)

const char *nerv_chunk_file_tname = "nerv.ChunkFile";
const char *nerv_chunk_file_handle_tname = "nerv.ChunkFileHandle";
const char *nerv_chunk_info_tname = "nerv.ChunkInfo";
const char *nerv_chunk_data_tname = "nerv.ChunkData";

int nerv_lua_chunk_file_new(lua_State *L) {
    int status;
    const char *fn = luaL_checkstring(L, 1);
    ChunkFile *cfp = nerv_chunk_file_create(fn,
                                            luaL_checkstring(L, 2),
                                            &status);
    if (status != CF_NORMAL)
    {
        nerv_error(L, "%s: %s", fn, nerv_chunk_file_errstr(status));
    }
    lua_newtable(L);
    luaT_pushudata(L, cfp, nerv_chunk_file_handle_tname);
    lua_setfield(L, -2, "handle");
    if (cfp->status == CF_READ)
    {
        ChunkInfo *cip;
        /* build a table with interpreted metadata in Lua because C API only
         * provides with linked list with uninterpreted metadata */
        lua_newtable(L);
        /* stack: self, metadata_table */
        for (cip = cfp->info; cip; cip = cip->next)
        {
            luaL_loadstring(L, cip->metadata);
            CHECK_FORMAT(lua_pcall(L, 0, 1, 0), 0, fn);
            CHECK_FORMAT(lua_istable(L, -1), 1, fn);
            lua_getfield(L, -1, "id");
            if (!lua_isstring(L, -1))
                nerv_error(L, "id field in metadata must be a string");
            /* stack: ... metadata_table, metadata, id */
            lua_pushvalue(L, -1);
            /* stack: ... metadata_table, metadata, id, id */
            lua_gettable(L, -4);
            /* stack: ... metadata_table, metadata, id, metadata_table[id] */
            if (!lua_isnil(L, -1))
                nerv_error(L, "conflicting id");
            lua_pop(L, 1);
            /* stack: ... metadata_table, metadata, id */
            lua_pushvalue(L, -2);
            /* stack: ... metadata_table, metadata, id, metadata */
            lua_settable(L, -4);
            /* stack: ... metadata_table, metadata */
            luaT_pushudata(L, cip, nerv_chunk_info_tname);
            /* stack: ... metadata_table, cip */
            lua_setfield(L, -2, "_chunk_info");
            /* stack: ... metadata_table */
            lua_pop(L, 1);
            /* stack: ... metadata_table */
        }
        lua_setfield(L, -2, "metadata");
        /* stack: ... */
    }
    luaT_pushmetatable(L, nerv_chunk_file_tname);
    lua_setmetatable(L, -2);
    return 1;
}

static void writer(void *L) {
    lua_call((lua_State *)L, 2, 0); /* let the write() to write */
}

int nerv_lua_chunk_file_write_chunkdata(lua_State *L) {
    int status;
    ChunkFile *cfp = luaT_checkudata(L, 1, nerv_chunk_file_handle_tname);
    const char *mdstr = lua_tolstring(L, 2, NULL);
    lua_getfield(L, 3, "write");
    if (!lua_isfunction(L, -1))
        nerv_error(L, "\"write\" method must be implemented");
    lua_pushvalue(L, 3); /* lua writer itself */
    lua_pushvalue(L, 1); /* pass handle as parameter to write() */
    nerv_chunk_file_write_chunkdata(cfp, mdstr, writer, (void *)L);
    return 0;
}

int nerv_lua_chunk_file_get_chunkdata(lua_State *L) {
    int status;
    ChunkFile *cfp = luaT_checkudata(L, 1, nerv_chunk_file_handle_tname);
    ChunkInfo *cip = luaT_checkudata(L, 2, nerv_chunk_info_tname);
    ChunkData *cdp = nerv_chunk_file_get_chunkdata(cfp, cip, &status);
    if (status != CF_NORMAL)
        nerv_error(L, "%s", nerv_chunk_file_errstr(status));
    luaT_pushudata(L, cdp, nerv_chunk_data_tname);
    return 1;
}

int nerv_lua_chunk_file_close(lua_State *L) {
    ChunkFile *cfp = luaT_checkudata(L, -1, nerv_chunk_file_handle_tname);
    nerv_chunk_file_close(cfp);
    return 0;
}

int nerv_lua_chunk_file_destroy(lua_State *L) {
    ChunkFile *cfp = luaT_checkudata(L, -1, nerv_chunk_file_handle_tname);
    nerv_chunk_file_destroy(cfp);
    return 0;
}

static int nerv_lua_chunk_data_destroy(lua_State *L) {
    ChunkData *cdp = luaT_checkudata(L, 1, nerv_chunk_data_tname);
    nerv_chunk_data_destroy(cdp);
    return 0;
}

static const luaL_Reg nerv_chunk_file_methods[] = {
    {"_get_chunkdata", nerv_lua_chunk_file_get_chunkdata},
    {"_write_chunkdata", nerv_lua_chunk_file_write_chunkdata},
    {"_close", nerv_lua_chunk_file_close},
    {NULL, NULL}
};

void nerv_chunk_file_init(lua_State *L) {
    luaT_newmetatable(L, nerv_chunk_file_tname, NULL,
                        nerv_lua_chunk_file_new,
                        NULL, NULL);
    luaL_register(L, NULL, nerv_chunk_file_methods);
    lua_pop(L, 1);
    luaT_newmetatable(L, nerv_chunk_file_handle_tname, NULL,
                        NULL, nerv_lua_chunk_file_destroy, NULL);
    luaT_newmetatable(L, nerv_chunk_info_tname, NULL,
                        NULL, NULL, NULL);
    luaT_newmetatable(L, nerv_chunk_data_tname, NULL,
                        NULL, nerv_lua_chunk_data_destroy, NULL);
}
