#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "../common.h"
#include "param.h"

#define INVALID_FORMAT_ERROR(fn) \
    nerv_error(L, "Invalid param file: %s", fn)
#define CHECK_FORMAT(exp, ret, fname) \
    do { \
        if ((exp) != (ret)) INVALID_FORMAT_ERROR(fn); \
    } while (0)

const char *nerv_param_file_tname = "nerv.ParamFile";
const char *nerv_param_file_handle_tname = "nerv.ParamFileHandle";
const char *nerv_param_chunk_info_tname = "nerv.ParamChunkInfo";
const char *nerv_param_chunk_data_tname = "nerv.ParamChunkData";

#define PARAM_HEADER_SIZE 16

enum {
    NORMAL,
    INVALID_FORMAT,
    END_OF_FILE
};

size_t read_param_header_plain(FILE *fp, int *status) {
    static char buff[PARAM_HEADER_SIZE];
    int i;
    size_t size = 0;
    *status = NORMAL;
    if (fread(buff, 1, PARAM_HEADER_SIZE, fp) != PARAM_HEADER_SIZE)
    {
        if (feof(fp)) *status = END_OF_FILE;
        else *status = INVALID_FORMAT;
    }
    for (i = 0; i < PARAM_HEADER_SIZE; i++)
        if (isdigit(buff[i]))
            size = size * 10 + buff[i] - '0';
    fprintf(stderr, "header: %d\n", size);
    return size;
}

ParamChunkData *get_param_chunk_data(FILE *fp, ParamChunkInfo *info) {
    ParamChunkData *pcd = (ParamChunkData *)malloc(sizeof(ParamChunkData));
    pcd->data = (char *)malloc(info->length);
    pcd->fp = fmemopen(pcd->data, info->length, "r");
    assert(fseeko(fp, info->offset, SEEK_SET) == 0);
    assert(fread(pcd->data, 1, info->length, fp) == (size_t)info->length);
    return pcd;
}

const char *read_param_metadata(lua_State *L, FILE *fp, const char *fn) {
#define LINEBUFF_SIZE 1024
    static char buff[7 + LINEBUFF_SIZE] = "return ";
    CHECK_FORMAT(fgets(buff + 7, LINEBUFF_SIZE, fp), buff + 7, fn);
    fprintf(stderr, "metadata: %s\n", buff);
    return buff;
}

int nerv_param_file_open_write(lua_State *L, const char *fn) {
    FILE *fp = fopen(fn, "w");
    if (!fp) nerv_error(L, "Error while opening param file: %s", fn);
    lua_newtable(L);
    return 1;
}

int nerv_param_file_open_read(lua_State *L, const char *fn) {
    FILE *fp = fopen(fn, "r");
    int i, status;
    size_t param_len;
    off_t offset;
    ParamFileHandle *lfp;

    if (!fp) nerv_error(L, "Error while opening param file: %s", fn);
    offset = ftello(fp);
    lua_newtable(L);
    fprintf(stderr, "%d\n", (int)offset);
    for (i = 0;; offset += param_len, i++)
    {
        ParamChunkInfo *pci;
        fprintf(stderr, "reading param chunk %d from %d\n", i, (int)offset);
        /* skip to the begining of param chunk i */
        CHECK_FORMAT(fseeko(fp, offset, SEEK_SET), 0, fn);
        /* read header */
        param_len = read_param_header_plain(fp, &status);
        if (status == END_OF_FILE) break;
        else if (status == INVALID_FORMAT)
            INVALID_FORMAT_ERROR(fn);
        /* read metadata */
        luaL_loadstring(L, read_param_metadata(L, fp, fn));
        CHECK_FORMAT(lua_pcall(L, 0, 1, 0), 0, fn);
        CHECK_FORMAT(lua_istable(L, -1), 1, fn);
        /* chunk info */
        pci = (ParamChunkInfo *)malloc(sizeof(ParamChunkInfo));
        pci->offset = ftello(fp);
        pci->length = param_len - (pci->offset - offset);
        fprintf(stderr, "%d + %d (skip %d)\n", (int)pci->offset,
                (int)pci->length, param_len);
        luaT_pushudata(L, pci, nerv_param_chunk_info_tname);
        lua_setfield(L, -2, "chunk");
        lua_rawseti(L, -2, i);
    }
    lua_setfield(L, -2, "metadata");
    lfp = (ParamFileHandle *)malloc(sizeof(ParamFileHandle));
    lfp->fp = fp;
    luaT_pushudata(L, lfp, nerv_param_file_handle_tname);
    lua_setfield(L, -2, "handle");
    luaT_pushmetatable(L, nerv_param_file_tname);
    lua_setmetatable(L, -2);
    return 1;
}

int nerv_param_file___init(lua_State *L) {
    const char *fn = luaL_checkstring(L, 2);
    const char *mode = luaL_checkstring(L, 3);
    int rd = 1, bin = 0;
    size_t i, len = strlen(mode);
    lua_pushvalue(L, 1);
    for (i = 0; i < len; i++)
        switch (mode[i])
        {
            case 'r': rd = 1; break;
            case 'w': rd = 0; break;
            case 'b': bin = 1; break;
        }
    return rd ? nerv_param_file_open_read(L, fn) : \
                nerv_param_file_open_write(L, fn);
}

int nerv_param_file_new(lua_State *L) {
    const char *fn = luaL_checkstring(L, 1);
    const char *mode = luaL_checkstring(L, 2);
    int rd = 1, bin = 0;
    size_t i, len = strlen(mode);
    for (i = 0; i < len; i++)
        switch (mode[i])
        {
            case 'r': rd = 1; break;
            case 'w': rd = 0; break;
            case 'b': bin = 1; break;
        }
    lua_newtable(L);
    return rd ? nerv_param_file_open_read(L, fn) : \
                nerv_param_file_open_write(L, fn);
}

int nerv_param_file_get_chunkdata(lua_State *L) {
    ParamFileHandle *pfh;
    ParamChunkInfo *pci;
    int k = luaL_checkinteger(L, 2);

    lua_getfield(L, 1, "handle");
    pfh = luaT_checkudata(L, -1, nerv_param_file_handle_tname);
    lua_pop(L, 1); /* pop handle */

    lua_getfield(L, 1, "metadata");
    /* now stack: self, k, metadata */
    lua_rawgeti(L, -1, k);
    /* now stack: self, k, metadata, ith{} */
    lua_getfield(L, -1, "chunk");
    pci = luaT_checkudata(L, -1, nerv_param_chunk_info_tname);

    luaT_pushudata(L, get_param_chunk_data(pfh->fp, pci),
                        nerv_param_chunk_data_tname);
    return 1;
}

int nerv_param_file_handle_destroy(lua_State *L) {
    ParamFileHandle *pfh = luaT_checkudata(L, 1,
                                nerv_param_file_handle_tname);
    fclose(pfh->fp);
    free(pfh);
    return 0;
}

static int nerv_param_chunk_destroy(lua_State *L) {
    ParamChunkInfo *pci = luaT_checkudata(L, 1, nerv_param_chunk_info_tname);
    free(pci);
    return 0;
}

static int nerv_param_chunk_data_destroy(lua_State *L) {
    ParamChunkData *pcd = luaT_checkudata(L, 1, nerv_param_chunk_data_tname);
    fclose(pcd->fp);
    free(pcd->data);
    free(pcd);
    return 0;
}

static const luaL_Reg nerv_param_file_methods[] = {
    {"get_chunkdata", nerv_param_file_get_chunkdata},
    {"__init", nerv_param_file___init},
    {NULL, NULL}
};

void nerv_param_file_init(lua_State *L) {
    luaT_newmetatable(L, nerv_param_file_tname, NULL,
                        nerv_param_file_new,
                        NULL, NULL);
    luaL_register(L, NULL, nerv_param_file_methods);
    lua_pop(L, 1);
    luaT_newmetatable(L, nerv_param_file_handle_tname, NULL,
                        NULL, nerv_param_file_handle_destroy, NULL);
    luaT_newmetatable(L, nerv_param_chunk_info_tname, NULL,
                        NULL, nerv_param_chunk_destroy, NULL);
    luaT_newmetatable(L, nerv_param_chunk_data_tname, NULL,
                        NULL, nerv_param_chunk_data_destroy, NULL);
}

