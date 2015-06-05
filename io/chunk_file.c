#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "../common.h"
#include "chunk_file.h"

#define INVALID_FORMAT_ERROR(fn) \
    nerv_error(L, "Invalid chunk file: %s", fn)
#define CHECK_FORMAT(exp, ret, fname) \
    do { \
        if ((exp) != (ret)) INVALID_FORMAT_ERROR(fn); \
    } while (0)

const char *nerv_chunk_file_tname = "nerv.ChunkFile";
const char *nerv_chunk_file_handle_tname = "nerv.ChunkFileHandle";
const char *nerv_chunk_info_tname = "nerv.ChunkInfo";
const char *nerv_chunk_data_tname = "nerv.ChunkData";

#define PARAM_HEADER_SIZE 16

enum {
    NORMAL,
    INVALID_FORMAT,
    END_OF_FILE,
    SECTION_OVERFLOW,
    WRITE_ERROR
};

size_t read_chunk_header_plain(FILE *fp, int *status) {
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
    fprintf(stderr, "header: %lu\n", size);
    return size;
}

#define CHECK_WRITE(status) \
    do { \
        if (status == SECTION_OVERFLOW) \
            nerv_error(L, "section overflowed"); \
        else if (status == WRITE_ERROR) \
            nerv_error(L, "error while writing"); \
    } while (0)

void write_chunk_header_plain(FILE *fp, size_t size, int *status) {
    static char buff[PARAM_HEADER_SIZE];
    int i;
    *status = NORMAL;
    for (i = PARAM_HEADER_SIZE - 3; i > 0; i--, size /= 10)
        buff[i] = size % 10 + '0';
    if (size)
    {
        *status = SECTION_OVERFLOW;
        return;
    }
    buff[0] = '[';
    buff[PARAM_HEADER_SIZE - 2] = ']';
    buff[PARAM_HEADER_SIZE - 1] = '\n';
    if (fwrite(buff, 1, PARAM_HEADER_SIZE, fp) != PARAM_HEADER_SIZE)
    {
        *status = WRITE_ERROR;
        return;
    }
}

ChunkData *get_chunk_data(FILE *fp, ChunkInfo *info) {
    ChunkData *pcd = (ChunkData *)malloc(sizeof(ChunkData));
    pcd->data = (char *)malloc(info->length);
    pcd->fp = fmemopen(pcd->data, info->length, "r");
    assert(fseeko(fp, info->offset, SEEK_SET) == 0);
    if (fread(pcd->data, 1, info->length, fp) != (size_t)info->length)
        return NULL;
    return pcd;
}

const char *read_chunk_metadata(lua_State *L, FILE *fp, const char *fn) {
#define LINEBUFF_SIZE 1024
    static char buff[7 + LINEBUFF_SIZE] = "return ";
    CHECK_FORMAT(fgets(buff + 7, LINEBUFF_SIZE, fp), buff + 7, fn);
    fprintf(stderr, "metadata: %s\n", buff);
    return buff;
}

void write_chunk_metadata(FILE *fp, const char *metadata_str, int *status) {
    size_t size = strlen(metadata_str);
    *status = NORMAL;
    if (fwrite(metadata_str, 1, size, fp) != size ||
        fprintf(fp, "\n") < 0)
    {
        *status = WRITE_ERROR;
        return;
    }
    fprintf(stderr, "metadata: %s\n", metadata_str);
}


int nerv_chunk_file_open_write(lua_State *L, const char *fn) {
    FILE *fp = fopen(fn, "w");
    ChunkFileHandle *lfp;
    if (!fp) nerv_error(L, "Error while opening chunk file: %s", fn);
    lfp = (ChunkFileHandle *)malloc(sizeof(ChunkFileHandle));
    lfp->fp = fp;
    luaT_pushudata(L, lfp, nerv_chunk_file_handle_tname);
    lua_setfield(L, -2, "handle");
    luaT_pushmetatable(L, nerv_chunk_file_tname);
    lua_setmetatable(L, -2);
    return 1;
}

int nerv_chunk_file_open_read(lua_State *L, const char *fn) {
    FILE *fp = fopen(fn, "r");
    int i, status;
    size_t chunk_len;
    off_t offset;
    ChunkFileHandle *lfp;

    if (!fp) nerv_error(L, "Error while opening chunk file: %s", fn);
    offset = ftello(fp);
    lua_newtable(L);
    fprintf(stderr, "%d\n", (int)offset);
    for (i = 0;; offset += chunk_len, i++)
    {
        ChunkInfo *pci;
        fprintf(stderr, "reading chunk %d from %d\n", i, (int)offset);
        /* skip to the begining of chunk i */
        CHECK_FORMAT(fseeko(fp, offset, SEEK_SET), 0, fn);
        /* read header */
        chunk_len = read_chunk_header_plain(fp, &status);
        if (status == END_OF_FILE) break;
        else if (status == INVALID_FORMAT)
            INVALID_FORMAT_ERROR(fn);
        /* read metadata */
        luaL_loadstring(L, read_chunk_metadata(L, fp, fn));
        CHECK_FORMAT(lua_pcall(L, 0, 1, 0), 0, fn);
        CHECK_FORMAT(lua_istable(L, -1), 1, fn);
        /* stack: obj_table, metadata */
        /* chunk info */
        pci = (ChunkInfo *)malloc(sizeof(ChunkInfo));
        pci->offset = ftello(fp);
        pci->length = chunk_len - (pci->offset - offset);
        fprintf(stderr, "%d + %d (skip %lu)\n", (int)pci->offset,
                (int)pci->length, chunk_len);
        luaT_pushudata(L, pci, nerv_chunk_info_tname);
        lua_setfield(L, -2, "chunk");
        /* stack: obj_table, metadata */
        /* get id */
        lua_getfield(L, -1, "id");
        /* stack: obj_table, metadata, id */
        if (!lua_isstring(L, -1))
            nerv_error(L, "id field in metadata must be a string");
        lua_pushvalue(L, -1);
        /* stack: obj_table, metadata, id, id */
        lua_gettable(L, -4);
        /* stack: obj_table, metadata, id, obj[id] */
        if (!lua_isnil(L, -1))
            nerv_error(L, "conflicting id");
        lua_pop(L, 1);
        /* stack: obj_table, metadata, id */
        lua_pushvalue(L, -2);
        /* stack: obj_table, metadata, id, metadata */
        lua_settable(L, -4);
        /* stack: obj_table, metadata */
        lua_pop(L, 1);
    }
    lua_setfield(L, -2, "metadata");
    lfp = (ChunkFileHandle *)malloc(sizeof(ChunkFileHandle));
    lfp->fp = fp;
    luaT_pushudata(L, lfp, nerv_chunk_file_handle_tname);
    lua_setfield(L, -2, "handle");
    luaT_pushmetatable(L, nerv_chunk_file_tname);
    lua_setmetatable(L, -2);
    return 1;
}

int nerv_chunk_file_new_(lua_State *L, const char *fn, const char *mode) {
    int rd = 1, bin = 0;
    size_t i, len = strlen(mode);
    for (i = 0; i < len; i++)
        switch (mode[i])
        {
            case 'r': rd = 1; break;
            case 'w': rd = 0; break;
            case 'b': bin = 1; break;
        }
    return rd ? nerv_chunk_file_open_read(L, fn) : \
                nerv_chunk_file_open_write(L, fn);
}

int nerv_chunk_file___init(lua_State *L) {
    lua_pushvalue(L, 1);
    return nerv_chunk_file_new_(L, luaL_checkstring(L, 2),
                                    luaL_checkstring(L, 3));
}

int nerv_chunk_file_new(lua_State *L) {
    lua_newtable(L);
    return nerv_chunk_file_new_(L, luaL_checkstring(L, 1),
                                    luaL_checkstring(L, 2));
}

int nerv_chunk_file_write_chunkdata(lua_State *L) {
    ChunkFileHandle *pfh;
    int status;
    off_t start;
    size_t size;
    const char *metadata_str = lua_tolstring(L, 2, NULL);
    lua_getfield(L, 1, "handle");
    pfh = luaT_checkudata(L, -1, nerv_chunk_file_handle_tname);
    start = ftello(pfh->fp);
    write_chunk_header_plain(pfh->fp, 0, &status); /* fill zeros */
    CHECK_WRITE(status);
    write_chunk_metadata(pfh->fp, metadata_str, &status);
    CHECK_WRITE(status);
    lua_pushvalue(L, 3);
    lua_getfield(L, -1, "write");
    if (!lua_isfunction(L, -1))
        nerv_error(L, "\"write\" method must be implemented");
    lua_pushvalue(L, -2);
    lua_pushvalue(L, 4); /* pass handle as parameter to write() */
    lua_call(L, 2, 0); /* let the write() to write */
    lua_pop(L, 1);
    size = ftello(pfh->fp) - start;
    fseeko(pfh->fp, start, SEEK_SET);
    /* write the calced size */
    write_chunk_header_plain(pfh->fp, size, &status);
    CHECK_WRITE(status);
    fseeko(pfh->fp, 0, SEEK_END);
    return 0;
}

int nerv_chunk_file_get_chunkdata(lua_State *L) {
    ChunkFileHandle *pfh;
    ChunkInfo *pci;
    ChunkData *pcd;
    const char *id = luaL_checkstring(L, 2);

    lua_getfield(L, 1, "handle");
    pfh = luaT_checkudata(L, -1, nerv_chunk_file_handle_tname);
    lua_pop(L, 1); /* pop handle */
    lua_getfield(L, 1, "metadata");
    /* now stack: self, k, metadata */
    lua_getfield(L, -1, id);
    /* now stack: self, k, metadata, kth{} */
    if (lua_isnil(L, -1)) /* no chunck with the id */
        return 0;
    lua_getfield(L, -1, "chunk");
    pci = luaT_checkudata(L, -1, nerv_chunk_info_tname);
    if (!(pcd = get_chunk_data(pfh->fp, pci)))
        nerv_error(L, "unexpected end of file");
    luaT_pushudata(L, pcd, nerv_chunk_data_tname);
    return 1;
}

int nerv_chunk_file_handle_destroy(lua_State *L) {
    ChunkFileHandle *pfh = luaT_checkudata(L, 1,
                                nerv_chunk_file_handle_tname);
    fclose(pfh->fp);
    free(pfh);
    return 0;
}

static int nerv_chunk_info_destroy(lua_State *L) {
    ChunkInfo *pci = luaT_checkudata(L, 1, nerv_chunk_info_tname);
    free(pci);
    return 0;
}

static int nerv_chunk_data_destroy(lua_State *L) {
    ChunkData *pcd = luaT_checkudata(L, 1, nerv_chunk_data_tname);
    fclose(pcd->fp);
    free(pcd->data);
    free(pcd);
    return 0;
}

static const luaL_Reg nerv_chunk_file_methods[] = {
    {"get_chunkdata", nerv_chunk_file_get_chunkdata},
    {"_write_chunkdata", nerv_chunk_file_write_chunkdata},
    {"__init", nerv_chunk_file___init},
    {NULL, NULL}
};

void nerv_chunk_file_init(lua_State *L) {
    luaT_newmetatable(L, nerv_chunk_file_tname, NULL,
                        nerv_chunk_file_new,
                        NULL, NULL);
    luaL_register(L, NULL, nerv_chunk_file_methods);
    lua_pop(L, 1);
    luaT_newmetatable(L, nerv_chunk_file_handle_tname, NULL,
                        NULL, nerv_chunk_file_handle_destroy, NULL);
    luaT_newmetatable(L, nerv_chunk_info_tname, NULL,
                        NULL, nerv_chunk_info_destroy, NULL);
    luaT_newmetatable(L, nerv_chunk_data_tname, NULL,
                        NULL, nerv_chunk_data_destroy, NULL);
}

