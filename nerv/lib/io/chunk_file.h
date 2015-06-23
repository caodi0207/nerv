#ifndef NERV_CHUNK_FILE_H
#define NERV_CHUNK_FILE_H
#include "../../common.h"
enum {
    CF_NORMAL,
    CF_INVALID_FORMAT,
    CF_END_OF_FILE,
    CF_SECTION_OVERFLOW,
    CF_WRITE_ERROR,
    CF_ERR_OPEN_FILE,
    CF_INVALID_OP,
    CF_READ,
    CF_WRITE,
    CF_CLOSED
};

typedef struct ChunkInfo {
    struct ChunkInfo *next;
    char *metadata;
    off_t offset, length;
} ChunkInfo;

typedef struct ChunkFile {
    FILE *fp;
    ChunkInfo *info;
    int status;
} ChunkFile;

typedef struct ChunkData {
    FILE *fp;
    char *data;
} ChunkData;

typedef void (*ChunkDataWriter_t)(void *);
ChunkFile *nerv_chunk_file_create(const char *fn, const char *mode, int *status);
int nerv_chunk_file_write_chunkdata(ChunkFile *cfp, const char *mdstr,
                                    ChunkDataWriter_t writer, void *writer_arg);
ChunkData *nerv_chunk_file_get_chunkdata(ChunkFile *cfp, ChunkInfo *cip, int *status);
void nerv_chunk_file_close(ChunkFile *cfp);
void nerv_chunk_file_destroy(ChunkFile *cfp);
void nerv_chunk_data_destroy(ChunkData *cdp);
const char *nerv_chunk_file_errstr(int status);
#endif
