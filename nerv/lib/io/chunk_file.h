#ifndef NERV_CHUNK_FILE_H
#define NERV_CHUNK_FILE_H
#include "../common.h"

typedef struct ChunkInfo {
    struct ChunkInfo *next;
    char *metadata;
    off_t offset, length;
} ChunkInfo;

typedef struct ChunkFile {
    FILE *fp;
    ChunkInfo *info;
    enum {
        CF_READ,
        CF_WRITE,
        CF_CLOSED
    } status;
} ChunkFile;

typedef struct ChunkData {
    FILE *fp;
    char *data;
} ChunkData;

typedef void (*ChunkDataWriter_t)(void *);
ChunkFile *nerv_chunk_file_create(const char *fn, const char *mode,
                                Status *status);
void nerv_chunk_file_write_chunkdata(ChunkFile *cfp, const char *mdstr,
                                    ChunkDataWriter_t writer, void *writer_arg,
                                    Status *status);
ChunkData *nerv_chunk_file_get_chunkdata(ChunkFile *cfp, ChunkInfo *cip,
                                        Status *status);
void nerv_chunk_file_close(ChunkFile *cfp);
void nerv_chunk_file_destroy(ChunkFile *cfp);
void nerv_chunk_data_destroy(ChunkData *cdp);
#endif
