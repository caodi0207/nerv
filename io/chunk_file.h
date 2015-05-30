#ifndef NERV_LAYER_FILE_H
#define NERV_LAYER_FILE_H

extern const char *nerv_chunk_file_tname;
extern const char *nerv_chunk_file_handle_tname;
extern const char *nerv_chunk_info_tname;
extern const char *nerv_chunk_data_tname;

typedef struct ChunkFileHandle {
    FILE *fp;
} ChunkFileHandle;

typedef struct ChunkInfo {
    off_t offset, length;
} ChunkInfo;

typedef struct ChunkData {
    FILE *fp;
    char *data;
} ChunkData;

#endif
