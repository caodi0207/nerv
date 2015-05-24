#ifndef NERV_LAYER_FILE_H
#define NERV_LAYER_FILE_H

extern const char *nerv_param_file_tname;
extern const char *nerv_param_file_handle_tname;
extern const char *nerv_param_chunk_info_tname;
extern const char *nerv_param_chunk_data_tname;

typedef struct ParamFileHandle {
    FILE *fp;
} ParamFileHandle;

typedef struct ParamChunkInfo {
    off_t offset, length;
} ParamChunkInfo;

typedef struct ParamChunkData {
    FILE *fp;
    char *data;
} ParamChunkData;

#endif
