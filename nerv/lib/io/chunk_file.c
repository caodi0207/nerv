#include "../common.h"
#include "chunk_file.h"
#include <stdlib.h>
#include <string.h>
#define PARAM_HEADER_SIZE 16

static size_t read_chunk_header_plain(FILE *fp, Status *status) {
    static char buff[PARAM_HEADER_SIZE];
    int i;
    size_t size = 0;
    if (fread(buff, 1, PARAM_HEADER_SIZE, fp) != PARAM_HEADER_SIZE)
    {
        if (feof(fp))
            NERV_SET_STATUS(status, CF_END_OF_FILE, 0);
        else
        {
            NERV_SET_STATUS(status, CF_INVALID_FORMAT, 0);
            return 0;
        }
    }
    else
        NERV_SET_STATUS(status, NERV_NORMAL, 0);
    for (i = 0; i < PARAM_HEADER_SIZE; i++)
        if (isdigit(buff[i]))
            size = size * 10 + buff[i] - '0';
/*    fprintf(stderr, "header: %lu\n", size); */
    return size;
}

static void write_chunk_header_plain(FILE *fp, size_t size, Status *status) {
    static char buff[PARAM_HEADER_SIZE];
    int i;
    for (i = PARAM_HEADER_SIZE - 3; i > 0; i--, size /= 10)
        buff[i] = size % 10 + '0';
    if (size)
        NERV_EXIT_STATUS(status, CF_SECTION_OVERFLOW, 0);
    buff[0] = '[';
    buff[PARAM_HEADER_SIZE - 2] = ']';
    buff[PARAM_HEADER_SIZE - 1] = '\n';
    if (fwrite(buff, 1, PARAM_HEADER_SIZE, fp) != PARAM_HEADER_SIZE)
        NERV_EXIT_STATUS(status, CF_WRITE_ERROR, 0);
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

static ChunkData *get_chunk_data(FILE *fp, ChunkInfo *info) {
    ChunkData *cdp = (ChunkData *)malloc(sizeof(ChunkData));
    cdp->data = (char *)malloc(info->length);
    cdp->fp = fmemopen(cdp->data, info->length, "r");
    assert(fseeko(fp, info->offset, SEEK_SET) == 0);
    if (fread(cdp->data, 1, info->length, fp) != (size_t)info->length)
        return NULL;
    return cdp;
}

static const char *read_chunk_metadata(FILE *fp, const char *fn,
                                        Status *status) {
#define LINEBUFF_SIZE 1024
#define LUA_RETURN "return "
#define LUA_RETURN_LEN (sizeof(LUA_RETURN) - 1)
    static char buff[LUA_RETURN_LEN + LINEBUFF_SIZE] = LUA_RETURN;
    NERV_SET_STATUS(status, (fgets(buff + LUA_RETURN_LEN,
                                LINEBUFF_SIZE, fp) == (buff + LUA_RETURN_LEN) ? \
                                NERV_NORMAL : CF_INVALID_FORMAT), 0);
    fprintf(stderr, "metadata: %s\n", buff);
    return buff;
}

static void write_chunk_metadata(FILE *fp, const char *mdstr,
                                Status *status) {
    size_t size = strlen(mdstr);
    if (fwrite(mdstr, 1, size, fp) != size ||
        fprintf(fp, "\n") < 0)
        NERV_EXIT_STATUS(status, CF_WRITE_ERROR, 0);
    /* fprintf(stderr, "metadata: %s\n", metadata_str); */
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

static ChunkFile *open_write(const char *fn, Status *status) {
    ChunkFile *cfp;
    FILE *fp = fopen(fn, "w");

    if (!fp)
    {
        NERV_SET_STATUS(status, CF_ERR_OPEN_FILE, 0);
        return NULL;
    }
    cfp = (ChunkFile *)malloc(sizeof(ChunkFile));
    cfp->fp = fp;
    cfp->info = NULL;
    cfp->status = CF_WRITE;
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return cfp;
}

static ChunkFile *open_read(const char *fn, Status *status) {
    size_t chunk_len;
    off_t offset;
    int i;
    const char *mdstr;
    ChunkFile *cfp;
    ChunkInfo *head = NULL;
    FILE *fp = fopen(fn, "r");

    if (!fp)
    {
        NERV_SET_STATUS(status, CF_ERR_OPEN_FILE, 0);
        return NULL;
    }
    cfp = (ChunkFile *)malloc(sizeof(ChunkFile));
    offset = ftello(fp);
    /* fprintf(stderr, "%d\n", (int)offset); */
    for (i = 0;; offset += chunk_len, i++)
    {
        ChunkInfo *cip;
        fprintf(stderr, "reading chunk %d from %d\n", i, (int)offset);
        /* skip to the begining of chunk i */
        if (fseeko(fp, offset, SEEK_SET) != 0)
        {
            NERV_SET_STATUS(status, CF_INVALID_FORMAT, 0);
            return NULL;
        }
        /* read header */
        chunk_len = read_chunk_header_plain(fp, status);
        if (status->err_code == CF_END_OF_FILE) break;
        if (status->err_code != NERV_NORMAL)
            return NULL;
        cip = (ChunkInfo *)malloc(sizeof(ChunkInfo));
        /* read metadata */
        mdstr = read_chunk_metadata(fp, fn, status);
        if (status->err_code != NERV_NORMAL)
            return NULL;
        cip->metadata = strdup(mdstr);
        cip->offset = ftello(fp);
        cip->length = chunk_len - (cip->offset - offset);
        /* fprintf(stderr, "%d + %d (skip %lu)\n", (int)cip->offset,
                (int)cip->length, chunk_len); */
        cip->next = head;
        head = cip;
    }
    cfp->fp = fp;
    cfp->info = head;
    cfp->status = CF_READ;
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return cfp;
}

ChunkFile *nerv_chunk_file_create(const char *fn, const char *mode,
                                Status *status) {
    int rd = 1, bin = 0;
    size_t i, len = strlen(mode);
    for (i = 0; i < len; i++)
        switch (mode[i])
        {
            case 'r': rd = 1; break;
            case 'w': rd = 0; break;
            case 'b': bin = 1; break;
        }
    return rd ? open_read(fn, status) : \
                open_write(fn, status);
}

void nerv_chunk_file_write_chunkdata(ChunkFile *cfp, const char *mdstr,
                                    ChunkDataWriter_t writer, void *writer_arg,
                                    Status *status) {
    off_t start;
    size_t size;
    if (cfp->status != CF_WRITE)
        NERV_EXIT_STATUS(status, CF_INVALID_OP, 0);
    start = ftello(cfp->fp);
    write_chunk_header_plain(cfp->fp, 0, status); /* fill zeros */
    if (status->err_code != NERV_NORMAL)
        return;
    write_chunk_metadata(cfp->fp, mdstr, status);
    if (status->err_code != NERV_NORMAL)
        return;
    writer(writer_arg);
    size = ftello(cfp->fp) - start;
    fseeko(cfp->fp, start, SEEK_SET);
    /* write the calced size */
    write_chunk_header_plain(cfp->fp, size, status);
    if (status->err_code != NERV_NORMAL)
        return;
    fseeko(cfp->fp, 0, SEEK_END);
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

ChunkData *nerv_chunk_file_get_chunkdata(ChunkFile *cfp, ChunkInfo *cip,
                                        Status *status) {
    ChunkData *cdp;
    if (cfp->status != CF_READ)
    {
        NERV_SET_STATUS(status, CF_INVALID_OP, 0);
        return NULL;
    }
    if (!(cdp = get_chunk_data(cfp->fp, cip)))
    {
        NERV_SET_STATUS(status, CF_END_OF_FILE, 0);
        return NULL;
    }
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return cdp;
}

void nerv_chunk_file_close(ChunkFile *cfp) {
    if (cfp->status != CF_CLOSED)
        fclose(cfp->fp);
    cfp->status = CF_CLOSED;
}

void nerv_chunk_file_destroy(ChunkFile *cfp) {
    ChunkInfo *i, *ni;
    if (cfp->info)
    {
        for (i = cfp->info; i; i = ni)
        {
            ni = i->next;
            free(i->metadata);
            free(i);
        }
    }
    if (cfp->status != CF_CLOSED) fclose(cfp->fp);
    free(cfp);
}

void nerv_chunk_data_destroy(ChunkData *cdp) {
    fclose(cdp->fp);
    free(cdp->data);
    free(cdp);
}
