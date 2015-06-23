#include "../../common.h"
#include "chunk_file.h"
#include <stdlib.h>
#include <string.h>
#define PARAM_HEADER_SIZE 16

static size_t read_chunk_header_plain(FILE *fp, int *status) {
    static char buff[PARAM_HEADER_SIZE];
    int i;
    size_t size = 0;
    if (fread(buff, 1, PARAM_HEADER_SIZE, fp) != PARAM_HEADER_SIZE)
    {
        if (feof(fp)) *status = CF_END_OF_FILE;
        else
        {
            *status = CF_INVALID_FORMAT;
            return 0;
        }
    }
    else *status = CF_NORMAL;
    for (i = 0; i < PARAM_HEADER_SIZE; i++)
        if (isdigit(buff[i]))
            size = size * 10 + buff[i] - '0';
/*    fprintf(stderr, "header: %lu\n", size); */
    return size;
}

static void write_chunk_header_plain(FILE *fp, size_t size, int *status) {
    static char buff[PARAM_HEADER_SIZE];
    int i;
    for (i = PARAM_HEADER_SIZE - 3; i > 0; i--, size /= 10)
        buff[i] = size % 10 + '0';
    if (size)
    {
        *status = CF_SECTION_OVERFLOW;
        return;
    }
    buff[0] = '[';
    buff[PARAM_HEADER_SIZE - 2] = ']';
    buff[PARAM_HEADER_SIZE - 1] = '\n';
    if (fwrite(buff, 1, PARAM_HEADER_SIZE, fp) != PARAM_HEADER_SIZE)
    {
        *status = CF_WRITE_ERROR;
        return;
    }
    *status = CF_NORMAL;
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

static const char *read_chunk_metadata(FILE *fp, const char *fn, int *status) {
#define LINEBUFF_SIZE 1024
#define LUA_RETURN "return "
#define LUA_RETURN_LEN (sizeof(LUA_RETURN) - 1)
    static char buff[LUA_RETURN_LEN + LINEBUFF_SIZE] = LUA_RETURN;
    *status = fgets(buff + LUA_RETURN_LEN,
                    LINEBUFF_SIZE, fp) == (buff + LUA_RETURN_LEN) ? \
                     CF_NORMAL : CF_INVALID_FORMAT;
    fprintf(stderr, "metadata: %s\n", buff);
    return buff;
}

static void write_chunk_metadata(FILE *fp, const char *mdstr, int *status) {
    size_t size = strlen(mdstr);
    if (fwrite(mdstr, 1, size, fp) != size ||
        fprintf(fp, "\n") < 0)
    {
        *status = CF_WRITE_ERROR;
        return;
    }
    /* fprintf(stderr, "metadata: %s\n", metadata_str); */
    *status = CF_NORMAL;
}

static ChunkFile *open_write(const char *fn, int *status) {
    ChunkFile *cfp;
    FILE *fp = fopen(fn, "w");

    if (!fp)
    {
        *status = CF_ERR_OPEN_FILE;
        return NULL;
    }
    cfp = (ChunkFile *)malloc(sizeof(ChunkFile));
    cfp->fp = fp;
    cfp->status = CF_WRITE;
    *status = CF_NORMAL;
    return cfp;
}

static ChunkFile *open_read(const char *fn, int *status) {
    size_t chunk_len;
    off_t offset;
    int i;
    const char *mdstr;
    ChunkFile *cfp;
    ChunkInfo *head = NULL;
    FILE *fp = fopen(fn, "r");

    if (!fp)
    {
        *status = CF_ERR_OPEN_FILE;
        return NULL;
    }
    cfp = (ChunkFile *)malloc(sizeof(ChunkFile));
    cfp->fp = fp;
    cfp->status = CF_READ;
    offset = ftello(fp);
    /* fprintf(stderr, "%d\n", (int)offset); */
    for (i = 0;; offset += chunk_len, i++)
    {
        ChunkInfo *cip;
        fprintf(stderr, "reading chunk %d from %d\n", i, (int)offset);
        /* skip to the begining of chunk i */
        if (fseeko(fp, offset, SEEK_SET) != 0)
        {
            *status = CF_INVALID_FORMAT;
            return NULL;
        }
        /* read header */
        chunk_len = read_chunk_header_plain(fp, status);
        if (*status == CF_END_OF_FILE) break;
        if (*status != CF_NORMAL)
            return NULL;
        cip = (ChunkInfo *)malloc(sizeof(ChunkInfo));
        /* read metadata */
        mdstr = read_chunk_metadata(fp, fn, status);
        if (*status != CF_NORMAL)
            return NULL;
        cip->metadata = strdup(mdstr);
        cip->offset = ftello(fp);
        cip->length = chunk_len - (cip->offset - offset);
        /* fprintf(stderr, "%d + %d (skip %lu)\n", (int)cip->offset,
                (int)cip->length, chunk_len); */
        cip->next = head;
        head = cip;
    }
    *status = CF_NORMAL;
    cfp->info = head;
    return cfp;
}

ChunkFile *nerv_chunk_file_create(const char *fn, const char *mode, int *status) {
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

int nerv_chunk_file_write_chunkdata(ChunkFile *cfp, const char *mdstr,
                                    ChunkDataWriter_t writer, void *writer_arg) {
    int status;
    off_t start;
    size_t size;
    if (cfp->status != CF_WRITE)
        return CF_INVALID_OP;
    start = ftello(cfp->fp);
    write_chunk_header_plain(cfp->fp, 0, &status); /* fill zeros */
    if (status != CF_NORMAL) return status;
    write_chunk_metadata(cfp->fp, mdstr, &status);
    if (status != CF_NORMAL) return status;
    writer(writer_arg);
    size = ftello(cfp->fp) - start;
    fseeko(cfp->fp, start, SEEK_SET);
    /* write the calced size */
    write_chunk_header_plain(cfp->fp, size, &status);
    if (status != CF_NORMAL) return status;
    fseeko(cfp->fp, 0, SEEK_END);
    return CF_NORMAL;
}

ChunkData *nerv_chunk_file_get_chunkdata(ChunkFile *cfp, ChunkInfo *cip, int *status) {
    ChunkData *cdp;
    if (cfp->status != CF_READ)
    {
        *status = CF_INVALID_OP;
        return NULL;
    }
    if (!(cdp = get_chunk_data(cfp->fp, cip)))
    {
        *status = CF_END_OF_FILE;
        return NULL;
    }
    *status = CF_NORMAL;
    return cdp;
}

void nerv_chunk_file_close(ChunkFile *cfp) {
    if (cfp->status != CF_CLOSED)
        fclose(cfp->fp);
    cfp->status = CF_CLOSED;
}

void nerv_chunk_file_destroy(ChunkFile *cfp) {
    ChunkInfo *i, *ni;
    if (cfp->status != CF_CLOSED) fclose(cfp->fp);
    for (i = cfp->info; i; i = ni)
    {
        ni = i->next;
        free(i->metadata);
        free(i);
    }
    free(cfp);
}

void nerv_chunk_data_destroy(ChunkData *cdp) {
    fclose(cdp->fp);
    free(cdp->data);
    free(cdp);
}

const char *nerv_chunk_file_errstr(int status) {
    switch (status)
    {
        case CF_INVALID_FORMAT: return "invalid format";
        case CF_END_OF_FILE: return "unexpected end of file";
        case CF_SECTION_OVERFLOW: return "section overflow";
        case CF_WRITE_ERROR: return "error while writing";
        case CF_ERR_OPEN_FILE: return "error while opening file";
        case CF_INVALID_OP: return "invalid operation";
        default: return "unknown";
    }
    return NULL;
}
