#ifdef NERV_GENERIC_MMATRIX
#include "matrix.h"
#include "elem_type.h"
#define MATRIX_DATA_FREE(ptr, status) host_matrix_(free)(ptr, status)
#define MATRIX_DATA_ALLOC(dptr, stride, width, height, status) \
                            host_matrix_(alloc)(dptr, stride, width, height, status)
#define NERV_GENERIC_MATRIX
#include "../../common.h"
#include "../../io/chunk_file.h"
#include "string.h"

static void host_matrix_(free)(MATRIX_ELEM *ptr, Status *status) {
    free(ptr);
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

static void host_matrix_(alloc)(MATRIX_ELEM **dptr, size_t *stride,
                                long width, long height, Status *status) {
    if ((*dptr = (MATRIX_ELEM *)malloc(width * height)) == NULL)
        NERV_EXIT_STATUS(status, MAT_INSUF_MEM, 0);
    *stride = width;
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

#include "matrix.c"
Matrix *nerv_matrix_(load)(ChunkData *cdp, Status *status) {
    int i, j;
    long nrow, ncol;
    FILE *fp = cdp->fp;
    Matrix *self;
    if (fscanf(fp, "%ld %ld", &nrow, &ncol) != 2)
        NERV_EXIT_STATUS(status, MAT_INVALID_FORMAT, 0);
    self = nerv_matrix_(create)(nrow, ncol, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    for (i = 0; i < nrow; i++)
    {
        MATRIX_ELEM *row = MATRIX_ROW_PTR(self, i);
        for (j = 0; j < ncol; j++)
            if (fscanf(fp, MATRIX_ELEM_FMT, row + j) != 1)
            {
                free(self);
                NERV_EXIT_STATUS(status, MAT_INVALID_FORMAT, 0);
            }
    }
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return self;
}

void nerv_matrix_(save)(Matrix *self, ChunkFile *cfp, Status *status) {
    int i, j;
    long nrow = self->nrow, ncol = self->ncol;
    FILE *fp = cfp->fp;
    if (fprintf(fp, "%ld %ld\n", nrow, ncol) < 0)
        NERV_EXIT_STATUS(status, MAT_WRITE_ERROR, 0);
    for (i = 0; i < nrow; i++)
    {
        MATRIX_ELEM *row = MATRIX_ROW_PTR(self, i);
        for (j = 0; j < ncol; j++)
            if (fprintf(fp, MATRIX_ELEM_WRITE_FMT " ", row[j]) < 0)
                NERV_EXIT_STATUS(status, MAT_WRITE_ERROR, 0);
        if (fprintf(fp, "\n") < 0)
            NERV_EXIT_STATUS(status, MAT_WRITE_ERROR, 0);
    }
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

void nerv_matrix_(copy_from)(Matrix *a, const Matrix *b,
                            int a_begin, int b_begin, int b_end,
                            Status *status) {
    if (!(0 <= b_begin && b_begin < b_end && b_end <= b->nrow &&
            a_begin + b_end - b_begin <= a->nrow))
        NERV_EXIT_STATUS(status, MAT_INVALID_COPY_INTERVAL, 0);
    if (a->ncol != b->ncol)
        NERV_EXIT_STATUS(status, MAT_MISMATCH_DIM, 0);
    memmove(MATRIX_ROW_PTR(a, a_begin),
            MATRIX_ROW_PTR(b, b_begin),
            sizeof(MATRIX_ELEM) * b->ncol * (b_end - b_begin));
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
}

#endif
