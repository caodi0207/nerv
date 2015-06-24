#ifndef NERV_GENERIC_MATRIX_H
#define NERV_GENERIC_MATRIX_H

#include <stddef.h>

typedef struct Matrix {
    size_t stride;              /* size of a row */
    long ncol, nrow, nmax;    /* dimension of the matrix */
    union {
        float *f;
        double *d;
        long *i;
    } data;                   /* pointer to actual storage */
    long *data_ref;
} Matrix;

#define MATRIX_ROW_PTR(self, row) \
    (MATRIX_ELEM *)((char *)MATRIX_ELEM_PTR(self) + (row) * (self)->stride)
#endif
