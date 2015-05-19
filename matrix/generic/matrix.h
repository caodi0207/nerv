#ifndef NERV_GENERIC_MATRIX_H
#define NERV_GENERIC_MATRIX_H

#include <stddef.h>
typedef struct Matrix {
    size_t stride;              /* size of a row */
    long ncol, nrow, nmax;    /* dimension of the matrix */
    union {
        float *f;
        double *d;
    } data;                   /* pointer to actual storage */
    long *data_ref;
} Matrix;

#endif
