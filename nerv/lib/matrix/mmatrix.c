#define NERV_GENERIC_MMATRIX
#include <stdlib.h>
#include "../common.h"

#define MATRIX_USE_FLOAT
#define host_matrix_(NAME) host_matrix_float_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_float_##NAME
#include "generic/matrix.h"
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT
#undef MATRIX_ELEM_WRITE_FMT

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_DOUBLE
#define host_matrix_(NAME) host_matrix_double_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_double_##NAME
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_DOUBLE
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT
#undef MATRIX_ELEM_WRITE_FMT

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_INT
#define host_matrix_(NAME) host_matrix_int_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_int_##NAME
#include "generic/mmatrix.c"

Matrix *nerv_matrix_(perm_gen)(int ncol, Status *status) {
    int i;
    Matrix *self = nerv_matrix_(create)(1, ncol, status);
    if (status->err_code != NERV_NORMAL)
        return NULL;
    long *prow = self->data.i;
    for (i = 0; i < ncol; i++)
        prow[i] = i;
    for (i = ncol - 1; i >= 0; i--)
    {
        size_t j = rand() % (i + 1);
        long tmp = prow[i];
        prow[i] = prow[j];
        prow[j] = tmp;
    }
    return self;
}
