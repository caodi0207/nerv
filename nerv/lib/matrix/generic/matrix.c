#ifdef NERV_GENERIC_MATRIX
#include "../../common.h"
#include "matrix.h"
/* FIXME: malloc failure detection */

void nerv_matrix_(data_free)(Matrix *self, Status *status) {
    assert(*self->data_ref > 0);
    if (--(*self->data_ref) == 0)
    {
        /* free matrix data */
        MATRIX_DATA_FREE(MATRIX_ELEM_PTR(self), status);
        free(self->data_ref);
        free(self);
    }
    else {
        free(self);
        NERV_SET_STATUS(status, NERV_NORMAL, 0);
    }
}

void nerv_matrix_(data_retain)(Matrix *self) {
    (*self->data_ref)++;
}

Matrix *nerv_matrix_(create)(long nrow, long ncol, Status *status) {
    Matrix *self = (Matrix *)malloc(sizeof(Matrix));
    self->nrow = nrow;
    self->ncol = ncol;
    self->nmax = self->nrow * self->ncol;
    self->dim = 2;
    MATRIX_DATA_ALLOC(&MATRIX_ELEM_PTR(self), &self->stride,
                     sizeof(MATRIX_ELEM) * self->ncol, self->nrow,
                     status);
    if (status->err_code != NERV_NORMAL)
    {
        free(self);
        return NULL;
    }
    self->data_ref = (long *)malloc(sizeof(long));
    *self->data_ref = 0;
    nerv_matrix_(data_retain)(self);
    NERV_SET_STATUS(status, NERV_NORMAL, 0);
    return self;
}

void nerv_matrix_(destroy)(Matrix *self, Status *status) {
    nerv_matrix_(data_free)(self, status);
}

Matrix *nerv_matrix_(getrow)(Matrix *self, int row) {
    Matrix *prow = (Matrix *)malloc(sizeof(Matrix));
    prow->ncol = self->ncol;
    prow->nrow = 1;
    prow->dim = 1;
    prow->stride = self->stride;
    prow->nmax = prow->ncol;
    MATRIX_ELEM_PTR(prow) = MATRIX_ROW_PTR(self, row);
    prow->data_ref = self->data_ref;
    nerv_matrix_(data_retain)(prow);
    return prow;
}
#endif
