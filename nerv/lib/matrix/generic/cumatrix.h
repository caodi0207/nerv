#include "../../common.h"

void nerv_matrix_(add)(Matrix *c, const Matrix *a, const Matrix *b,
                            MATRIX_ELEM alpha, MATRIX_ELEM beta,
                            Status *status);
void nerv_matrix_(mul)(Matrix *c, const Matrix *a, const Matrix *b,
                            MATRIX_ELEM alpha, MATRIX_ELEM beta,
                            int ta, int tb, Status *status);
void nerv_matrix_(sigmoid)(Matrix *a, const Matrix *b, Status *status);
void nerv_matrix_(sigmoid_grad)(Matrix *nerr, const Matrix *err,
                                const Matrix *output, Status *status);

Matrix *nerv_matrix_(softmax)(Matrix *b, const Matrix *a, Status *status);
Matrix *nerv_matrix_(rowsum)(Matrix *a, Status *status);
Matrix *nerv_matrix_(colsum)(Matrix *a, Status *status);
Matrix *nerv_matrix_(colsame)(Matrix *a, const Matrix *ref,
                                Status *status);
Matrix *nerv_matrix_(rowmax)(Matrix *a, Status *status);
void nerv_matrix_(rowmax_idx)(Matrix *a, Matrix **b, Matrix **idx,
                                Status *status);
void nerv_matrix_(add_row)(Matrix *b, const Matrix *a, double beta,
                            Status *status);
void nerv_matrix_(clip)(Matrix *self, double val_1, double val_2, Status *status);
void nerv_matrix_(fill)(Matrix *self, double val, Status *status);
void nerv_matrix_(copy_fromd)(Matrix *a, const Matrix *b,
                            int a_begin, int b_begin, int b_end,
                            Status *status);
void nerv_matrix_(copy_fromh)(Matrix *a, const Matrix *b,
                            int a_begin, int b_begin, int b_end,
                            Status *status);
void nerv_matrix_(copy_toh)(Matrix *a, const Matrix *b,
                            int a_begin, int a_end, int b_begin,
                            Status *status);
Matrix *nerv_matrix_(trans)(Matrix *a, Status *status);
void nerv_matrix_(mul_elem)(Matrix *c, const Matrix *a, const Matrix *b,
                            Status *status);

void nerv_matrix_(log_elem)(Matrix *b, const Matrix *a, Status *status);

Matrix *nerv_matrix_(decompress)(const Matrix *a, int orig_col, Status *status);
void nerv_matrix_(copy_rows_fromh_by_idx)(Matrix *a, const Matrix *b,
                            const Matrix *idx, int b_begin, Status *status);

void nerv_matrix_(expand_frm)(Matrix *a, const Matrix *b,
                            int context, Status *status);
void nerv_matrix_(rearrange_frm)(Matrix *a, const Matrix *b,
                                int step, Status *status);
void nerv_matrix_(scale_rows_by_col)(Matrix *a, const Matrix *b,
                                    Status *status);
void nerv_matrix_(scale_rows_by_row)(Matrix *a, const Matrix *b,
                                    Status *status);
