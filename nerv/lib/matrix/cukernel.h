#ifdef NERV_GENERIC_CUKERNEL
void cudak_(cuda_mul_elem)(const Matrix *a, const Matrix *b, Matrix *c);
void cudak_(cuda_log_elem)(const Matrix *a, Matrix *b);
void cudak_(cuda_sigmoid)(const Matrix *a, Matrix *b);
void cudak_(cuda_sigmoid_grad)(const Matrix *output, const Matrix *err, Matrix *nerr);
void cudak_(cuda_rowsum)(const Matrix *a, Matrix *b);
void cudak_(cuda_rowmax)(const Matrix *a, Matrix *b);
void cudak_(cuda_rowmax_idx)(const Matrix *a, Matrix *b, Matrix *idx);
void cudak_(cuda_colsum)(const Matrix *a, Matrix *b);
void cudak_(cuda_colsame)(const Matrix *a, const Matrix *ref, Matrix *b);
void cudak_(cuda_softmax_denominator)(const Matrix *a, const Matrix *max, Matrix *b);
void cudak_(cuda_softmax_final)(const Matrix *a, const Matrix *max, const Matrix *deno, Matrix *b);
void cudak_(cuda_add_row)(const Matrix *a, Matrix *b, double beta);
void cudak_(cuda_fill)(Matrix *a, double val);
void cudak_(cuda_clip)(Matrix *a, double val_1, double val_2);
void cudak_(cuda_expand_frm)(const Matrix *a, Matrix *b, int context);
void cudak_(cuda_rearrange_frm)(const Matrix *a, Matrix *b, int step);
void cudak_(cuda_scale_rows_by_row)(const Matrix *a, Matrix *b);
void cudak_(cuda_scale_rows_by_col)(const Matrix *a, Matrix *b);
void cudak_(cuda_decompress)(const Matrix *a, Matrix *b);
#endif
