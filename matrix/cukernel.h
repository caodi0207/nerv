#ifdef NERV_GENERIC_CUKERNEL
void cudak_(cuda_sigmoid)(const Matrix *a, Matrix *b);
void cudak_(cuda_rowsum)(const Matrix *a, Matrix *b);
void cudak_(cuda_rowmax)(const Matrix *a, Matrix *b);
void cudak_(cuda_colsum)(const Matrix *a, Matrix *b);
void cudak_(cuda_softmax_denominator)(const Matrix *a, const Matrix *max, Matrix *b);
void cudak_(cuda_softmax_final)(const Matrix *a, const Matrix *max, const Matrix *deno, Matrix *b);
#endif
