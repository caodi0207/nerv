#ifndef NERV_CUKERNEL_H
#define NERV_CUKERNEL_H
void cuda_sigmoid(const Matrix *a, Matrix *b);
void cuda_colsum(const Matrix *a, Matrix *b);
void cuda_colmax(const Matrix *a, Matrix *b);
void cuda_softmax_denominator(const Matrix *a, const Matrix *max, Matrix *b);
void cuda_softmax_final(const Matrix *a, const Matrix *max, const Matrix *deno, Matrix *b);
#endif
