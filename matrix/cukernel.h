#ifndef NERV_CUKERNEL_H
#define NERV_CUKERNEL_H
void cuda_sigmoid(const Matrix *a, Matrix *b);
void cuda_rowsum(const Matrix *a, Matrix *b);
#endif
