#define NERV_GENERIC_CUMATRIX

#define MATRIX_USE_FLOAT
#define cuda_matrix_(NAME) cuda_matrix_float_##NAME
#define nerv_matrix_(NAME) nerv_matrix_cuda_float_##NAME
#define cudak_(NAME) cudak_float_ ## NAME
#define NERV_CUBLAS_(NAME) cublasS##NAME
const char *nerv_matrix_(tname) = "nerv.CuMatrixFloat";
#include "generic/cumatrix.c"
#undef NERV_CUBLAS_
#undef cudak_
#undef nerv_matrix_
#undef cuda_matrix_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT

#define MATRIX_USE_DOUBLE
#define cuda_matrix_(NAME) cuda_matrix_double_##NAME
#define nerv_matrix_(NAME) nerv_matrix_cuda_double_##NAME
#define cudak_(NAME) cudak_double_ ## NAME
#define NERV_CUBLAS_(NAME) cublasD##NAME
const char *nerv_matrix_(tname) = "nerv.CuMatrixDouble";
#include "generic/cumatrix.c"
