#define NERV_GENERIC_CUMATRIX

#define MATRIX_USE_FLOAT
#define cuda_matrix_(NAME) cuda_matrix_float_ ## NAME
#define nerv_matrix_(NAME) nerv_matrix_float_cuda_ ## NAME
#define cudak_(NAME) cudak_float_ ## NAME
#define NERV_CUBLAS_(NAME) cublasS##NAME
const char *nerv_matrix_(tname) = "nerv.FloatCuMatrix";
#include "generic/cumatrix.c"
#undef NERV_CUBLAS_
#undef cudak_
#undef nerv_matrix_
#undef cuda_matrix_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR

#define MATRIX_USE_DOUBLE
#define cuda_matrix_(NAME) cuda_matrix_double_ ## NAME
#define nerv_matrix_(NAME) nerv_matrix_double_cuda_ ## NAME
#define cudak_(NAME) cudak_double_ ## NAME
#define NERV_CUBLAS_(NAME) cublasD##NAME
const char *nerv_matrix_(tname) = "nerv.DoubleCuMatrix";
#include "generic/cumatrix.c"
