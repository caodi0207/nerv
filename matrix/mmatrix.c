#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_FLOAT
#define host_matrix_(NAME) host_matrix_float_ ## NAME
#define nerv_matrix_(NAME) nerv_matrix_float_host_ ## NAME
#include "generic/mmatrix.c"
