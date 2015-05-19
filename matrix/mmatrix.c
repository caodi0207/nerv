#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_FLOAT
#define host_matrix_(NAME) host_matrix_float_ ## NAME
#define nerv_matrix_(NAME) nerv_matrix_float_host_ ## NAME
const char *nerv_matrix_(tname) = "nerv.FloatMMatrix";
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_DOUBLE
#define host_matrix_(NAME) host_matrix_double_ ## NAME
#define nerv_matrix_(NAME) nerv_matrix_double_host_ ## NAME
const char *nerv_matrix_(tname) = "nerv.DoubleMMatrix";
#include "generic/mmatrix.c"
