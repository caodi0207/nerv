#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_FLOAT
#define host_matrix_(NAME) host_matrix_float_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_float_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixFloat";
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_DOUBLE
#define host_matrix_(NAME) host_matrix_double_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_double_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixDouble";
#include "generic/mmatrix.c"
#undef nerv_matrix_
#undef host_matrix_
#undef MATRIX_USE_DOUBLE
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR
#undef MATRIX_ELEM_FMT

#define NERV_GENERIC_MMATRIX
#define MATRIX_USE_INT
#define host_matrix_(NAME) host_matrix_int_##NAME
#define nerv_matrix_(NAME) nerv_matrix_host_int_##NAME
const char *nerv_matrix_(tname) = "nerv.MMatrixInt";
#include "generic/mmatrix.c"

