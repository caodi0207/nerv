#define NERV_GENERIC_CUKERNEL

#define cudak_(NAME) cudak_float_ ## NAME
#define MATRIX_USE_FLOAT
#include "generic/elem_type.h"
#include "generic/cukernel.cu"
#undef cudak_
#undef MATRIX_USE_FLOAT
#undef MATRIX_ELEM
#undef MATRIX_ELEM_PTR

#define cudak_(NAME) cudak_double_ ## NAME
#define MATRIX_USE_DOUBLE
#include "generic/elem_type.h"
#include "generic/cukernel.cu"
