#ifdef MATRIX_USE_FLOAT

#define MATRIX_ELEM float
#define MATRIX_ELEM_FMT "%f"
#define MATRIX_ELEM_WRITE_FMT "%.8f"
#define MATRIX_ELEM_PTR(self) ((self)->data.f)

#elif defined(MATRIX_USE_DOUBLE)

#define MATRIX_ELEM double
#define MATRIX_ELEM_FMT "%lf"
#define MATRIX_ELEM_WRITE_FMT "%.8lf"
#define MATRIX_ELEM_PTR(self) ((self)->data.d)

#elif defined(MATRIX_USE_INT)

#define MATRIX_ELEM long
#define MATRIX_ELEM_FMT "%ld"
#define MATRIX_ELEM_WRITE_FMT "%ld"
#define MATRIX_ELEM_PTR(self) ((self)->data.i)

#endif
