#ifdef MATRIX_USE_FLOAT

#define MATRIX_ELEM float
#define MATRIX_ELEM_PTR(self) ((self)->data.f)

#elif defined(MATRIX_USE_DOUBLE)

#define MATRIX_ELEM double
#define MATRIX_ELEM_PTR(self) ((self)->data.d)

#endif
