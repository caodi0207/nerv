#The Nerv Matrix Package#
Part of the [Nerv](../README.md) toolkit.

##Description##
###Underlying structure###
In the begining is could be useful to know something about the underlying structure of a __Nerv__ matrix. Please keep in mind that matrice in __Nerv__ is row-major.  
Every matrix object is a encapsulation of a C struct that describes the attributes of this matrix.  
```
typedef struct Matrix {
    size_t stride;              /* size of a row */
    long ncol, nrow, nmax;    /* dimension of the matrix, nmax is simply nrow * ncol */
    union {
        float *f;
        double *d;
        long *i;
    } data;                   /* pointer to actual storage */
    long *data_ref;
} Matrix;
```
It is worth mentioning that that `data_ref` is a counter which counts the number of references to its memory space, mind that it will also be increased when a row of the matrix is referenced(`col = m[2]`). A __Nerv__ matrix will deallocate its space when this counter is decreased to zero.   
Also note that all assigning operation in __Nerv__ is reference copy, you can use `copy_tod` or `copy_toh` method to copy value. Also, row assigning operations like `m1[2]=m2[3]` is forbidden in __Nerv__.

###Class hierarchy###
The class hierarchy of the matrix classes can be clearly observed in `matrix/init.c`.   
First there is a abstract base class __Nerv.Matrix__, which is inherited by __Nerv.CuMatrix__ and __Nerv.MMatrix__(also abstract).  
Finally, there is __Nerv.CuMatrixFloat__, __Nerv.CuMatrixDouble__, inheriting __Nerv.CuMatrix__, and __Nerv.MMatrixFloat__, __Nerv.MMatrixDouble__, inheriting __Nerv.MMatrix__.

##Methods##
Mind that usually a matrix object can only do calculation with matrix of its own type(a __Nerv.CuMatrixFloat__ matrix can only do add operation with a __Nerv.CuMatrixFloat__).  
In the methods description below, __Matrix__ could be __Nerv.CuMatrixFloat__, __Nerv.CuMatrixDouble__, __Nerv.MMatrixFloat__ or __Nerv.MMatrixDouble__. __Element_type__ could be `float` or `double`, respectively.    
* __Matrix = Matrix(int nrow, int ncol)__  
Returns a __Matrix__ object of `nrow` rows and `ncol` columns.
* __Element_type = Matrix.get_elem(Matrix self, int index)__  
Returns the element value at the specific index(treating the matrix as a vector). The index should be less than `nmax` of the matrix.
* __void Matrix.set_elem(Matrix self, int index, Element_type value)__  
Set the value at `index` to be `value`.
* __int Matrix.ncol(Matrix self)__  
Get `ncol`, the number of columns.
* __int Matrix.nrow(Matrix self)__  
Get `nrow`, the number of rows.
* __int Matrix.get_dataref_value(Matrix self)__  
Returns the value(not a pointer) of space the `data_ref` pointer pointed to. This function is mainly for debugging.  
* __Matrix/Element\_type, boolean Matrix.\_\_index\_\_(Matrix self, int index)__  
If the matrix has more than one row, will return the row at `index` as a __Matrix__ . Otherwise it will return the value at `index`.
* __void Matrix.\_\_newindex\_\_(Matrix self, int index, Element_type value)__
Set the element at `index` to be `value`.
---
