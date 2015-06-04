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
Finally, there is __Nerv.CuMatrixFloat__, __Nerv.CuMatrixDouble__, inheriting __Nerv.CuMatrix__, and __Nerv.MMatrixFloat__, __Nerv.MMatrixDouble__, __Nerv.MMatrixInt__ , inheriting __Nerv.MMatrix__.

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
* __Matrix Matrix.create(Matrix a)__  
Return a new __Matrix__ of `a`'s size(of the same number of rows and columns).
* __Matrix Matrix.colsum(Matrix self)__  
Return a new __Matrix__ of size (1,`self.ncol`), which stores the sum of all columns of __Matrix__ `self`.
* __Matrix Matrix.rowsum(Matrix self)__  
Return a new __Matrix__ of size (`self.nrow`,1), which stores the sum of all rows of __Matrix__ `self`.
* __Matrix Matrix.rowmax(Matrix self)__  
Return a new __Matrix__ of size (`self.nrow`,1), which stores the max value of all rows of __Matrix__ `self`.
* __Matrix Matrix.trans(Matrix self)__  
Return a new __Matrix__ of size (`self.ncol`,`self.nrow`), which stores the transpose of __Matrix__ `self`.
* __void Matrix.copy_fromh(Matrix self, MMatrix a)__  
Copy the content of a __MMatrix__ `a` to __Matrix__ `self`, they should be of the same size.
* __void Matrix.copy_fromd(Matrix self, CuMatrix a)__  
Copy the content of a __CuMatrix__ `a` to __Matrix__ `self`, they should be of the same size.
* __void Matrix.copy_toh(Matrix self, MMatrix a)__  
Copy the content of the __Matrix__ `self` to a __MMatrix__ `a`.
* __void Matrix.copy_tod(Matrix self, CuMatrix a)__  
Copy the content of the __Matrix__ `self` to a __CuMatrix__ `a`.
* __void Matrix.add(Matrix self, Matrix ma, Matrix mb, Element_type alpha, Element_type beta)__  
It sets the content of __Matrix__ `self` to be `alpha * ma + beta * mb`.__Matrix__ `ma,mb,self` should be of the same size.
* __void Matrix.mul(Matrix self, Matrix ma, Matrix mb, Element_type alpha, Element_type beta, [string ta, string tb])__  
It sets the content of __Matrix__ `self` to be `beta * self + alpha * ma * mb`. `ta` and `tb` is optional, if `ta` is 'T', then `ma` will be transposed, also if `tb` is 'T', `mb` will be transposed.
* __void Matrix.add_row(Matrix self, Matrix va, Element_type beta)__  
Add `beta * va` to every row of __Matrix__ `self`.
* __void Matrix.fill(Matrix self, Element_type value)__  
Fill the content of __Matrix__ `self` to be `value`.
* __void Matrix.sigmoid(Matrix self, Matrix ma)__  
Set the element of __Matrix__ `self` to be elementwise-sigmoid of `ma`.
* __void Matrix.sigmoid_grad(Matrix self, Matrix err, Matrix output)__  
Set the element of __Matrix__ `self`, to be `self[i][j]=err[i][j]*output[i][j]*(1-output[i][j])`. This function is used to propagate sigmoid layer error.
* __void Matrix.softmax(Matrix self, Matrix a)__  
Calculate a row-by-row softmax of __Matrix__ `a` and save the result in `self`.
* __void Matrix.mul_elem(Matrix self, Matrix ma, Matrix mb)__  
Calculate element-wise multiplication of __Matrix__ `ma` and `mb`, store the result in `self`.
* __void Matrix.log_elem(Matrix self, Matrix ma)__  
Calculate element-wise log of __Matrix__ `ma`, store the result in `self`.
* __void Matrix.copy_rows_fromh_by_idx(Matrix self, MMatrix ma, MMatrixInt idx)__  
`idx` should be a row vector. This function copy the rows of `ma` to `self` according to `idx`, in other words, it assigns `ma[idx[i]]` to `self[i]`.
* __void Matrix.expand_frm(Matrix self, Matrix a, int context)__  
Treating each row of `a` as speech feature, and do a feature expansion. The `self` should of size `(a.nrow, a.ncol * (context * 2 + 1))`. `self[i]` will be `(a[i-context] a[i-context+1] ... a[i] a[i+1] a[i+context])`. `a[0]` and `a[nrow]` will be copied to extend the index range.
* __void Matrix.rearrange_frm(Matrix self, Matrix a, int step)__  
Rearrange `a` according to its feature dimension. The `step` is the length of context. So, `self[i][j]` will be assigned `a[i][j / step + (j % step) * (a.ncol / step)]`. `a` and `self` should be of the same size and `step` should be divisible by `a.ncol`.
* __void Matrix.scale_row(Matrix self, Matrix scale)__  
Scale each column of `self` according to a vector `scale`. `scale` should be of size `1 * self.ncol`.
* __Matrix Matrix.\_\_add\_\_(Matrix ma, Matrix mb)__  
Returns a new __Matrix__ which stores the result of `ma+mb`.
* __Matrix Matrix.\_\_sub\_\_(Matrix ma, Matrix mb)__  
Returns a new __Matrix__ which stores the result of `ma-mb`.
* __Matrix Matrix.\_\_mul\_\_(Matrix ma, Matrix mb)__  
Returns a new __Matrix__ which stores the result of `ma*mb`.
* __CuMatrix CuMatrix.new_from_host(MMatrix m)__  
Return a new __CuMatrix__ which is a copy of `m`.
* __MMatrix CuMatrix.new_to_host(CuMatrix self)__  
Return a new __MMatrix__ which is a copy of `self`.
* __string Matrix.\_\_tostring\_\_(Matrix self)__  
Returns a string containing values of __Matrix__ `self`.
---
* __MMatrix MMatrix.load(ChunkData chunk)__  
Return a new __MMatrix__ loaded from the file position in `chunk`.
* __void MMatrix.save(MMatrix self, ChunkFileHandle chunk)__  
Write `self` to the file position in `chunk`.
* __void MMatrix.copy_from(MMatrix ma, MMatrix mb,[int b_bgein, int b_end, int a_begin])__  
Copy a part of `mb`(rows of index `[b_begin..b_end)`) to `ma` beginning at row index `a_begin`. If not specified, `b_begin` will be `0`, `b_end` will be `b.nrow`, `a_begin` will be `0`.

##Examples##
* Use `get_dataref_value` to test __Nerv__'s matrix space allocation.  
```
m = 10
n = 10
fm = nerv.MMatrixFloat(m, n)
dm = nerv.MMatrixDouble(m, n)
for i = 0, m - 1 do
    for j = 0, n - 1 do
        t = i / (j + 1)
        fm[i][j] = t
        dm[i][j] = t
    end
end
print("test fm:get_dataref_value:", fm:get_dataref_value())
print("forced a garbade collect")
collectgarbage("collect")
print("test fm:get_dataref_value:", fm:get_dataref_value())
print(fm)
print(dm)
```
* Test some __Matrix__ calculations.
```
m = 4
n = 4
fm = nerv.CuMatrixFloat(m, n)
dm = nerv.CuMatrixDouble(m, n)
for i = 0, m - 1 do
    for j = 0, n - 1 do
        -- local t = math.random(10)
        t = i / (j + 1)
        fm[i][j] = t
        dm[i][j] = t
    end
end
print(fm)
fs = fm:create()
fs:softmax(fm)
-- print(fs)
print(dm)
ds = dm:create()
ds:softmax(dm)
-- print(ds)
print(fs)
print(fs + fs)
print(ds + ds)
print(fs - fs)
print(ds - ds)
a = fs:create()
a:mul_elem(fs, fs)
print(a)
a:log_elem(fs)
print(a)
```