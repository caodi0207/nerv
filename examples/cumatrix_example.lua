m = 10
n = 10
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
