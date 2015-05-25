m = 10
n = 10
fm = nerv.MMatrixFloat(m, n)
dm = nerv.MMatrixDouble(m, n)
for i = 0, m - 1 do
    for j = 0, n - 1 do
        -- local t = math.random(10)
        t = i / (j + 1)
        fm[i][j] = t
        dm[i][j] = t
    end
end
print(fm)
print(dm)

fc = nerv.CuMatrixFloat(m, n)
dc = nerv.CuMatrixDouble(m, n)
fc:copy_from(fm)
dc:copy_from(dm)
print(fc)
print(dc)
print(fc:softmax())
print(dc:softmax())
