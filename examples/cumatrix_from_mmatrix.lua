m = 3
n = 4
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
fc:copy_fromh(fm)
dc:copy_fromh(dm)
print("fc and dc")
print(fc)
print(dc)
dc[1]:copy_tod(dc[0])
print("dc[1] copied to dc[0]")
print(dc)
print("softmax of fc and dc")
print(fc:softmax())
print(dc:softmax())
