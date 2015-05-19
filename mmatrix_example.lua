m = 10
n = 10
fm = nerv.FloatMMatrix(m, n)
dm = nerv.DoubleMMatrix(m, n)
for i = 0, m - 1 do
    for j = 0, n - 1 do
        -- local t = math.random(10)
        t = i / (j + 1)
        fm[i][j] = t
        dm[i][j] = t
    end
end
print(fm)
-- print(fm:softmax())
print(dm)
-- print(dm:softmax())
