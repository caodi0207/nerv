m = 600
n = 600
t = nerv.FloatCuMatrix(m, n)
t2 = nerv.FloatCuMatrix(m, n)
-- print(t)
a = t[1]
for i = 0, m - 1 do
    tt = t[i]
    tt2 = t2[i]
    for j = 0, n - 1 do
        tt[j] = i + j
        tt2[j] = t[i][j]
    end
end
-- print(t:rowsum())
