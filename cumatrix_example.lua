m = 10
n = 10
t = nerv.FloatCuMatrix(m, n)
-- print(t)
a = t[1]
for i = 0, m - 1 do
    for j = 0, n - 1 do
--        t[i][j] = i + j
        t[i][j] = math.random(10)
    end
end
print(t)
print(t:colsum())
print(t:colmax())
print(t:softmax())
