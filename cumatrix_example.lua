t = nerv.FloatCuMatrix(10, 20)
print(t)
a = t[1]
for i = 0, 9 do
    for j = 0, 19 do
        t[i][j] = i + j
    end
end
print(t)
print(a)
