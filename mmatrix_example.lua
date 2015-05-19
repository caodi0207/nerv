t = nerv.FloatMMatrix(5, 10)
a = t[1]
for i = 0, 4 do
    for j = 0, 9 do
        t[i][j] = i + j
    end
end
print(t)
print(a)
