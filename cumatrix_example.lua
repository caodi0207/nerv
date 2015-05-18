m = 2
n = 3
t = nerv.FloatCuMatrix(m, n)
t2 = nerv.FloatCuMatrix(m, n)
print(t)
a = t[1]
for i = 0, m - 1 do
    for j = 0, n - 1 do
        t[i][j] = i + j
        t2[i][j] = t[i][j]
    end
end
print(a)
print(t)
print(t2)
print(t + t2)
d = nerv.FloatCuMatrix(3, 3)
for i = 0, 2 do
    for j = 0, 2 do
        d[i][j] = 0
    end
end
d[0][0] = 1
d[1][1] = 2
d[2][2] = 3
print(d)
print(t * d)
print(t:sigmoid())
