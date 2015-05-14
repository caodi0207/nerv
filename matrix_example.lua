t = nerv.FloatMatrix(2, 3)
print(t:get_elem(1))
t:set_elem(1, 3.23432);
print(t:get_elem(1))
print(t)
t = nerv.FloatMatrix(10, 20)
t:set_elem(1, 3.34);
print(t)
a = t[1]
for i = 0, 9 do
    for j = 0, 19 do
        t[i][j] = i + j
    end
end
print(t)
print(a)
