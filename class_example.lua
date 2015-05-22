A = nerv.class()
function A:_init(x)
    self.x = x
end
function A:f()
    return self.x
end

function A:g()
    return self.x + 1
end

B = nerv.class(A)

function B:f()
    return self.x * self.x
end

a = A(3)
b = B(3)
print(a:f())
print(b:f())
print(b:g())

