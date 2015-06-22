p = nerv.Point(0, 0) -- create a Point instance
print(p)
print(p:norm()) -- get 2-norm of the Point
p:set_x(1.0)
p:set_y(2.0)
print(p:norm()) -- get 2-norm of the Point

bp = nerv.BetterPoint(1, 2)
-- use methods from base class
bp:set_x(1.0)
bp:set_y(2.0)
print(bp)
print(bp:norm()) --get 1-norm of the Point

print(p.__typename)
print(bp.__typename)
