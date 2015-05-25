p = nerv.Point(0, 0) -- create a Point instance
print(p)
print(p:norm()) -- get 2-norm of the Point
p:set_x(1.0)
p:set_y(2.0)
print(p:norm()) -- get 2-norm of the Point

p = nerv.BetterPoint(1, 2)
print(p)
print(p:norm()) --get 1-norm of the Point

-- create a subclass using lua
local EvenBetterPoint = nerv.class('nerv.EvenBetterPoint', 'nerv.BetterPoint')
bp = nerv.EvenBetterPoint(1, 2)
print(p:norm())
