#The Nerv OOP#
Part of the [Nerv](../README.md) toolkit.
##Methods##
* __metatable mt, metatable mpt = nerv.class(string tname, string parenttname)__  
This method is used to create a class by the name `tname`, which inherits `parenttname` in __Nerv__, then you create a new instance of this class by calling `obj=tname(...)`. The  `tname.__init(...)` method(if defined) will be called in the constructing. The metatable of the class and its parent class will be returned.

##Examples##
* This example implements a simple `nerv.Counter` class which is inherited by `nerv.BetterCounter`.  

```
do
    nerv.class("nerv.Counter")
    function nerv.Counter:__init(c)
        if (c) then
            self.c = c
        else
            self.c = 0
        end
    end
end
do
    local mt, mpt = nerv.class("nerv.BetterCounter", "nerv.Counter")
    function nerv.BetterCounter:__init(c, bc)
        mpt.__init(self, c)
        if (bc) then
            self.bc = bc
        else
            self.bc = 0
        end
    end
end
c1 = nerv.Counter(1)
print(c1.c)
bc1 = nerv.BetterCounter(1, 1)
print(bc1.c, bc1.bc)
```