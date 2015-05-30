#The Nerv utility functions#
Part of the [Nerv](../README.md) toolkit.
##Methods##
* __string = nerv.typename(obj a)__
A registered function, the original function is `luaT_lua_typename`. In some cases if you call `type(a)`  for object of some class in __Nerv__(like __Nerv.CuMatrix__) it will only return "userdata"(because it is created in C), in this case you can use this method to get its type.

---

* __metatable = nerv.getmetatable(string tname)__
A registered function, the original function is `luaT_lua_getmetatable`. `tname` should be a class name that has been registered in __luaT__.

* __metatable = nerv.newmetatable(string tname, string parenttname, function constructor, function destructor, function factory)__
A registered function, the original function is `luaT_newmetatable`, it returns the metatable of the created class by the name `tname`.
* __string = nerv.setmetatable(table self, string tname)__  
A registered function, the original function is `luaT_lua_setmetatable`. It assigns the metatable registered in __luaT__ by the name *tname* to the table *self*. And return *tname* to user.
