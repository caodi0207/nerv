#The Nerv utility functions#
Part of the [Nerv](../README.md) toolkit.
##Methods##
* __string = nerv.setmetatable(table self, string tname)__  
A registered function, the original function is `luaT_lua_setmetatable`. It assigns the metatable registered in __luaT__ by the name *tname* to the table *self*. And return *tname* to user.
