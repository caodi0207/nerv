#The Nerv Parameter Package#
Part of the [Nerv](../README.md) toolkit.

##Description##
###Class hierarchy###
There is a base class __Nerv.Param__ defined in `layer/init.lua`.

###Class hierarchy and their members###
* __nerv.MatrixParam__ inherits __nerv.Param__  
	* `Matrix trans` stores the parameter matrix.
* __nerv.LinearTransParam__ inherits __Nerv.MatrixParam__.  
* __Nerv.BiasParam__ inherits __Nerv.MatrixParam__.  

##Methods##
* __void Param.\_\_init(Param self, string id, table global_conf)__  
Constructor of a __Param__, it will set `self.id` to be `id` and `self.gconf` to be `global_conf`.
* __void Param.set_info(Param self, table info)__  
Set `self.info` to be `info`.
* __table Param.get_info(Param self)__  
Returns `self.info`.
* __void Param.read(Param self, ChunkData pcdata)__  
Abstract method.  
In this method, `self` should in turn calls its members to load from `pcdata`.
* __void Param.write(Param self, ChunkFileHandle pfhandle)__  
Abstract method.  
Save parameters to file. In this method, `self` should in turn calls its members to save to `pfhandle`.

