#The Nerv Parameter Package#
Part of the [Nerv](../README.md) toolkit.

##Description##
###Class hierarchy###
There is a base class __Nerv.Param__ defined in `layer/init.lua`.  
__Nerv.MatrixParam__ inherits __Nerv.Param__.  
__Nerv.LinearTransParam__, __Nerv.BiasParam__ inherits __Nerv.MatrixParam__.  
###Class member###
* __Nerv.MatrixParam__
	* __nerv.CuMatrix__ trans
	Stores the parameter matrix.

##Methods##
* __void Param.\_\_init(Param self, string id, table global_conf)__  
Constructor of a __Param__, it will set `self.id` to be `id` and `self.gconf` to be `global_conf`.
* __void Param.set_info(table info)__  
Set `self.info` to be `info`.
* __table Param.get_info()__  
Returns `self.info`.
* __void Param.read(ChunkData pcdata)__  
This is not implemented in `nerv.Param`.
* __void MatrixParam.read(MatrixParam self, ChunkData pcdata)__  
Read `self.trans` from `pcdata`.
* __void MatrixParam.write(MatrixParam self, ChunkFileHandle pfhandle)__  
Write `self.trans` to `pfhandle`.