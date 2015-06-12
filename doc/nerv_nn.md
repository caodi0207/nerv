#The Nerv NN Package#
Part of the [Nerv](../README.md) toolkit.

##Description##
###Class hierarchy###
it contains __nerv.LayerRepo__, __nerv.ParamRepo__, and __nerv.DAGLayer__(inherits __nerv.Layer__).

###Class hierarchy and their members###
* __nerv.ParamRepo__ Get parameter object by ID.  
	* `table param_table` Contains the mapping of parameter ID to parameter file(__nerv.ChunkFile__) 
*  __nerv.LayerRepo__ Get layer object by ID.  
	* `table layers` Contains the mapping of layer ID to layer object.
objects.
* __nerv.DAGLayer__ inherits __nerv.Layer__.  
	* `table layers` Mapping from a layer ID to its "ref". A ref is of the structure below:
	 ```
     nerv.Layer layer --its layer
     nerv.Matrix inputs	
     nerv.Matrix outputs 
     nerv.Matrix err_inputs
     nerv.Matrix err_outputs
     table next_layers
     int input_len -- #dim_in
     int output_len -- #dim_out
     int in_deg 
     bool visited -- used in topology sort
     ```
	* `inputs`
	* `outputs`
	* `parsed_conn`
	* `queue`
	
##Methods##
###__nerv.ParamRepo__###
* __void ParamRepo:\_\_init(table param_files)__  
`param_files` is a list of file names that stores parameters, the newed __ParamRepo__ will read them from file and store the mapping for future fetching.  
* __nerv.Param ParamRepo.get_param(ParamRepo self, string pid, table global_conf)__  
__ParamRepo__ will find the __nerv.ChunkFile__ `pf` that contains parameter of ID `pid` and return `pf:read_chunk(pid, global_conf)`.

###__nerv.LayerRepo__###
* __void LayerRepo:\_\_init(table layer_spec, ParamRepo param_repo, table global_conf)__  
__LayerRepo__ will construct the layers specified in `layer_spec`. Every entry in the `layer_spec` table should follow the format below:  
```
layer_spec : {[layer_type1] = llist1, [layer_type2] = llist2, ...}
llist : {layer1, layer2, ...}
layer : layerid = {param_config, layer_config}
param_config : {param1 = paramID1, param2 = paramID2}
```
__LayerRepo__ will merge `param_config` into `layer_config` and construct a layer by calling `layer_type(layerid, global_conf, layer_config)`.

* __LayerRepo.get_layer(self, lid)__  
 	* Returns  
      	__nerv.LayerRepo__ the layer with ID `lid`.
 	* Parameters
    	`self`:__nerv.LayerRepo__.  
    	`lid`:__string__, the ID of the layer to fetch.

###__nerv.DAGLayer__###
* __DAGLayer:\_\_init(id, global_conf, layer_conf, [a, b, ...])__  
	Returns: 
	__string__, dfdfdfddf
    __asasa__, asasasasa
    Parameters:
	`id`: __string__, the ID of the layer.  
    `global_conf`:__table__,the global config.  
	
     