#The Nerv NN Package#
Part of the [Nerv](../README.md) toolkit.

##Description##
###Class hierarchy###
it contains __nerv.LayerRepo__, __nerv.ParamRepo__, and __nerv.DAGLayer__(inherits __nerv.Layer__).

###Class hierarchy and their members###
####nerv.ParamRepo#### 
Get parameter object by ID.  
*	`table param_table` Contains the mapping of parameter ID to parameter file(__nerv.ChunkFile__)  
*  __nerv.LayerRepo__ Get layer object by ID.  
* 	`table layers` Contains the mapping of layer ID to layer object.
objects.

####__nerv.DAGLayer__####
Inherits __nerv.Layer__.  
* 	`layers`: __table__, a mapping from a layer ID to its "ref". A ref is a structure that contains reference to space allocations and other info of the layer.
* 	`inputs`: __table__, a mapping from the inputs ports of the DAG layer to the input ports of the sublayer, the key is the port number, the value is `{ref, port}`.
* 	`outputs`:__table__, the counterpart of `inputs`.
* 	`parsed_conn`: __table__, a list of parsed connections, each entry is of format `{{ref_from, port_from}, {ref_to, port_to}}`.
* 	`queue`: __table__, a list of "ref"s, the propagation of the DAGLayer will follow this order, and back-propagation will follow a reverse order.
	
##Methods##

###__nerv.ParamRepo__###

####nerv.ParamRepo:\_\_init(param\_files)####
* 	Parameters:  
	`param_files`: __table__
*	Description:  
	`param_files` is a list of file names that stores parameters, the newed __ParamRepo__ will read them from file and store the mapping for future fetching.  
    
####nerv.Param ParamRepo.get_param(ParamRepo self, string pid, table global_conf)####
*	Returns:  
	__nerv.Layer__  
*	Parameters:  
	`self`: __nerv.ParamRepo__.  
    `pid`: __string__.  
    `global_conf`: __table__.  
*	Description:  
	__ParamRepo__ will find the __nerv.ChunkFile__ `pf` that contains parameter of ID `pid` and return `pf:read_chunk(pid, global_conf)`.

###__nerv.LayerRepo__###
####nerv.LayerRepo:\_\_init(layer\_spec, param\_repo, global\_conf)####
* 	Returns:  
  	__nerv.LayerRepo__.  
* 	Parameters:  
  	`self`: __nerv.ParamRepo__.  
  	`layer_spec`: __table__.  
  	`param_repo`: __nerv.ParamRepo__.  
  	`global_conf`: __table__.  
* 	Description:  
  	__LayerRepo__ will construct the layers specified in `layer_spec`. Every entry in the `layer_spec` table should follow the format below:  
	
    > layer_spec : {[layer_type1] = llist1, [layer_type2] = llist2, ...}   
  	> llist : {layer1, layer2, ...}   
  	> layer : layerid = {param_config, layer_config}   
  	> param_config : {param1 = paramID1, param2 = paramID2}  	 
    
  	__LayerRepo__ will merge `param_config` into `layer_config` and construct a layer by calling `layer_type(layerid, global_conf, layer_config)`.    

####nerv.LayerRepo.get\_layer(self, lid)####
* 	Returns:  
	__nerv.LayerRepo__, the layer with ID `lid`.
* 	Parameters:  
	`self`:__nerv.LayerRepo__.  
	`lid`:__string__.  
*	Description:   
	Returns the layer with ID `lid`.
    
###nerv.DAGLayer###
####nerv.DAGLayer:\_\_init(id, global\_conf, layer\_conf)####
*	Returns:  
	__nerv.DAGLayer__  
*	Parameters:  
	`id`: __string__  
    `global_conf`: __table__  
    `layer_conf`: __table__  
*	Description:  
	The `layer_conf` should contain `layer_conf.sub_layers` which is a __nerv.LayerRepo__ storing the sub layers of the DAGLayer. It should also contain `layer_conf.connections`, which is a string-to-string mapping table describing the DAG connections. See an example below:
    
    ```
    dagL = nerv.DAGLayer("DAGL", global_conf, {["dim_in"] = {input_dim, 2}, ["dim_out"] = {}, ["sub_layers"] = layerRepo,
    	["connections"] = {
    	["<input>[1]"] = "AffineL[1]",
    	["AffineL[1]"] = "SoftmaxL[1]",
    	["<input>[2]"] = "SoftmaxL[2]",
  	}})
    ```
    
####nerv.DAGLayer.init(self, batch\_size)####
*	Parameters:  
	`self`: __nerv.DAGLayer__  
    `batch_size`: __int__
* 	Description:  
	This initialization method will allocate space for output and input matrice, and will call `init()` for each of its sub layers.
    

####nerv.DAGLayer.propagate(self, input, output)####
*	Parameters:  
	`self`: __nerv.DAGLayer__  
    `input`: __table__  
    `output`: __table__  
*	Description:  
	The same function as __nerv.Layer.propagate__, do propagation for each layer in the order of `self.queue`.

####nerv.DAGLayer.back\_propagate(self, next\_bp\_err, bp\_err, input, output)####
*	Parameters:  
	`self`: __nerv.DAGLayer__  
    `next_bp_err`: __table__  
    `bp_err`: __table__  
    `input`: __table__  
    `output`: __table__  
*	Description:  
	The same function as __nerv.Layer.back_propagate__, do back-propagation for each layer in the reverse order of `self.queue`.

####nerv.DAGLayer.update(self, bp\_err, input, output)####
*	Parameters:  
	`self`: __nerv.DAGLayer__  
    `bp_err`: __table__  
    `input`: __table__  
    `output`: __table__  
*	Description:  
	The same function as __nerv.Layer.update__, do update for each layer in the order of `self.queue`.
    
##Examples##
*	aaa
	
```
require 'math'

require 'layer.affine'
require 'layer.softmax_ce'

--[[Example using DAGLayer, a simple two-classification problem]]--

--[[begin global setting and data generation]]--
global_conf =  {lrate = 10, 
                wcost = 1e-6,
                momentum = 0.9,
                cumat_type = nerv.CuMatrixFloat,
               }

input_dim = 5
data_num = 100
param_fn = "../tmp"
ansV = nerv.CuMatrixFloat(input_dim, 1)
for i = 0, input_dim - 1, 1 do
    ansV[i][0] = math.random() - 0.5
end
ansB = math.random() - 0.5
print('displaying ansV')
print(ansV)
print('displaying ansB(bias)')
print(ansB)

dataM = nerv.CuMatrixFloat(data_num, input_dim)
for i = 0, data_num - 1, 1 do
    for j = 0, input_dim - 1, 1 do
        dataM[i][j] = math.random() * 2 - 1
    end
end
refM = nerv.CuMatrixFloat(data_num, 1)
refM:fill(ansB)
refM:mul(dataM, ansV, 1, 1) --refM = dataM * ansV + ansB

labelM = nerv.CuMatrixFloat(data_num, 2)
for i = 0, data_num - 1, 1 do
    if (refM[i][0] > 0) then
        labelM[i][0] = 1 
        labelM[i][1] = 0
    else
        labelM[i][0] = 0
        labelM[i][1] = 1
    end
end
--[[global setting and data generation end]]--


--[[begin network building]]--
--parameters
do
    local affineL_ltp = nerv.LinearTransParam('AffineL_ltp', global_conf)
    affineL_ltp.trans = nerv.CuMatrixFloat(input_dim, 2)
    for i = 0, input_dim - 1, 1 do
        for j = 0, 1, 1 do
            affineL_ltp.trans[i][j] = math.random() - 0.5 
        end
    end
    local affineL_bp = nerv.BiasParam('AffineL_bp', global_conf)
    affineL_bp.trans = nerv.CuMatrixFloat(1, 2)
    for j = 0, 1, 1 do
        affineL_bp.trans[j] = math.random() - 0.5
    end

    local chunk = nerv.ChunkFile(param_fn, 'w')
    chunk:write_chunk(affineL_ltp)
    chunk:write_chunk(affineL_bp)
    chunk:close()

    paramRepo = nerv.ParamRepo({param_fn})
end

--layers
layerRepo = nerv.LayerRepo({
        ["nerv.AffineLayer"] = 
        {
            ["AffineL"] = {{["ltp"] = "AffineL_ltp", ["bp"] = "AffineL_bp"}, {["dim_in"] = {input_dim}, ["dim_out"] = {2}}},
        },
        ["nerv.SoftmaxCELayer"] = 
        {
            ["SoftmaxL"] = {{}, {["dim_in"] = {2, 2}, ["dim_out"] = {}}}
        },
        }, paramRepo, global_conf)
affineL = layerRepo:get_layer("AffineL")
softmaxL = layerRepo:get_layer("SoftmaxL")
print('layers initializing...')
dagL = nerv.DAGLayer("DAGL", global_conf, {["dim_in"] = {input_dim, 2}, ["dim_out"] = {}, ["sub_layers"] = layerRepo,
        ["connections"] = {
           ["<input>[1]"] = "AffineL[1]",
           ["AffineL[1]"] = "SoftmaxL[1]",
           ["<input>[2]"] = "SoftmaxL[2]",
        }})
dagL:init(data_num)
--affineL:init()
--softmaxL:init()
--[[network building end]]--


--[[begin space allocation]]--
print('network input&output&error space allocation...')
dagL_input = {dataM, labelM}
dagL_output = {}
dagL_err = {}
dagL_ierr = {nerv.CuMatrixFloat(data_num, input_dim), nerv.CuMatrixFloat(data_num, 2)}
--[[space allocation end]]--


--[[begin training]]--
ce_last = 0
for l = 0, 10, 1 do
    dagL:propagate(dagL_input, dagL_output)
    dagL:back_propagate(dagL_ierr, dagL_err, dagL_input, dagL_output)
    dagL:update(dagL_err, dagL_input, dagL_output)
    
    if (l % 2 == 0) then
        nerv.utils.printf("training iteration %d finished\n", l)
        nerv.utils.printf("cross entropy: %.8f\n", softmaxL.total_ce - ce_last)
        --nerv.utils.printf("accurate labels: %d\n", calculate_accurate(output, labelM))
        nerv.utils.printf("total frames processed: %.8f\n", softmaxL.total_frames)
    end
    ce_last = softmaxL.total_ce 
end
--[[end training]]--
```