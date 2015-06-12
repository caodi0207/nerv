#The Nerv Layer Package#
Part of the [Nerv](../README.md) toolkit.

##Description##
__nerv.Layer__ is the base class and most of its methods are abstract.  
###Class hierarchy and their members###
* __nerv.Layer__.  
	* `table dim_in` It specifies the dimensions of the inputs.  
	* `table dim_out` It specifies the dimensions of the outputs.  
	* `string id` ID of this layer.
	* `table gconf` Stores the `global_conf`.
* __nerv.AffineLayer__ inherits __nerv.Layer__, both `#dim_in` and `#dim_out` are 1. 
	* `MatrixParam ltp` The liner transform parameter.
	* `BiasParam bp` The bias parameter.
* __nerv.BiasLayer__ inherits __nerv.Layer__, both `#dim_in` nad `#dim_out` are 1.
	* `BiasParam bias` The bias parameter.
* __nerv.SigmoidLayer__ inherits __nerv.Layer__, both `#dim_in` and `#dim_out` are 1.
* __nerv.SoftmaxCELayer__ inherits __nerv.Layer__, `#dim_in` is 2 and `#dim_out` is -1(optional). `input[1]` is the input to the softmax layer, `input[2]` is the reference distribution. In its `propagate(input, output)` method, if `output[1] ~= nil`, cross\_entropy value will outputed.
	* `float total_ce` Records the accumlated cross entropy value.
	* `int total_frams` Records how many frames have passed.  
	* `bool compressed` The reference distribution can be a one-hot format. This feature is enabled by `layer_conf.compressed`.

##Methods##
* __void Layer.\_\_init(Layer self, string id, table global_conf, table layer_conf)__  
Abstract method.  
The constructing method should assign `id` to `self.id` and `global_conf` to `self.gconf`, `layer_conf.dim_in` to `self.dim_in`, `layer_conf.dim_out` to `self.dim_out`. `dim_in` and `dim_out` are a list specifies the dimensions of the inputs and outputs. Also, `layer_conf` will include the parameters, which should also be properly saved.
* __void Layer.init(Layer self)__  
Abstract method.  
Initialization method, in this method the layer should do some self-checking and allocate space for intermediate results.
* __void Layer.update(Layer self, table bp_err, table input, table output)__  
Abstract method.  
`bp_err[i]` should be the error on `output[i]`. In this method the parameters of `self` is updated.
* __void Layer.propagate(Layer self, table input, table output)__  
Abstract method.  
Given `input` and the current parameters, propagate and store the result in `output`.
* __void Layer.back_propagate(Layer self, Matrix next_bp_err, Matrix bp_err, Matrix input, Matrix output)__  
Abstract method.  
Calculate the error on the inputs and store them in `next_bp_err`.

* __void Layer.check_dim_len(int len_in, int len_out)__  
Check whether `#self.dim_in == len_in` and `#self.dim_out == len_out`, if violated, an error will be posted.
* __void Layer.get_params(Layer self)__  
Abstract method.  
The layer should return a list containing its parameters.

####nerv.Layer.get\_dim(self)####
*	Returns:
	`dim_in`: __table__.  
    `dim_out`: __table__.  
*	Parameters:  
	`self`: __nerv.Layer__.
*	Description:  
	Returns `self.dim_in, self.dim_out`.

##Examples##
* a basic example using __Nerv__ layers to a linear classification.

```
require 'math'

require 'layer.affine'
require 'layer.softmax_ce'

--[[Example using layers, a simple two-classification problem]]--

function calculate_accurate(networkO, labelM)
    sum = 0
    for i = 0, networkO:nrow() - 1, 1 do
        if (labelM[i][0] == 1 and networkO[i][0] >= 0.5) then
            sum = sum + 1
        end
        if (labelM[i][1] == 1 and networkO[i][1] >= 0.5) then
            sum = sum + 1
        end 
    end
    return sum
end

--[[begin global setting and data generation]]--
global_conf =  {lrate = 10, 
                wcost = 1e-6,
                momentum = 0.9,
                cumat_type = nerv.CuMatrixFloat}

input_dim = 5
data_num = 100
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
affineL_ltp = nerv.LinearTransParam('AffineL_ltp', global_conf)
affineL_ltp.trans = nerv.CuMatrixFloat(input_dim, 2)
for i = 0, input_dim - 1, 1 do
    for j = 0, 1, 1 do
        affineL_ltp.trans[i][j] = math.random() - 0.5 
    end
end
affineL_bp = nerv.BiasParam('AffineL_bp', global_conf)
affineL_bp.trans = nerv.CuMatrixFloat(1, 2)
for j = 0, 1, 1 do
    affineL_bp.trans[j] = math.random() - 0.5
end

--layers
affineL = nerv.AffineLayer('AffineL', global_conf, {['ltp'] = affineL_ltp,
                                                      ['bp'] = affineL_bp,
                                                      dim_in = {input_dim},
                                                      dim_out = {2}})
softmaxL = nerv.SoftmaxCELayer('softmaxL', global_conf, {dim_in = {2, 2},
                                                         dim_out = {}})
print('layers initializing...')
affineL:init()
softmaxL:init()
--[[network building end]]--


--[[begin space allocation]]--
print('network input&output&error space allocation...')
affineI = {dataM} --input to the network is data
affineO = {nerv.CuMatrixFloat(data_num, 2)}
softmaxI = {affineO[1], labelM}
softmaxO = {}
output = nerv.CuMatrixFloat(data_num, 2)

affineE = {nerv.CuMatrixFloat(data_num, 2)}
--[[space allocation end]]--


--[[begin training]]--
ce_last = 0
for l = 0, 10, 1 do
    affineL:propagate(affineI, affineO)
    softmaxL:propagate(softmaxI, softmaxO)
    output:softmax(softmaxI[1])

    softmaxL:back_propagate(affineE, {}, softmaxI, softmaxO)
    
    affineL:update(affineE, affineI, affineO) 

    if (l % 5 == 0) then
        nerv.utils.printf("training iteration %d finished\n", l)
        nerv.utils.printf("cross entropy: %.8f\n", softmaxL.total_ce - ce_last)
        ce_last = softmaxL.total_ce 
        nerv.utils.printf("accurate labels: %d\n", calculate_accurate(output, labelM))
        nerv.utils.printf("total frames processed: %.8f\n", softmaxL.total_frames)
    end
end
--[[end training]]--
```
