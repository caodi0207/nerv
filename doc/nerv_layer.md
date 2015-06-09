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
* __nerv.SoftmaxCELayer__ inherits __nerv.Layer__, `#dim_in` is 2 and `#dim_out` is 1.
	* `float total_ce` 
	* `int total_frams` Records how many frames have passed.
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

