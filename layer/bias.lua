local BiasLayer = nerv.class("nerv.BiasLayer", "nerv.Layer")

function BiasLayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.gconf = global_conf
    self.bias = layer_conf.bias
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self:check_dim_len(1, 1)
end

function BiasLayer:init()
    if self.dim_in[1] ~= self.bias.trans:ncol() then
        nerv.error("mismatching dimensions of input and bias parameter")
    end
    if self.dim_out[1] ~= self.bias.trans:ncol() then
        nerv.error("mismatching dimensions of output and bias parameter")
    end
end

function BiasLayer:propagate(input, output)
    output[1]:copy_fromd(input[1])
    output[1]:add_row(self.bias.trans, 1.0)
end

function BiasLayer:get_params()
    return nerv.ParamRepo({self.bias})
end
