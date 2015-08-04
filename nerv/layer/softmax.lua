local SoftmaxLayer = nerv.class("nerv.SoftmaxLayer", "nerv.Layer")

function SoftmaxLayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.gconf = global_conf
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self:check_dim_len(1, 1) -- two inputs: nn output and label
end

function SoftmaxLayer:init(batch_size)
    if self.dim_in[1] ~= self.dim_out[1] then
        nerv.error("mismatching dimensions of input and output")
    end
end

function SoftmaxLayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function SoftmaxLayer:propagate(input, output)
    output[1]:softmax(input[1])
end

function SoftmaxLayer:back_propagate(bp_err, next_bp_err, input, output)
    nerv.error_method_not_implemented()
end

function SoftmaxLayer:get_params()
    return nerv.ParamRepo({})
end
