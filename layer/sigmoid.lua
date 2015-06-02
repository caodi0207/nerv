local SigmoidLayer = nerv.class("nerv.SigmoidLayer", "nerv.Layer")

function SigmoidLayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.gconf = global_conf
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self:check_dim_len(1, 1)
end

function SigmoidLayer:init()
    if self.dim_in[1] ~= self.dim_out[1] then
        nerv.error("mismatching dimensions of input and output")
    end
end

function SigmoidLayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function SigmoidLayer:propagate(input, output)
    output[1]:sigmoid(input[1])
end

function SigmoidLayer:back_propagate(next_bp_err, bp_err, input, output)
    next_bp_err[1]:sigmoid_grad(bp_err[1], output[1])
end
