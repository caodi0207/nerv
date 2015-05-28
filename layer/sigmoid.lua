local SigmoidLayer = nerv.class("nerv.SigmoidLayer", "nerv.Layer")

function SigmoidLayer:__init(id, global_conf)
    self.id = id
    self.gconf = global_conf
end

function SigmoidLayer:init()
end

function SigmoidLayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function SigmoidLayer:propagate(input, output)
    output[0]:sigmoid(input[0])
end

function SigmoidLayer:back_propagate(next_bp_err, bp_err, input, output)
    next_bp_err[0]:sigmoid_grad(bp_err[0], output[0])
end
