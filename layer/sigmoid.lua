local SigmoidLayer = nerv.class("nerv.SigmoidLayer", "nerv.Layer")

function SigmoidLayer:__init(id, global_conf)
    self.id = id
    self.gconf = global_conf
end

function SigmoidLayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function SigmoidLayer:propagate(input, output)
    output:sigmoid(input)
end

function SigmoidLayer:back_propagate(next_bp_err, bp_err, input, output)
    next_bp_err:sigmoid_grad(bp_err, output)
end
