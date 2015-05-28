local SoftmaxCELayer = nerv.class("nerv.SoftmaxCELayer", "nerv.Layer")

function SoftmaxCELayer:__init(id, global_conf)
    self.id = id
    self.gconf = global_conf
end

function SoftmaxCELayer:init()
    self.total_ce = 0.0
    self.total_frames = 0
end

function SoftmaxCELayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function SoftmaxCELayer:propagate(input, output)
    local soutput = input[0]:create()  -- temporary value for calc softmax
    self.soutput = soutput
    soutput:softmax(input[0])
    local ce = soutput:create()
    ce:log_elem(soutput)
    ce:mul_elem(ce, input[1])
     -- add total ce
    self.total_ce = self.total_ce - ce:rowsum():colsum()[0]
    self.total_frames = self.total_frames + soutput:nrow()
end

function SoftmaxCELayer:back_propagate(next_bp_err, bp_err, input, output)
    -- softmax output - label
    next_bp_err[0]:add(self.soutput, input[1], 1.0, -1.0)
end
