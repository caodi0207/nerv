local SoftmaxCELayer = nerv.class("nerv.SoftmaxCELayer", "nerv.Layer")

function SoftmaxCELayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.gconf = global_conf
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self:check_dim_len(2, -1) -- two inputs: nn output and label
end

function SoftmaxCELayer:init()
    if self.dim_in[1] ~= self.dim_in[1] then
        nerv.error("mismatching dimensions of previous network output and labels")
    end
    self.total_ce = 0.0
    self.total_frames = 0
end

function SoftmaxCELayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function SoftmaxCELayer:propagate(input, output)
    local soutput = input[1]:create()  -- temporary value for calc softmax
    self.soutput = soutput
    soutput:softmax(input[1])
    local ce = soutput:create()
    ce:log_elem(soutput)
    ce:mul_elem(ce, input[2])
--     print(input[1][0])
--     print(soutput[1][0])
     -- add total ce
    self.total_ce = self.total_ce - ce:rowsum():colsum()[0]
    self.total_frames = self.total_frames + soutput:nrow()
end

function SoftmaxCELayer:back_propagate(next_bp_err, bp_err, input, output)
    -- softmax output - label
    next_bp_err[1]:add(self.soutput, input[2], 1.0, -1.0)
end
