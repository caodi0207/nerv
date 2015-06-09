local SoftmaxCELayer = nerv.class("nerv.SoftmaxCELayer", "nerv.Layer")

function SoftmaxCELayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.gconf = global_conf
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self.compressed = layer_conf.compressed
    if self.compressed == nil then
        self.compressed = false
    end
    self:check_dim_len(2, -1) -- two inputs: nn output and label
end

function SoftmaxCELayer:init()
    if not self.compressed and (self.dim_in[1] ~= self.dim_in[2]) then
        nerv.error("mismatching dimensions of previous network output and labels")
    end
    self.total_ce = 0.0
    self.total_correct = 0
    self.total_frames = 0
end

function SoftmaxCELayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function SoftmaxCELayer:propagate(input, output)
    local soutput = input[1]:create()  -- temporary value for calc softmax
    self.soutput = soutput
    local classified = soutput:softmax(input[1])
    local ce = soutput:create()
    ce:log_elem(soutput)
    local label = input[2]
    if self.compressed then
        label = label:decompress(input[1]:ncol())
    end
    ce:mul_elem(ce, label)
    -- add total ce
    self.total_ce = self.total_ce - ce:rowsum():colsum()[0]
    self.total_frames = self.total_frames + soutput:nrow()
    -- TODO: add colsame for uncompressed label
    if self.compressed then
        self.total_correct = self.total_correct + classified:colsame(input[2])[0]
    end
end

function SoftmaxCELayer:back_propagate(next_bp_err, bp_err, input, output)
    -- softmax output - label
    local label = input[2]
    if self.compressed then
        label = label:decompress(input[1]:ncol())
    end
    next_bp_err[1]:add(self.soutput, label, 1.0, -1.0)
end

function SoftmaxCELayer:get_params()
    return {}
end
