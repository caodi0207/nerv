local MatrixParam = nerv.class('nerv.MatrixParam', 'nerv.Param')
local LinearTransParam = nerv.class('nerv.LinearTransParam', 'nerv.MatrixParam')
local BiasParam = nerv.class('nerv.BiasParam', 'nerv.MatrixParam')
local AffineLayer = nerv.class('nerv.AffineLayer', 'nerv.Layer')

function MatrixParam:read(handle)
    self.trans = self.gconf.cumat_type.new_from_host(
                    nerv.MMatrixFloat.load(handle))
end

function MatrixParam:write(handle)
    self.trans:new_to_host():save(handle)
end

function MatrixParam:train_init()
    self.correction = self.trans:create()
    self.correction:fill(0)
end

function MatrixParam:update(gradient)
    local gconf = self.gconf
    self.correction:add(self.correction, gradient, gconf.momentum, 1.0)
    -- momentum gain
    local mmt_gain = 1.0 / (1.0 - gconf.momentum);
    local n = self.gconf.batch_size * mmt_gain
    -- perform update
    self.trans:add(self.trans, self.correction, 1.0, -gconf.lrate / n)
end

function LinearTransParam:update(gradient)
    MatrixParam.update(self, gradient)
    local gconf = self.gconf
    -- weight decay
    self.trans:add(self.trans, self.trans, 1.0, -gconf.lrate * gconf.wcost)
end

function AffineLayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.ltp = layer_conf.ltp
    self.bp = layer_conf.bp
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self.gconf = global_conf
    self:check_dim_len(1, 1) -- exactly one input and one output
    self.direct_update = layer_conf.direct_update
end

function AffineLayer:init(batch_size)
    if self.ltp.trans:ncol() ~= self.bp.trans:ncol() then
        nerv.error("mismatching dimensions of linear transform and bias paramter")
    end
    if self.dim_in[1] ~= self.ltp.trans:nrow() then
        nerv.error("mismatching dimensions of linear transform parameter and input")
    end
    if self.dim_out[1] ~= self.ltp.trans:ncol() then
        nerv.error("mismatching dimensions of linear transform parameter and output")
    end
    self.ltp_grad = self.ltp.trans:create()
    self.ltp:train_init()
    self.bp:train_init()
end

function AffineLayer:update(bp_err, input, output)
    if self.direct_update then
        self.ltp.correction:mul(input[1], bp_err[1], 1.0, gconf.momentum, 'T', 'N')
        -- momentum gain
        local mmt_gain = 1.0 / (1.0 - gconf.momentum);
        local n = self.gconf.batch_size * mmt_gain
        -- perform update
        self.ltp.trans:add(self.ltp.trans, self.ltp.correction, 1.0, -gconf.lrate / n)
    else
        self.ltp_grad:mul(input[1], bp_err[1], 1.0, 0.0, 'T', 'N')
        self.ltp:update(self.ltp_grad)
    end
    self.bp:update(bp_err[1]:colsum())
end

function AffineLayer:propagate(input, output)
    -- apply linear transform
    output[1]:mul(input[1], self.ltp.trans, 1.0, 0.0, 'N', 'N')
    -- add bias
    output[1]:add_row(self.bp.trans, 1.0)
end

function AffineLayer:back_propagate(bp_err, next_bp_err, input, output)
    next_bp_err[1]:mul(bp_err[1], self.ltp.trans, 1.0, 0.0, 'N', 'T')
end

function AffineLayer:get_params()
    return nerv.ParamRepo({self.ltp, self.bp})
end
