local MatrixParam = nerv.class('nerv.MatrixParam', 'nerv.Param')
local LinearTransParam = nerv.class('nerv.LinearTransParam', 'nerv.MatrixParam')
local BiasParam = nerv.class('nerv.BiasParam', 'nerv.MatrixParam')
local AffineLayer = nerv.class('nerv.AffineLayer', 'nerv.Layer')

function MatrixParam:read(pcdata)
    self.trans = self.gconf.mat_type.new_from_host(
                    nerv.MMatrixFloat.load(pcdata))
end

function MatrixParam:write(pfhandle)
    self.trans:new_to_host():save(pfhandle)
end

function AffineLayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.ltp = layer_conf.ltp
    self.bp = layer_conf.bp
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self.gconf = global_conf
    self:check_dim_len(1, 1) -- exactly one input and one output
end

function AffineLayer:init()
    if self.ltp.trans:ncol() ~= self.bp.trans:ncol() then
        nerv.error("mismatching dimensions of linear transform and bias paramter")
    end
    if self.dim_in[1] ~= self.ltp.trans:nrow() then
        nerv.error("mismatching dimensions of linear transform parameter and input")
    end
    if self.dim_out[1] ~= self.ltp.trans:ncol() then
        nerv.error("mismatching dimensions of linear transform parameter and output")
    end

    -- linear transform correction
    self.ltc = self.ltp.trans:create()
    self.ltc:fill(0)
    -- bias correction
    self.bc = self.bp.trans:create()
    self.bc:fill(0)
end

function nerv.AffineLayer:update(bp_err, input, output)
    local ltp = self.ltp.trans
    local bp = self.bp.trans
    local ltc = self.ltc
    local bc = self.bc
    local gconf = self.gconf
    -- momentum gain
    local mmt_gain = 1.0 / (1.0 - gconf.momentum);
    local n = input[1]:nrow() * mmt_gain
    -- update corrections (accumulated errors)
    ltc:mul(input[1], bp_err[1], 1.0, gconf.momentum, 'T', 'N')
    bc:add(bc, bp_err[1]:colsum(), gconf.momentum, 1.0)
    -- perform update
    ltp:add(ltp, ltc, 1.0, -gconf.lrate / n)
    bp:add(bp, bc, 1.0, -gconf.lrate / n)
    -- weight decay
    ltp:add(ltp, ltp, 1.0, -gconf.lrate * gconf.wcost)
end

function nerv.AffineLayer:propagate(input, output)
    -- apply linear transform
    output[1]:mul(input[1], self.ltp.trans, 1.0, 0.0, 'N', 'N')
    -- add bias
    output[1]:add_row(self.bp.trans, 1.0)
end

function nerv.AffineLayer:back_propagate(next_bp_err, bp_err, input, output)
    next_bp_err[1]:mul(bp_err[1], self.ltp.trans, 1.0, 0.0, 'N', 'T')
end
