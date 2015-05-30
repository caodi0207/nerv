local LinearTransParam = nerv.class('nerv.LinearTransParam', 'nerv.Param')
local BiasParam = nerv.class('nerv.BiasParam', 'nerv.LinearTransParam')
local AffineLayer = nerv.class('nerv.AffineLayer', 'nerv.Layer')

function LinearTransParam:read(pcdata)
    self.trans = self.gconf.mat_type.new_from_host(
                    nerv.MMatrixFloat.load(pcdata))
end

function LinearTransParam:write(pfhandle)
    self.trans:new_to_host():save(pfhandle)
end

function AffineLayer:__init(id, global_conf, ltp, bp)
    self.id = id
    self.ltp = ltp
    self.bp = bp
    self.gconf = global_conf
end

function AffineLayer:init()
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
    local n = input[0]:nrow() * mmt_gain
    -- update corrections (accumulated errors)
    ltc:mul(input[0], bp_err[0], 1.0, gconf.momentum, 'T', 'N')
    bc:add(bc, bp_err[0]:colsum(), gconf.momentum, 1.0)
    -- perform update
    ltp:add(ltp, ltc, 1.0, -gconf.lrate / n)
    bp:add(bp, bc, 1.0, -gconf.lrate / n)
    -- weight decay
    ltp:add(ltp, ltp, 1.0, -gconf.lrate * gconf.wcost)
end

function nerv.AffineLayer:propagate(input, output)
    -- apply linear transform
    output[0]:mul(input[0], self.ltp.trans, 1.0, 0.0, 'N', 'N')
    -- add bias
    output[0]:add_row(self.bp.trans, 1.0)
end

function nerv.AffineLayer:back_propagate(next_bp_err, bp_err, input, output)
    next_bp_err[0]:mul(bp_err[0], self.ltp.trans, 1.0, 0.0, 'N', 'T')
end
