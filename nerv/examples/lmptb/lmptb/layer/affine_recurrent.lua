local Recurrent = nerv.class('nerv.AffineRecurrentLayer', 'nerv.Layer')

--id: string
--global_conf: table
--layer_conf: table
--Get Parameters
function Recurrent:__init(id, global_conf, layer_conf)
    self.id = id
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self.gconf = global_conf

    self.bp = layer_conf.bp
    self.ltp_ih = layer_conf.ltp_ih --from input to hidden
    self.ltp_hh = layer_conf.ltp_hh --from hidden to hidden
    
    self:check_dim_len(2, 1)
    self.direct_update = layer_conf.direct_update
end

--Check parameter 
function Recurrent:init(batch_size)
    if (self.ltp_ih.trans:ncol() ~= self.bp.trans:ncol() or 
        self.ltp_hh.trans:ncol() ~= self.bp.trans:ncol()) then
        nerv.error("mismatching dimensions of ltp and bp")
    end
    if (self.dim_in[1] ~= self.ltp_ih.trans:nrow() or
        self.dim_in[2] ~= self.ltp_hh.trans:nrow()) then
        nerv.error("mismatching dimensions of ltp and input")
    end
    if (self.dim_out[1] ~= self.bp.trans:ncol()) then
        nerv.error("mismatching dimensions of bp and output")
    end
    
    self.ltp_ih_grad = self.ltp_ih.trans:create()
    self.ltp_hh_grad = self.ltp_hh.trans:create()
    self.ltp_ih:train_init()
    self.ltp_hh:train_init()
    self.bp:train_init()
end

function Recurrent:update(bp_err, input, output)
    if (self.direct_update == true) then
        local ltp_ih = self.ltp_ih.trans
        local ltp_hh = self.ltp_hh.trans
        local bp = self.bp.trans
        local ltc_ih = self.ltc_ih
        local ltc_hh = self.ltc_hh
        local bc = self.bc
        local gconf = self.gconf
        -- momentum gain
        local mmt_gain = 1.0 / (1.0 - gconf.momentum);
        local n = input[1]:nrow() * mmt_gain
        -- update corrections (accumulated errors)
        self.ltp_ih.correction:mul(input[1], bp_err[1], 1.0, gconf.momentum, 'T', 'N')
        self.ltc_hh.correction:mul(input[2], bp_err[1], 1.0, gconf.momentum, 'T', 'N')
        self.bp.correction:add(bc, bp_err[1]:colsum(), gconf.momentum, 1.0)
        -- perform update
        ltp_ih:add(ltp_ih, self.ltp_ih.correction, 1.0, -gconf.lrate / n)
        ltp_hh:add(ltp_hh, self.ltp_hh.correction, 1.0, -gconf.lrate / n)
        bp:add(bp, self.bp.correction, 1.0, -gconf.lrate / n)
        -- weight decay
        ltp_ih:add(ltp_ih, ltp_ih, 1.0, -gconf.lrate * gconf.wcost)
        ltp_hh:add(ltp_hh, ltp_hh, 1.0, -gconf.lrate * gconf.wcost)
    else
        self.ltp_ih_grad:mul(input[1], bp_err[1], 1.0, 0.0, 'T', 'N') 
        self.ltp_ih:update(self.ltp_ih_grad)
        self.ltp_hh_grad:mul(input[2], bp_err[1], 1.0, 0.0, 'T', 'N') 
        self.ltp_hh:update(self.ltp_hh_grad)       
        self.bp:update(bp_err[1]:colsum())
    end
end

function Recurrent:propagate(input, output)
    output[1]:mul(input[1], self.ltp_ih.trans, 1.0, 0.0, 'N', 'N')
    output[1]:mul(input[2], self.ltp_hh.trans, 1.0, 1.0, 'N', 'N')
    output[1]:add_row(self.bp.trans, 1.0)
end

function Recurrent:back_propagate(bp_err, next_bp_err, input, output)
    next_bp_err[1]:mul(bp_err[1], self.ltp_ih.trans, 1.0, 0.0, 'N', 'T')
    next_bp_err[2]:mul(bp_err[1], self.ltp_hh.trans, 1.0, 0.0, 'N', 'T')
    for i = 0, next_bp_err[2]:nrow() - 1 do
        for j = 0, next_bp_err[2]:ncol() - 1 do
            if (next_bp_err[2][i][j] > 10) then next_bp_err[2][i][j] = 10 end
            if (next_bp_err[2][i][j] < -10) then next_bp_err[2][i][j] = -10 end
        end
    end
end

function Recurrent:get_params()
    return {self.ltp_ih, self.ltp_hh, self.bp}
end
