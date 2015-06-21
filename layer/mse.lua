local MSELayer = nerv.class("nerv.MSELayer", "nerv.Layer")

function MSELayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self.gconf = global_conf
    self:check_dim_len(2, -1)
end

function MSELayer:init(batch_size)
    if self.dim_in[1] ~= self.dim_in[2] then
        nerv.error("mismatching dimensions of previous network output and labels")
    end
    self.scale = 1 / self.dim_in[1]
    self.total_mse = 0.0
    self.total_frames = 0
    self.mse = self.gconf.cumat_type(batch_size, self.dim_in[1])
    self.mse_sum = self.gconf.cumat_type(batch_size, 1)
    self.diff = self.mse:create()
end

function MSELayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function MSELayer:propagate(input, output)
    local mse = self.mse
    local mse_sum = self.mse_sum
    mse:add(input[1], input[2], 1.0, -1.0)
    self.diff:copy_fromd(mse)
    mse:mul_elem(mse, mse)
    mse_sum:add(mse_sum, mse:rowsum(mse), 0.0, self.scale)
    if output[1] ~= nil then
        output[1]:copy_fromd(mse_sum)
    end
    self.total_mse = self.total_mse + mse_sum:colsum()[0]
    self.total_frames = self.total_frames + mse_sum:nrow()
end

-- NOTE: must call propagate before back_propagate
function MSELayer:back_propagate(bp_err, next_bp_err, input, output)
    local nbe = next_bp_err[1]
    nbe:add(nbe, self.diff, 0.0, 2 * self.scale)
    if bp_err[1] ~= nil then
        nbe:scale_rows_by_col(bp_err[1])
    end
end

function MSELayer:get_params()
    return nerv.ParamRepo({})
end
