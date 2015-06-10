local MSELayer = nerv.class("nerv.MSELayer", "nerv.Layer")

function MSELayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self.gconf = global_conf
    self:check_dim_len(2, -1)
end

function MSELayer:init()
    if self.dim_in[1] ~= self.dim_in[2] then
        nerv.error("mismatching dimensions of previous network output and labels")
    end
    self.total_mse = 0.0
    self.total_frames = 0
end

function MSELayer:update(bp_err, input, output)
    -- no params, therefore do nothing
end

function MSELayer:propagate(input, output)
    local mse = input[1]:create()
    mse:add(input[1], input[2], 1.0, -1.0)
    self.diff = mse:create()
    self.diff:copy_fromd(mse)
    mse:mul_elem(mse, mse)
    mse = mse:rowsum(mse)
    local scale = nerv.CuMatrixFloat(mse:nrow(), 1)
    scale:fill(1 / input[1]:ncol())
    mse:scale_rows_by_col(scale)
    if output[1] ~= nil then
        output[1]:copy_fromd(mse)
    end
    self.total_mse = self.total_mse + mse:colsum()[0]
    self.total_frames = self.total_frames + mse:nrow()
end

-- NOTE: must call propagate before back_propagate
function MSELayer:back_propagate(next_bp_err, bp_err, input, output)
    local nbe = next_bp_err[1]
    nbe:copy_fromd(self.diff)
    self.diff = nil
    if bp_err[1] ~= nil then
        nbe:scale_rows_by_col(bp_err[1])
    end
end

function MSELayer:get_params()
    return {}
end
