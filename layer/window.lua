local WindowLayer = nerv.class("nerv.WindowLayer", "nerv.Layer")

function WindowLayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.gconf = global_conf
    self.window = layer_conf.window
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self:check_dim_len(1, 1)
end

function WindowLayer:init()
    if self.dim_in[1] ~= self.window.trans:ncol() then
        nerv.error("mismatching dimensions of input and window parameter")
    end
    if self.dim_out[1] ~= self.window.trans:ncol() then
        nerv.error("mismatching dimensions of output and window parameter")
    end
end

function WindowLayer:propagate(input, output)
    output[1]:copy_fromd(input[1])
    output[1]:scale_row(self.window.trans)
end

function WindowLayer:get_params()
    return {self.window}
end
