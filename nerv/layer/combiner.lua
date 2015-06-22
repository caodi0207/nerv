local CombinerLayer = nerv.class('nerv.CombinerLayer', 'nerv.Layer')

function CombinerLayer:__init(id, global_conf, layer_conf)
    self.id = id
    self.lambda = layer_conf.lambda
    self.dim_in = layer_conf.dim_in
    self.dim_out = layer_conf.dim_out
    self.gconf = global_conf
    self:check_dim_len(#self.lambda, -1)
    if #self.dim_in < 1 then
        nerv.error("no input specified")
    end
    if #self.dim_out < 1 then
        nerv.error("no output specified")
    end
end

function CombinerLayer:init(batch_size)
    local dim = self.dim_in[1]
    for i = 2, #self.dim_in do
        if self.dim_in[i] ~= dim then
            nerv.error("mismatching dimensions of inputs")
        end
    end
    for i = 1, #self.dim_out do
        if self.dim_out[i] ~= dim then
            nerv.error("mismatching dimensions of inputs/outputs")
        end
    end
    self.sum = self.gconf.cumat_type(batch_size, dim)
end

function CombinerLayer:update(bp_err, input, output)
end

function CombinerLayer:propagate(input, output)
    output[1]:fill(0)
    for i = 1, #self.dim_in do
        output[1]:add(output[1], input[i], 1.0, self.lambda[i])
    end
    for i = 2, #self.dim_out do
        output[i]:copy_fromd(output[1])
    end
end

function CombinerLayer:back_propagate(bp_err, next_bp_err, input, output)
    local sum = self.sum
    sum:copy_fromd(bp_err[1])
    for i = 2, #self.dim_out do
        sum:add(sum, bp_err[i], 1.0, 1.0)
    end
    for i = 1, #self.dim_in do
        next_bp_err[i]:add(next_bp_err[i], sum, 0.0, self.lambda[i])
    end
end

function CombinerLayer:get_params()
    return nerv.ParamRepo({})
end
