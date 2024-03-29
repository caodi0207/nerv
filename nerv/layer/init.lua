-- The following methods must be implemented to let a layer work properly

local Param = nerv.class('nerv.Param')

function Param:__init(id, global_conf)
    self.id = id
    self.gconf = global_conf
end

function Param:get_info()
    return self.info
end

function Param:set_info(info)
    self.info = info
end

function Param:read(handle)
    nerv.error_method_not_implemented()
end

function Param:write(handle)
    nerv.error_method_not_implemented()
end

function Param:update(gradient)
    nerv.error_method_not_implemented()
end

local Layer = nerv.class('nerv.Layer')

function Layer:__init(id, global_conf, layer_conf)
    nerv.error_method_not_implemented()
end

function Layer:init(batch_size)
    nerv.error_method_not_implemented()
end

function Layer:update(bp_err, input, output)
    nerv.error_method_not_implemented()
end

function Layer:propagate(input, output)
    nerv.error_method_not_implemented()
end

function Layer:back_propagate(bp_err, next_bp_err, input, output)
    nerv.error_method_not_implemented()
end

function Layer:check_dim_len(len_in, len_out)
    local expected_in = #self.dim_in
    local expected_out = #self.dim_out
    if len_in > 0 and expected_in ~= len_in then
        nerv.error("layer %s expects %d inputs, %d given",
                    self.id, len_in, expected_in)
    end
    if len_out > 0 and expected_out ~= len_out then
        nerv.error("layer %s expects %d outputs, %d given",
                    self.id, len_out, expected_out)
    end
end

function Layer:get_params()
    nerv.error_method_not_implemented()
end

function Layer:get_dim()
    return self.dim_in, self.dim_out
end

nerv.include('affine.lua')
nerv.include('sigmoid.lua')
nerv.include('softmax_ce.lua')
nerv.include('bias.lua')
nerv.include('window.lua')
nerv.include('mse.lua')
nerv.include('combiner.lua')
nerv.include('affine_recurrent.lua')
nerv.include('softmax.lua')
