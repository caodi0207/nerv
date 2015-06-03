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

function Param:read(pfhandle)
    nerv.error_method_not_implemented()
end

function Param:write(pfhandle)
    nerv.error_method_not_implemented()
end

local Layer = nerv.class('nerv.Layer')

function Layer:__init(id, global_conf, layer_conf)
    nerv.error_method_not_implemented()
end

function Layer:init(id)
    nerv.error_method_not_implemented()
end

function Layer:update(bp_err, input, output)
    nerv.error_method_not_implemented()
end

function Layer:propagate(input, output)
    nerv.error_method_not_implemented()
end

function Layer:back_propagate(next_bp_err, bp_err, input, output)
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

function Layer:get_dim()
    return self.dim_in, self.dim_out
end

require 'layer.affine'
require 'layer.sigmoid'
require 'layer.softmax_ce'
require 'layer.bias'
require 'layer.window'
