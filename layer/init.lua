-- The following methods must be implemented to let a layer work properly

local Param = nerv.class('nerv.Param')

function nerv.Param:__init(id)
    self.id = id
end

function nerv.Param:get_info()
    return self.info
end

function nerv.Param:set_info(info)
    self.info = info
end

function nerv.Param:read(pfhandle)
    nerv.error_method_not_implemented()
end

function nerv.Param:write(pfhandle)
    nerv.error_method_not_implemented()
end

local Layer = nerv.class('nerv.Layer')

function nerv.Layer:_init(id, global_conf, ...)
    nerv.error_method_not_implemented()
end

function nerv.Layer:update(bp_err, input, output)
    nerv.error_method_not_implemented()
end

function nerv.Layer:propagate(input, output)
    nerv.error_method_not_implemented()
end

function nerv.Layer:back_propagate(next_bp_err, bp_err, input, output)
    nerv.error_method_not_implemented()
end
