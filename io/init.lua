function nerv.ParamFile:write_chunkdata(metadata, writer)
    if type(metadata) ~= "table" then
        nerv.error("metadata should be a Lua table")
        return
    end
    return self:_write_chunkdata(table.tostring(metadata), writer)
end

function nerv.ParamFile:write_param(param)
    local id = param.id
    local type = param.__typename
    if id == nil then
        nerv_error("id of param %s must be specified", type)
    end
    self:write_chunkdata({id = id,
                            type = type,
                            info = param:get_info()}, param)
end

function nerv.ParamFile:read_param(id)
    local metadata = self.metadata[id]
    if metadata == nil then
        nerv_error("param with id %s does not exist", id)
    end
    local param = assert(loadstring("return " .. metadata.type .. "(" .. id .. ")"))()
    param:set_info(metadata.info)
    param:read(self:get_chunkdata(id))
end
