function nerv.ParamFile:write_chunkdata(metadata, writer)
    if type(metadata) ~= "table" then
        nerv.error("metadata should be a Lua table")
        return
    end
    return self:_write_chunkdata(table.tostring(metadata), writer)
end
