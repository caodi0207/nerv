function nerv.ChunkFile:write_chunkdata(metadata, writer)
    if type(metadata) ~= "table" then
        nerv.error("metadata should be a Lua table")
        return
    end
    return self:_write_chunkdata(table.tostring(metadata), writer)
end

function nerv.ChunkFile:write_chunk(chunk)
    local id = chunk.id
    local type = chunk.__typename
    if id == nil then
        nerv_error("id of chunk %s must be specified", type)
    end
    self:write_chunkdata({id = id,
                            type = type,
                            info = chunk:get_info()}, chunk)
end

function nerv.ChunkFile:read_chunk(id, global_conf)
    local metadata = self.metadata[id]
    if metadata == nil then
        nerv.error("chunk with id %s does not exist", id)
    end
    local chunk_type = assert(loadstring("return " .. metadata.type))()
    local chunk = chunk_type(id, global_conf)
    chunk:set_info(metadata.info)
    chunk:read(self:get_chunkdata(id))
    return chunk
end
