function nerv.ChunkFile:write_chunkdata(metadata, writer)
    if type(metadata) ~= "table" then
        nerv.error("metadata should be a Lua table")
        return
    end
    return self._write_chunkdata(self.handle, table.tostring(metadata), writer)
end

function nerv.ChunkFile:write_chunk(chunk)
    local id = chunk.id
    local type = chunk.__typename
    if id == nil then
        nerv.error("id of chunk %s must be specified", type)
    end
    self:write_chunkdata({id = id,
                            type = type,
                            info = chunk:get_info()}, chunk)
end

function nerv.ChunkFile:read_chunk(id, global_conf)
    if self.metadata == nil then
        nerv.error("wrong file opening mode")
    end
    local metadata = self.metadata[id]
    if metadata == nil then
        nerv.error("chunk with id %s does not exist", id)
    end
    local chunk_type = nerv.get_type(metadata.type)
    local chunk = chunk_type(id, global_conf)
    chunk:set_info(metadata.info)
    chunk:read(self._get_chunkdata(self.handle, metadata._chunk_info))
    return chunk
end

function nerv.ChunkFile:close()
    self._close(self.handle)
end

local DataReader = nerv.class("nerv.DataReader")

function DataReader:__init(global_conf, reader_conf)
    nerv.error_method_not_implemented()
end

function DataReader:get_data()
    nerv.error_method_not_implemented()
end

local DataBuffer = nerv.class("nerv.DataBuffer")

function DataBuffer:__init(global_conf, buffer_conf)
    nerv.error_method_not_implemented()
end

function DataBuffer:get_batch()
    nerv.error_method_not_implemented()
end

nerv.include('sgd_buffer.lua')
