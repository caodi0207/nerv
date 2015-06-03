local ParamRepo = nerv.class("nerv.ParamRepo")

function ParamRepo:__init(param_files)
    local param_table = {}
    if type(param_files) ~= "table" then
        nerv.error("param file table is need")
    end
    for i = 1, #param_files do
        local pf = nerv.ChunkFile(param_files[i], "r")
        for cid, cspec in pairs(pf.metadata) do
            if param_table[cid] ~= nil then
                nerv.error("conflicting chunk id in param files")
            end
            param_table[cid] = pf
        end
    end
    self.param_table = param_table
end

function ParamRepo:get_param(pid, global_conf)
    local pf = self.param_table[pid]
    if pf == nil then
        nerv.error("param with id %s not found", pid)
    end
    return pf:read_chunk(pid, global_conf)
end
