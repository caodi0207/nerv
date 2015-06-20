local ParamRepo = nerv.class("nerv.ParamRepo")
function ParamRepo:__init(plist)
    self.params = {}
    if plist ~= nil then
        for i, p in ipairs(plist) do
            self.params[p.id] = p
        end
    end
end

function ParamRepo:add(pid, p)
    if self.params[pid] ~= nil then
        nerv.error("duplicate params with the same id: %s", pid)
    end
    self.params[pid] = p
end

function ParamRepo:remove(pid, p)
    if self.params[pid] == nil then
        nerv.error("param %s does not exit", pid)
    end
    table.remove(self.params, pid)
end

function ParamRepo.merge(repos)
    local self = nerv.ParamRepo()
    for i, repo in ipairs(repos) do
        if not nerv.is_type(repo, "nerv.ParamRepo") then
            nerv.error("nerv.ParamRepo objects expected, got %s", repo)
        end
        for pid, p in pairs(repo.params) do
            self:add(pid, p)
        end
    end
    return self
end

function ParamRepo:import(param_files, pids, gconf)
    if type(param_files) ~= "table" then
        nerv.error("param file table is need")
    end
    for i = 1, #param_files do
        local pf = nerv.ChunkFile(param_files[i], "r")
        for cid, cspec in pairs(pf.metadata) do
            if pids == nil or pids[cid] ~= nil then
                local p = pf:read_chunk(cid, gconf)
                if not nerv.is_type(p, "nerv.Param") then
                    nerv.error("param chunk is expected")
                end
                self:add(cid, p)
            end
        end
    end
end

function ParamRepo:export(param_file, pids)
    cf = nerv.ChunkFile(param_file, "w")
    if pids == nil then
        for id, p in pairs(self.params) do
            cf:write_chunk(p)
        end
    else
        for i, pid in ipairs(pids) do
            cf:write_chunk(self:get_param(pid))
        end
    end
    cf:close()
end

function ParamRepo:get_param(pid)
    local p = self.params[pid]
    if p == nil then
        nerv.error("param with id %s not found", pid)
    end
    return p
end
