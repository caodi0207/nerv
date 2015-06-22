local LayerRepo = nerv.class("nerv.LayerRepo")

function LayerRepo:__init(layer_spec, param_repo, global_conf)
    local layers = {}
    for ltype, llist in pairs(layer_spec) do
        local layer_type = nerv.get_type(ltype)
        for id, spec in pairs(llist) do
            if layers[id] ~= nil then
                nerv.error("a layer with id %s already exists", id)
            end
            nerv.info("create layer: %s", id)
            if type(spec[2]) ~= "table" then
                nerv.error("layer config table is need")
            end
            layer_config = spec[2]
            if type(spec[1]) ~= "table" then
                nerv.error("parameter description table is needed")
            end
            for pname, pid in pairs(spec[1]) do
                layer_config[pname] = param_repo:get_param(pid)
            end
            layers[id] = layer_type(id, global_conf, layer_config)
        end
    end
    self.layers = layers
end

function LayerRepo:get_layer(lid)
    local layer = self.layers[lid]
    if layer == nil then
        nerv.error("layer with id %s not found", lid)
    end
    return layer
end
