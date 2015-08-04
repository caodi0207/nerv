local DAGLayer = nerv.class("nerv.DAGLayer", "nerv.Layer")

local function parse_id(str)
    local id, port, _
    _, _, id, port = string.find(str, "([a-zA-Z0-9_]+)%[([0-9]+)%]")
    if id == nil or port == nil then
        _, _, id, port = string.find(str, "(.+)%[([0-9]+)%]")
        if not (id == "<input>" or id == "<output>") then
            nerv.error("wrong format of connection id")
        end
    end
    port = tonumber(port)
    return id, port
end

local function discover(id, layers, layer_repo)
    local ref = layers[id]
    if id == "<input>" or id == "<output>" then
        return nil
    end
    if ref == nil then
        local layer = layer_repo:get_layer(id)
        local dim_in, dim_out = layer:get_dim()
        ref = {
            layer = layer,
            inputs = {},
            outputs = {},
            err_inputs = {},
            err_outputs = {},
            next_layers = {},
            input_len = #dim_in,
            output_len = #dim_out,
            in_deg = 0,
            visited = false
        }
        layers[id] = ref
    end
    return ref
end

function DAGLayer:__init(id, global_conf, layer_conf)
    local layers = {}
    local inputs = {}
    local outputs = {}
    local dim_in = layer_conf.dim_in
    local dim_out = layer_conf.dim_out
    local parsed_conn = {}
    for from, to in pairs(layer_conf.connections) do
        local id_from, port_from = parse_id(from)
        local id_to, port_to = parse_id(to)
        local ref_from = discover(id_from, layers, layer_conf.sub_layers)
        local ref_to = discover(id_to, layers, layer_conf.sub_layers)
        local input_dim, output_dim, _
        if ref_from and ref_from.outputs[port_from] ~= nil then
            nerv.error("%s has already been attached", from)
        end
        if ref_to and ref_to.inputs[port_to] ~= nil then
            nerv.error("%s has already been attached", to)
        end
        if id_from == "<input>" then
            input_dim, _ = ref_to.layer:get_dim()
            if dim_in[port_from] ~= input_dim[port_to] then
                nerv.error("mismatching data dimension between %s and %s", from, to)
            end
            inputs[port_from] = {ref_to, port_to}
            ref_to.inputs[port_to] = inputs -- just a place holder
        elseif id_to == "<output>" then
            _, output_dim = ref_from.layer:get_dim()
            if output_dim[port_from] ~= dim_out[port_to] then
                nerv.error("mismatching data dimension between %s and %s", from, to)
            end
            outputs[port_to] = {ref_from, port_from}
            ref_from.outputs[port_from] = outputs -- just a place holder
        else
            _, output_dim = ref_from.layer:get_dim()
            input_dim, _ = ref_to.layer:get_dim()
            if output_dim[port_from] ~= input_dim[port_to] then
                nerv.error("mismatching data dimension between %s and %s", from, to)
            end

            table.insert(parsed_conn,
                {{ref_from, port_from}, {ref_to, port_to}})
            table.insert(ref_from.next_layers, ref_to) -- add edge
            ref_to.in_deg = ref_to.in_deg + 1          -- increase the in-degree of the target layer
        end
    end

    -- topology sort
    local queue = {}
    local l = 1
    local r = 1
    for id, ref in pairs(layers) do
        if ref.in_deg == 0 then
            table.insert(queue, ref)
            nerv.info("adding source layer: %s", id)
            r = r + 1
        end
    end
    if l == r then
        nerv.error("loop detected")
    end
    while l < r do
        local cur = queue[l]
        cur.visited = true
        l = l + 1
        for _, nl in pairs(cur.next_layers) do
            nl.in_deg = nl.in_deg - 1 
            if nl.in_deg == 0 then
                table.insert(queue, nl)
                r = r + 1
            end
        end
    end
    for i = 1, #queue do
        nerv.info("enqueued layer: %s", queue[i].layer.id)
    end

    for id, ref in pairs(layers) do
        -- check wether the graph is connected
        if ref.visited == false then
            nerv.warning("layer %s is ignored", id)
        end
    end

    self.layers = layers
    self.inputs = inputs
    self.outputs = outputs
    self.dim_in = dim_in
    self.dim_out = dim_out
    self.parsed_conn = parsed_conn
    self.queue = queue
    self.gconf = global_conf
end

function DAGLayer:init(batch_size)
    for i, conn in ipairs(self.parsed_conn) do
        local _, output_dim
        local ref_from, port_from, ref_to, port_to
        ref_from, port_from = unpack(conn[1])
        ref_to, port_to = unpack(conn[2])
        _, output_dim = ref_from.layer:get_dim()
        local mid = self.gconf.cumat_type(batch_size,
                                        output_dim[port_from])
        local err_mid = mid:create()

        ref_from.outputs[port_from] = mid
        ref_to.inputs[port_to] = mid

        ref_from.err_inputs[port_from] = err_mid
        ref_to.err_outputs[port_to] = err_mid
    end
    for id, ref in pairs(self.layers) do
        for i = 1, ref.input_len do
            if ref.inputs[i] == nil then
                nerv.error("dangling input port %d of layer %s", i, id)
            end
        end
        for i = 1, ref.output_len do
            if ref.outputs[i] == nil then
                nerv.error("dangling output port %d of layer %s", i, id)
            end
        end
        -- initialize sub layers
        ref.layer:init(batch_size)
    end
    for i = 1, #self.dim_in do
        if self.inputs[i] == nil then
            nerv.error("dangling port %d of layer <input>", i)
        end
    end
    for i = 1, #self.dim_out do
        if self.outputs[i] == nil then
            nerv.error("dangling port %d of layer <output>", i)
        end
    end
end

function DAGLayer:set_inputs(input)
    for i = 1, #self.dim_in do
        if input[i] == nil then
            nerv.error("some input is not provided");
        end
        local layer = self.inputs[i][1]
        local port = self.inputs[i][2]
        layer.inputs[port] = input[i]
    end
end

function DAGLayer:set_outputs(output)
    for i = 1, #self.dim_out do
        if output[i] == nil then
            nerv.error("some output is not provided");
        end
        local layer = self.outputs[i][1]
        local port = self.outputs[i][2]
        layer.outputs[port] = output[i]
    end
end

function DAGLayer:set_err_inputs(bp_err)
    for i = 1, #self.dim_out do
        local layer = self.outputs[i][1]
        local port = self.outputs[i][2]
        layer.err_inputs[port] = bp_err[i]
    end
end

function DAGLayer:set_err_outputs(next_bp_err)
    for i = 1, #self.dim_in do
        local layer = self.inputs[i][1]
        local port = self.inputs[i][2]
        layer.err_outputs[port] = next_bp_err[i]
    end
end

function DAGLayer:update(bp_err, input, output)
    self:set_err_inputs(bp_err)
    self:set_inputs(input)
    self:set_outputs(output)
    -- print("update")
    for id, ref in pairs(self.queue) do
        -- print(ref.layer.id)
        ref.layer:update(ref.err_inputs, ref.inputs, ref.outputs)
    end
end

function DAGLayer:propagate(input, output)
    self:set_inputs(input)
    self:set_outputs(output)
    for i = 1, #self.queue do
        local ref = self.queue[i]
        -- print(ref.layer.id)
        ref.layer:propagate(ref.inputs, ref.outputs)
    end
end

function DAGLayer:back_propagate(bp_err, next_bp_err, input, output)
    self:set_err_outputs(next_bp_err)
    self:set_err_inputs(bp_err)
    self:set_inputs(input)
    self:set_outputs(output)
    for i = #self.queue, 1, -1 do
        local ref = self.queue[i]
        -- print(ref.layer.id)
        ref.layer:back_propagate(ref.err_inputs, ref.err_outputs, ref.inputs, ref.outputs)
    end
end

function DAGLayer:get_params()
    local param_repos = {}
    for id, ref in pairs(self.queue) do
        table.insert(param_repos, ref.layer:get_params())
    end
    return nerv.ParamRepo.merge(param_repos)
end
