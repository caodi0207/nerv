local k,l,_=pcall(require,"luarocks.loader") _=k and l.add_context("nerv","scm-1")
require 'nerv'
local arg = {...}
dofile(arg[1])
local param_repo = nerv.ParamRepo()
param_repo:import(gconf.initialized_param, nil, gconf)
local sublayer_repo = make_sublayer_repo(param_repo)
local layer_repo = make_layer_repo(sublayer_repo, param_repo)
local network = get_network(layer_repo)
local batch_size = 1
network:init(batch_size)

function propagator(input, output)
    local gpu_input = nerv.CuMatrixFloat(input:nrow(), input:ncol())
    local gpu_output = nerv.CuMatrixFloat(output:nrow(), output:ncol())
    gpu_input:copy_fromh(input)
    print(gpu_input)
    network:propagate({gpu_input}, {gpu_output})
    gpu_output:copy_toh(output)
    print(output)
    -- collect garbage in-time to save GPU memory
    collectgarbage("collect")
end

return network.dim_in[1], network.dim_out[1], propagator