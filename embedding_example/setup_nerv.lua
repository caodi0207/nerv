local k,l,_=pcall(require,"luarocks.loader") _=k and l.add_context("nerv","scm-1")
require 'nerv'
local arg = {...}
dofile(arg[1])
local param_repo = nerv.ParamRepo()
param_repo:import(gconf.initialized_param, nil, gconf)
local layer_repo = make_layer_repo(param_repo)
local network = get_decode_network(layer_repo)
local global_transf = get_global_transf(layer_repo)
local batch_size = 1
network:init(batch_size)

function propagator(input, output)
    local transformed = nerv.speech_utils.global_transf(input,
                            global_transf, 0, gconf) -- preprocessing
    local gpu_input = nerv.CuMatrixFloat(transformed:nrow(), transformed:ncol())
    local gpu_output = nerv.CuMatrixFloat(output:nrow(), output:ncol())
    print(transformed)
    gpu_input:copy_fromh(transformed)
    network:propagate({gpu_input}, {gpu_output})
    gpu_output:copy_toh(output)
    -- collect garbage in-time to save GPU memory
    collectgarbage("collect")
end

return network.dim_in[1], network.dim_out[1], propagator
