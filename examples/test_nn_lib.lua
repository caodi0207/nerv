require 'layer.affine'
require 'layer.sigmoid'
require 'layer.softmax_ce'

gconf = {lrate = 0.8, wcost = 1e-6, momentum = 0.9,
        mat_type = nerv.CuMatrixFloat,
        batch_size = 10}

param_repo = nerv.ParamRepo({"affine.param"})
sublayer_repo = nerv.LayerRepo(
    {
        ["nerv.AffineLayer"] =
        {
            affine1 = {{ltp = "a", bp = "b"}, {dim_in = {429}, dim_out = {2048}}}
        },
        ["nerv.SigmoidLayer"] =
        {
            sigmoid1 = {{}, {dim_in = {2048}, dim_out = {2048}}}
        },
        ["nerv.SoftmaxCELayer"] =
        {
            softmax_ce1 = {{}, {dim_in = {2048, 2048}, dim_out = {}}}
        }
    }, param_repo, gconf)

layer_repo = nerv.LayerRepo(
    {
        ["nerv.DAGLayer"] =
        {
            main = {{}, {
                dim_in = {429, 2048}, dim_out = {},
                sub_layers = sublayer_repo,
                connections = {
                    ["<input>[1]"] = "affine1[1]",
                    ["affine1[1]"] = "sigmoid1[1]",
                    ["sigmoid1[1]"] = "softmax_ce1[1]",
                    ["<input>[2]"] = "softmax_ce1[2]"
                }
            }}
        }
    }, param_repo, gconf)

df = nerv.ChunkFile("input.param", "r")
label = nerv.CuMatrixFloat(10, 2048)
label:fill(0)
for i = 0, 9 do
    label[i][i] = 1.0
end

input = {df:read_chunk("input", gconf).trans, label}
output = {}
err_input = {}
err_output = {input[1]:create()}
sm = sublayer_repo:get_layer("softmax_ce1")
main = layer_repo:get_layer("main")
main:init()
for i = 0, 3 do
    main:propagate(input, output)
    main:back_propagate(err_output, err_input, input, output)
    main:update(err_input, input, output)
    nerv.utils.printf("cross entropy: %.8f\n", sm.total_ce)
    nerv.utils.printf("frames: %.8f\n", sm.total_frames)
end
