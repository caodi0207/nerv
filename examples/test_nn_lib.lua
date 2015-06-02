-- require 'layer.affine'
-- require 'layer.sigmoid'
-- require 'layer.softmax_ce'

gconf = {lrate = 0.8, wcost = 1e-6, momentum = 0.9,
        mat_type = nerv.CuMatrixFloat,
        batch_size = 10}

param_repo = nerv.ParamRepo({"converted.nerv"})
sublayer_repo = nerv.LayerRepo(
    {
        ["nerv.AffineLayer"] =
        {
            affine0 = {{ltp = "affine0_ltp", bp = "affine0_bp"},
                        {dim_in = {429}, dim_out = {2048}}},
            affine1 = {{ltp = "affine1_ltp", bp = "affine1_bp"},
                        {dim_in = {2048}, dim_out = {2048}}},
            affine2 = {{ltp = "affine2_ltp", bp = "affine2_bp"},
                        {dim_in = {2048}, dim_out = {2048}}},
            affine3 = {{ltp = "affine3_ltp", bp = "affine3_bp"},
                        {dim_in = {2048}, dim_out = {2048}}},
            affine4 = {{ltp = "affine4_ltp", bp = "affine4_bp"},
                        {dim_in = {2048}, dim_out = {2048}}},
            affine5 = {{ltp = "affine5_ltp", bp = "affine5_bp"},
                        {dim_in = {2048}, dim_out = {2048}}},
            affine6 = {{ltp = "affine6_ltp", bp = "affine6_bp"},
                        {dim_in = {2048}, dim_out = {2048}}},
            affine7 = {{ltp = "affine7_ltp", bp = "affine7_bp"},
                        {dim_in = {2048}, dim_out = {3001}}}
        },
        ["nerv.SigmoidLayer"] =
        {
            sigmoid0 = {{}, {dim_in = {2048}, dim_out = {2048}}},
            sigmoid1 = {{}, {dim_in = {2048}, dim_out = {2048}}},
            sigmoid2 = {{}, {dim_in = {2048}, dim_out = {2048}}},
            sigmoid3 = {{}, {dim_in = {2048}, dim_out = {2048}}},
            sigmoid4 = {{}, {dim_in = {2048}, dim_out = {2048}}},
            sigmoid5 = {{}, {dim_in = {2048}, dim_out = {2048}}},
            sigmoid6 = {{}, {dim_in = {2048}, dim_out = {2048}}}
        },
        ["nerv.SoftmaxCELayer"] =
        {
            softmax_ce0 = {{}, {dim_in = {3001, 3001}, dim_out = {}}}
        }
    }, param_repo, gconf)

layer_repo = nerv.LayerRepo(
    {
        ["nerv.DAGLayer"] =
        {
            main = {{}, {
                dim_in = {429, 3001}, dim_out = {},
                sub_layers = sublayer_repo,
                connections = {
                    ["<input>[1]"] = "affine0[1]",
                    ["affine0[1]"] = "sigmoid0[1]",
                    ["sigmoid0[1]"] = "affine1[1]",
                    ["affine1[1]"] = "sigmoid1[1]",
                    ["sigmoid1[1]"] = "affine2[1]",
                    ["affine2[1]"] = "sigmoid2[1]",
                    ["sigmoid2[1]"] = "affine3[1]",
                    ["affine3[1]"] = "sigmoid3[1]",
                    ["sigmoid3[1]"] = "affine4[1]",
                    ["affine4[1]"] = "sigmoid4[1]",
                    ["sigmoid4[1]"] = "affine5[1]",
                    ["affine5[1]"] = "sigmoid5[1]",
                    ["sigmoid5[1]"] = "affine6[1]",
                    ["affine6[1]"] = "sigmoid6[1]",
                    ["sigmoid6[1]"] = "affine7[1]",
                    ["affine7[1]"] = "softmax_ce0[1]",
                    ["<input>[2]"] = "softmax_ce0[2]"
                }
            }}
        }
    }, param_repo, gconf)

df = nerv.ChunkFile("input.param", "r")
label = nerv.CuMatrixFloat(10, 3001)
label:fill(0)
for i = 0, 9 do
    label[i][i] = 1.0
end

input = {df:read_chunk("input", gconf).trans, label}
output = {}
err_input = {}
err_output = {input[1]:create()}
sm = sublayer_repo:get_layer("softmax_ce0")
main = layer_repo:get_layer("main")
main:init()
for i = 0, 3 do
    main:propagate(input, output)
    main:back_propagate(err_output, err_input, input, output)
    main:update(err_input, input, output)
    nerv.utils.printf("cross entropy: %.8f\n", sm.total_ce)
    nerv.utils.printf("frames: %.8f\n", sm.total_frames)
end
