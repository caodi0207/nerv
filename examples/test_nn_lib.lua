require 'speech.init'
gconf = {lrate = 0.8, wcost = 1e-6, momentum = 0.9,
        cumat_type = nerv.CuMatrixFloat,
        mmat_type = nerv.MMatrixFloat,
        batch_size = 256}

param_repo = nerv.ParamRepo({"converted.nerv", "global_transf.nerv"})
sublayer_repo = nerv.LayerRepo(
    {
        -- global transf
        ["nerv.BiasLayer"] =
        {
            blayer1 = {{bias = "bias1"}, {dim_in = {429}, dim_out = {429}}},
            blayer2 = {{bias = "bias2"}, {dim_in = {429}, dim_out = {429}}}
        },
        ["nerv.WindowLayer"] =
        {
            wlayer1 = {{window = "window1"}, {dim_in = {429}, dim_out = {429}}},
            wlayer2 = {{window = "window2"}, {dim_in = {429}, dim_out = {429}}}
        },
        -- biased linearity
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
            softmax_ce0 = {{}, {dim_in = {3001, 1}, dim_out = {}, compressed = true}}
        }
    }, param_repo, gconf)

layer_repo = nerv.LayerRepo(
    {
        ["nerv.DAGLayer"] =
        {
            global_transf = {{}, {
                dim_in = {429}, dim_out = {429},
                sub_layers = sublayer_repo,
                connections = {
                    ["<input>[1]"] = "blayer1[1]",
                    ["blayer1[1]"] = "wlayer1[1]",
                    ["wlayer1[1]"] = "blayer2[1]",
                    ["blayer2[1]"] = "wlayer2[1]",
                    ["wlayer2[1]"] = "<output>[1]"
                }
            }},
            main = {{}, {
                dim_in = {429, 1}, dim_out = {},
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

tnet_reader = nerv.TNetReader(gconf,
    {
        id = "main_scp",
        scp_file = "/slfs1/users/mfy43/swb_ivec/train_bp.scp",
--        scp_file = "t.scp",
        conf_file = "/slfs1/users/mfy43/swb_ivec/plp_0_d_a.conf",
        frm_ext = 5,
        mlfs = {
            ref = {
                file = "/slfs1/users/mfy43/swb_ivec/ref.mlf",
                format = "map",
                format_arg = "/slfs1/users/mfy43/swb_ivec/dict",
                dir = "*/",
                ext = "lab"
            }
        },
        global_transf = layer_repo:get_layer("global_transf")
    })

buffer = nerv.SGDBuffer(gconf,
    {
        buffer_size = 81920,
        randomize = true,
        readers = {
            { reader = tnet_reader,
              data = {main_scp = 429, ref = 1}}
        }
    })

sm = sublayer_repo:get_layer("softmax_ce0")
main = layer_repo:get_layer("main")
main:init(gconf.batch_size)
gconf.cnt = 0
-- data = buffer:get_data()
-- input = {data.main_scp, data.ref}
-- while true do
for data in buffer.get_data, buffer do
--    if gconf.cnt == 100 then break end
--    gconf.cnt = gconf.cnt + 1

    input = {data.main_scp, data.ref}
    output = {}
    err_input = {}
    err_output = {input[1]:create()}
    
    main:propagate(input, output)
    main:back_propagate(err_output, err_input, input, output)
    main:update(err_input, input, output)

--    nerv.printf("cross entropy: %.8f\n", sm.total_ce)
--    nerv.printf("correct: %d\n", sm.total_correct)
--    nerv.printf("frames: %d\n", sm.total_frames)
--    nerv.printf("err/frm: %.8f\n", sm.total_ce / sm.total_frames)
--    nerv.printf("accuracy: %.8f\n", sm.total_correct / sm.total_frames)
    collectgarbage("collect")
end
nerv.printf("cross entropy: %.8f\n", sm.total_ce)
nerv.printf("correct: %d\n", sm.total_correct)
nerv.printf("accuracy: %.3f%%\n", sm.total_correct / sm.total_frames * 100)
nerv.printf("writing back...\n")
cf = nerv.ChunkFile("output.nerv", "w")
for i, p in ipairs(main:get_params()) do
    print(p)
    cf:write_chunk(p)
end
cf:close()
nerv.Matrix.print_profile()
