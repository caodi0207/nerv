require 'htk_io'
gconf = {lrate = 0.8, wcost = 1e-6, momentum = 0.9,
        cumat_type = nerv.CuMatrixFloat,
        mmat_type = nerv.MMatrixFloat,
        frm_ext = 5,
        tr_scp = "/slfs1/users/mfy43/swb_ivec/train_bp.scp",
        cv_scp = "/slfs1/users/mfy43/swb_ivec/train_cv.scp",
        htk_conf = "/slfs1/users/mfy43/swb_ivec/plp_0_d_a.conf",
        initialized_param = {"/slfs1/users/mfy43/swb_init.nerv",
                "/slfs1/users/mfy43/swb_global_transf.nerv"},
        debug = false}

function make_layer_repo(param_repo)
    local layer_repo = nerv.LayerRepo(
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
        ["nerv.SoftmaxCELayer"] = -- softmax + ce criterion layer for finetune output
        {
            ce_crit = {{}, {dim_in = {3001, 1}, dim_out = {1}, compressed = true}}
        },
        ["nerv.SoftmaxLayer"] = -- softmax for decode output
        {
            softmax = {{}, {dim_in = {3001}, dim_out = {3001}}}
        }
    }, param_repo, gconf)

    layer_repo:add_layers(
    {
        ["nerv.DAGLayer"] =
        {
            global_transf = {{}, {
                dim_in = {429}, dim_out = {429},
                sub_layers = layer_repo,
                connections = {
                    ["<input>[1]"] = "blayer1[1]",
                    ["blayer1[1]"] = "wlayer1[1]",
                    ["wlayer1[1]"] = "blayer2[1]",
                    ["blayer2[1]"] = "wlayer2[1]",
                    ["wlayer2[1]"] = "<output>[1]"
                }
            }},
            main = {{}, {
                dim_in = {429}, dim_out = {3001},
                sub_layers = layer_repo,
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
                    ["affine7[1]"] = "<output>[1]"
                }
            }}
        }
    }, param_repo, gconf)

    layer_repo:add_layers(
    {
        ["nerv.DAGLayer"] =
        {
            ce_output = {{}, {
                dim_in = {429, 1}, dim_out = {1},
                sub_layers = layer_repo,
                connections = {
                    ["<input>[1]"] = "main[1]",
                    ["main[1]"] = "ce_crit[1]",
                    ["<input>[2]"] = "ce_crit[2]",
                    ["ce_crit[1]"] = "<output>[1]"
                }
            }},
            softmax_output = {{}, {
                dim_in = {429}, dim_out = {3001},
                sub_layers = layer_repo,
                connections = {
                    ["<input>[1]"] = "main[1]",
                    ["main[1]"] = "softmax[1]",
                    ["softmax[1]"] = "<output>[1]"
                }
            }}
        }
    }, param_repo, gconf)

    return layer_repo
end

function get_network(layer_repo)
    return layer_repo:get_layer("ce_output")
end

function get_decode_network(layer_repo)
    return layer_repo:get_layer("softmax_output")
end

function get_global_transf(layer_repo)
    return layer_repo:get_layer("global_transf")
end

function make_readers(scp_file, layer_repo)
    return {
                {reader = nerv.TNetReader(gconf,
                    {
                        id = "main_scp",
                        scp_file = scp_file,
                        conf_file = gconf.htk_conf,
                        frm_ext = gconf.frm_ext,
                        mlfs = {
                            phone_state = {
                                file = "/slfs1/users/mfy43/swb_ivec/ref.mlf",
                                format = "map",
                                format_arg = "/slfs1/users/mfy43/swb_ivec/dict",
                                dir = "*/",
                                ext = "lab"
                            }
                        },
                        global_transf = layer_repo:get_layer("global_transf")
                    }),
                data = {main_scp = 429, phone_state = 1}}
            }
end

function make_buffer(readers)
    return nerv.SGDBuffer(gconf,
        {
            buffer_size = gconf.buffer_size,
            randomize = gconf.randomize,
            readers = readers
        })
end

function get_input_order()
    return {"main_scp", "phone_state"}
end

function get_accuracy(layer_repo)
    local ce_crit = layer_repo:get_layer("ce_crit")
    return ce_crit.total_correct / ce_crit.total_frames * 100
end

function print_stat(layer_repo)
    local ce_crit = layer_repo:get_layer("ce_crit")
    nerv.info("*** training stat begin ***")
    nerv.printf("cross entropy:\t\t%.8f\n", ce_crit.total_ce)
    nerv.printf("correct:\t\t%d\n", ce_crit.total_correct)
    nerv.printf("frames:\t\t\t%d\n", ce_crit.total_frames)
    nerv.printf("err/frm:\t\t%.8f\n", ce_crit.total_ce / ce_crit.total_frames)
    nerv.printf("accuracy:\t\t%.3f%%\n", get_accuracy(layer_repo))
    nerv.info("*** training stat end ***")
end
