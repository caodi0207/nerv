require 'lmptb.lmvocab'
require 'lmptb.lmfeeder'
require 'lmptb.lmutil'
nerv.include('lmptb/layer/init.lua')

--[[global function rename]]--
printf = nerv.printf
--[[global function rename ends]]--

--global_conf: table
--first_time: bool
--Returns: a ParamRepo
function prepare_parameters(global_conf, first_time)
    printf("%s preparing parameters...\n", global_conf.sche_log_pre) 
    
    if (first_time) then
        ltp_ih = nerv.LinearTransParam("ltp_ih", global_conf)  
        ltp_ih.trans = global_conf.cumat_type(global_conf.vocab:size(), global_conf.hidden_size)  
        ltp_ih.trans:generate(global_conf.param_random)

        ltp_hh = nerv.LinearTransParam("ltp_hh", global_conf)
        ltp_hh.trans = global_conf.cumat_type(global_conf.hidden_size, global_conf.hidden_size)
        ltp_hh.trans:generate(global_conf.param_random) 

        ltp_ho = nerv.LinearTransParam("ltp_ho", global_conf)
        ltp_ho.trans = global_conf.cumat_type(global_conf.hidden_size, global_conf.vocab:size())
        ltp_ho.trans:generate(global_conf.param_random)

        bp_h = nerv.BiasParam("bp_h", global_conf)
        bp_h.trans = global_conf.cumat_type(1, global_conf.hidden_size)
        bp_h.trans:generate(global_conf.param_random)

        bp_o = nerv.BiasParam("bp_o", global_conf)
        bp_o.trans = global_conf.cumat_type(1, global_conf.vocab:size())
        bp_o.trans:generate(global_conf.param_random)

        local f = nerv.ChunkFile(global_conf.param_fn, 'w')
        f:write_chunk(ltp_ih)
        f:write_chunk(ltp_hh)
        f:write_chunk(ltp_ho)
        f:write_chunk(bp_h)
        f:write_chunk(bp_o)
        f:close()
    end
    
    local paramRepo = nerv.ParamRepo()
    paramRepo:import({global_conf.param_fn}, nil, global_conf)

    printf("%s preparing parameters end.\n", global_conf.sche_log_pre)

    return paramRepo
end

--global_conf: table
--Returns: nerv.LayerRepo
function prepare_layers(global_conf, paramRepo)
    printf("%s preparing layers...\n", global_conf.sche_log_pre)
    local recurrentLconfig = {{["bp"] = "bp_h", ["ltp_ih"] = "ltp_ih", ["ltp_hh"] = "ltp_hh"}, {["dim_in"] = {global_conf.vocab:size(), global_conf.hidden_size}, ["dim_out"] = {global_conf.hidden_size}, ["break_id"] = global_conf.vocab:get_sen_entry().id, ["independent"] = global_conf.independent}}
    local layers = {
        ["nerv.LMAffineRecurrentLayer"] = {
            ["recurrentL1"] = recurrentLconfig, 
        },

        ["nerv.SigmoidLayer"] = {
            ["sigmoidL1"] = {{}, {["dim_in"] = {global_conf.hidden_size}, ["dim_out"] = {global_conf.hidden_size}}}
        },
        
        ["nerv.AffineLayer"] = {
            ["outputL"] = {{["ltp"] = "ltp_ho", ["bp"] = "bp_o"}, {["dim_in"] = {global_conf.hidden_size}, ["dim_out"] = {global_conf.vocab:size()}}},
        },

        ["nerv.SoftmaxCELayer"] = {
            ["softmaxL"] = {{}, {["dim_in"] = {global_conf.vocab:size(), global_conf.vocab:size()}, ["dim_out"] = {1}}},
        },
    }
    
    printf("%s adding %d bptt layers...\n", global_conf.sche_log_pre, global_conf.bptt)
    for i = 1, global_conf.bptt do
        layers["nerv.LMAffineRecurrentLayer"]["recurrentL" .. (i + 1)] = recurrentLconfig 
        layers["nerv.SigmoidLayer"]["sigmoidL" .. (i + 1)] = {{}, {["dim_in"] = {global_conf.hidden_size}, ["dim_out"] = {global_conf.hidden_size}}}
    end
    local layerRepo = nerv.LayerRepo(layers, paramRepo, global_conf)
    printf("%s preparing layers end.\n", global_conf.sche_log_pre)
    return layerRepo
end

--global_conf: table
--layerRepo: nerv.LayerRepo
--Returns: a nerv.DAGLayer
function prepare_dagLayer(global_conf, layerRepo)
    printf("%s Initing daglayer ...\n", global_conf.sche_log_pre)

    --input: input_w, input_w, ... input_w_now, last_activation
    local dim_in_t = {}
    for i = 1, global_conf.bptt + 1 do dim_in_t[i] = global_conf.vocab:size() end
    dim_in_t[global_conf.bptt + 2] = global_conf.hidden_size
    dim_in_t[global_conf.bptt + 3] = global_conf.vocab:size()
       --[[                            softmax     
                                          |      \
                                        ouptut  i(bptt+3)
                                          |
    recurrentL(bptt+1)... recurrentL2-recurrentL1
       /    |                 |           |
 i(bptt+2) i(bptt+1)          i2         i1
    ]]--
    local connections_t = {
                ["recurrentL1[1]"] = "sigmoidL1[1]",
                ["sigmoidL1[1]"] = "outputL[1]",
                ["outputL[1]"] = "softmaxL[1]",
                ["softmaxL[1]"] = "<output>[1]"
    }
    for i = 1, global_conf.bptt, 1 do
        connections_t["<input>["..i.."]"] = "recurrentL"..i.."[1]"
        connections_t["recurrentL"..(i+1).."[1]"] = "sigmoidL"..(i+1).."[1]"
        connections_t["sigmoidL"..(i+1).."[1]"] = "recurrentL"..i.."[2]"
    end
    connections_t["<input>["..(global_conf.bptt+1).."]"] = "recurrentL"..(global_conf.bptt+1).."[1]"
    connections_t["<input>["..(global_conf.bptt+2).."]"] = "recurrentL"..(global_conf.bptt+1).."[2]"
    connections_t["<input>["..(global_conf.bptt+3).."]"] = "softmaxL[2]"
    printf("%s printing DAG connections:\n", global_conf.sche_log_pre)
    for key, value in pairs(connections_t) do
        printf("\t%s->%s\n", key, value)
    end

    local dagL = nerv.DAGLayer("dagL", global_conf, {["dim_in"] = dim_in_t, ["dim_out"] = {1}, ["sub_layers"] = layerRepo,
            ["connections"] = connections_t, 
        })
    dagL:init(global_conf.batch_size)
    printf("%s Initing DAGLayer end.\n", global_conf.sche_log_pre)
    return dagL
end

--Returns: table
function create_dag_input(global_conf, token_store, hidden_store, tnow)
    local dagL_input = {}
    for i = 1, global_conf.bptt + 1 do
        dagL_input[i] = nerv.LMUtil.create_onehot(token_store[tnow - i + 1], global_conf.vocab, global_conf.cumat_type)
    end
    dagL_input[global_conf.bptt + 2] = hidden_store[tnow - global_conf.bptt - 1]
    dagL_input[global_conf.bptt + 3] = nerv.LMUtil.create_onehot(token_store[tnow + 1], global_conf.vocab, global_conf.cumat_type)
 
    return dagL_input
end

--global_conf: table
--dagL: nerv.DAGLayer
--fn: string
--config: table
--Returns: table, result
function propagateFile(global_conf, dagL, fn, config)
    printf("%s Begining doing on %s...\n", global_conf.sche_log_pre, fn)
    if (config.do_train == true) then printf("%s do_train in config is true.\n", global_conf.sche_log_pre) end
    local feeder = nerv.LMFeeder(global_conf, global_conf.batch_size, global_conf.vocab)
    feeder:open_file(fn)

    local tnow = 1
    local token_store = {}
    local hidden_store = {}
    local sigmoidL_ref = dagL.layers["sigmoidL1"]

    token_store[tnow] = feeder:get_batch()
    for i = 1, global_conf.bptt + 1 do
        hidden_store[tnow - i] = global_conf.cumat_type(global_conf.batch_size, global_conf.hidden_size)
        hidden_store[tnow - i]:fill(0)
        token_store[tnow - i] = {}
        for j = 1, global_conf.batch_size do token_store[tnow - i][j] = global_conf.vocab.null_token end
    end

    local dagL_output = {global_conf.cumat_type(global_conf.batch_size, 1)}
    local dagL_err = {nil} --{global_conf.cumat_type(global_conf.batch_size, 1)}
    local dagL_input_err = {}
    for i = 1, global_conf.bptt + 1 do
        dagL_input_err[i] = global_conf.cumat_type(global_conf.batch_size, global_conf.vocab:size())
    end
    dagL_input_err[global_conf.bptt + 2] = global_conf.cumat_type(global_conf.batch_size, global_conf.hidden_size)
    dagL_input_err[global_conf.bptt + 3] = global_conf.cumat_type(global_conf.batch_size, global_conf.vocab:size())

    local result = nerv.LMResult(global_conf, global_conf.vocab)
    result:init("rnn")

    while (1) do
        token_store[tnow + 1] = feeder:get_batch() --The next word(to predict)
        if (token_store[tnow + 1] == nil) then break end

        local dagL_input = create_dag_input(global_conf, token_store, hidden_store, tnow)
        --dagL:propagate(dagL_input, dagL_output)

        hidden_store[tnow] = global_conf.cumat_type(global_conf.batch_size, global_conf.hidden_size)
        hidden_store[tnow]:copy_fromd(sigmoidL_ref.outputs[1])

        if (config.do_train == true) then
            --dagL:back_propagate(dagL_err, dagL_input_err, dagL_input, dagL_output)
            --dagL:update(dagL_err, dagL_input, dagL_output)
        end
        
        for i = 1, global_conf.batch_size, 1 do
            if (token_store[tnow + 1][i] ~= global_conf.vocab.null_token) then
                result:add("rnn", token_store[tnow + 1][i], math.exp(dagL_output[1][i - 1][0]))
                if (config.report_word == true) then
                    printf("%s %s: <stream %d> <prob %f>\n", global_conf.sche_log_pre, token_store[tnow + 1][i], i, math.exp(dagL_output[1][i - 1][0]))
                end
            end
            if (result["rnn"].cn_w % global_conf.log_w_num == 0) then
                printf("%s %d words processed.\n", global_conf.sche_log_pre, result["rnn"].cn_w) 
            end
        end
 
        token_store[tnow - 2 - global_conf.bptt] = nil
        hidden_store[tnow - 2 - global_conf.bptt] = nil
        collectgarbage("collect")                                                                                                                                                               
        tnow = tnow + 1
    end

    printf("%s Displaying result:\n", global_conf.sche_log_pre)
    printf("%s %s\n", global_conf.sche_log_pre, result:status("rnn"))
    printf("%s Doing on %s end.\n", global_conf.sche_log_pre, fn)

    return result
end

--returns dagL, paramRepo
function load_net(global_conf)
    local paramRepo = prepare_parameters(global_conf, false)
    local layerRepo = prepare_layers(global_conf, paramRepo) 
    local dagL = prepare_dagLayer(global_conf, layerRepo)
    return dagL, paramRepo
end

--[[global settings]]--
local set = "ptb"

if (set == "ptb") then
    train_fn = "/home/slhome/txh18/workspace/nerv-project/nerv/examples/lmptb/PTBdata/ptb.train.txt"
    valid_fn = "/home/slhome/txh18/workspace/nerv-project/nerv/examples/lmptb/PTBdata/ptb.valid.txt"
    test_fn = "/home/slhome/txh18/workspace/nerv-project/nerv/examples/lmptb/PTBdata/ptb.test.txt"
    work_dir_base = "/home/slhome/txh18/workspace/nerv-project/lmptb-work"
    global_conf = {
        lrate = 0.1, wcost = 1e-6, momentum = 0,
        cumat_type = nerv.CuMatrixFloat,
        mmat_type = nerv.CuMatrixFloat,

        hidden_size = 100,
        batch_size = 10,
        bptt = 3, --train bptt_block's words. could be set to zero
        max_iter = 15,
        param_random = function() return (math.random() / 5 - 0.1) end,
        independent = true,

        train_fn = train_fn,
        valid_fn = valid_fn,
        test_fn = test_fn,
        sche_log_pre = "[SCHEDULER]:",
        log_w_num = 100000, --give a message when log_w_num words have been processed
    }
    global_conf.work_dir = work_dir_base.."/h"..global_conf.hidden_size.."bp"..global_conf.bptt.."slr"..global_conf.lrate..os.date("_%bD%dH%H")
    global_conf.param_fn = global_conf.work_dir.."/params"
elseif (set == "test") then
    train_fn = "/home/slhome/txh18/workspace/nerv-project/some-text"
    valid_fn = "/home/slhome/txh18/workspace/nerv-project/some-text"
    test_fn = "/home/slhome/txh18/workspace/nerv-project/some-text"
    work_dir = "/home/slhome/txh18/workspace/nerv-project/lmptb-work-play"
    global_conf = {
        lrate = 0.1, wcost = 1e-6, momentum = 0,
        cumat_type = nerv.CuMatrixFloat,
        mmat_type = nerv.CuMatrixFloat,

        hidden_size = 5,
        batch_size = 1,
        bptt = 1, --train bptt_block's words. could be set to zero
        max_iter = 15,
        param_random = function() return (math.random() / 5 - 0.1) end,
        independent = true,

        train_fn = train_fn,
        valid_fn = valid_fn,
        test_fn = test_fn,
        work_dir = work_dir,
        param_fn = work_dir .. "/params",

        sche_log_pre = "[SCHEDULER]:",
        log_w_num = 80000, --give a message when log_w_num words have been processed
    }
end

local vocab = nerv.LMVocab()
global_conf["vocab"] = vocab

printf("%s printing global_conf...\n", global_conf.sche_log_pre)
for key, value in pairs(global_conf) do
    printf("\t%s=%s\n", key, value)
end
printf("%s wait 3 seconds...\n", global_conf.sche_log_pre)
nerv.LMUtil.wait(3)
printf("%s creating work_dir...\n", global_conf.sche_log_pre)
os.execute("mkdir -p "..global_conf.work_dir)

scheduler = "   printf(\"===INITIAL VALIDATION===\\n\") \
    dagL, paramRepo = load_net(global_conf) \
    local result = propagateFile(global_conf, dagL, global_conf.valid_fn, {do_train = false, report_word = false}) \
    ppl_rec = {} \
    ppl_rec[0] = result:ppl_net(\"rnn\")  ppl_last = ppl_rec[0] \
    printf(\"\\n\") \
    for iter = 1, global_conf.max_iter, 1 do \
        printf(\"===ITERATION %d LR %f===\\n\", iter, global_conf.lrate) \
        global_conf.sche_log_pre = \"[SCHEDULER ITER\"..iter..\" LR\"..global_conf.lrate..\"]:\" \
        dagL, paramRepo = load_net(global_conf) \
        propagateFile(global_conf, dagL, global_conf.train_fn, {do_train = true, report_word = false}) \
        printf(\"===VALIDATION %d===\\n\", iter) \
        local result = propagateFile(global_conf, dagL, global_conf.valid_fn, {do_train = false, report_word = false}) \
        ppl_rec[iter] = result:ppl_net(\"rnn\") \
        if (ppl_last / ppl_rec[iter] < 1.03) then \
            global_conf.lrate = (global_conf.lrate / 2) \
        end \
        if (ppl_rec[iter] < ppl_last) then \
            printf(\"%s saving net to file %s...\\n\", global_conf.sche_log_pre, global_conf.param_fn) \
            paramRepo:export(global_conf.param_fn, nil) \
            ppl_last = ppl_rec[iter] \
        else \
            printf(\"%s PPL did not improve, rejected...\\n\", global_conf.sche_log_pre) \
        end \
        printf(\"\\n\") \
        nerv.LMUtil.wait(2) \
    end \
    printf(\"===VALIDATION PPL record===\\n\") \
    for i = 0, #ppl_rec do printf(\"<%d: %.2f> \", i, ppl_rec[i]) end \
    printf(\"\\n\") \
    printf(\"===FINAL TEST===\\n\") \
    global_conf.sche_log_pre = \"[SCHEDULER FINAL_TEST]:\" \
    dagL, _ = load_net(global_conf) \
    propagateFile(global_conf, dagL, global_conf.test_fn, {do_train = false, report_word = false})"
printf("%s printing schedule:\n", global_conf.sche_log_pre)
printf("%s\n", scheduler)
printf("%s wait 3 seconds...\n", global_conf.sche_log_pre)
nerv.LMUtil.wait(3)
--[[global settings end]]--

global_conf.vocab:build_file(global_conf.train_fn)

prepare_parameters(global_conf, true)

assert(loadstring(scheduler))()
