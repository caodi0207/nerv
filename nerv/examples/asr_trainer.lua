function build_trainer(ifname)
    local param_repo = nerv.ParamRepo()
    param_repo:import(ifname, nil, gconf)
    local layer_repo = make_layer_repo(param_repo)
    local network = get_network(layer_repo)
    local input_order = get_input_order()
    local iterative_trainer = function (prefix, scp_file, bp)
        gconf.randomize = bp
        -- build buffer
        local buffer = make_buffer(make_readers(scp_file, layer_repo))
        -- initialize the network
        network:init(gconf.batch_size)
        gconf.cnt = 0
        err_input = {nerv.CuMatrixFloat(gconf.batch_size, 1)}
        err_input[1]:fill(1)
        for data in buffer.get_data, buffer do
            -- prine stat periodically
            gconf.cnt = gconf.cnt + 1
            if gconf.cnt == 1000 then
                print_stat(layer_repo)
                nerv.CuMatrix.print_profile()
                nerv.CuMatrix.clear_profile()
                gconf.cnt = 0
                -- break
            end
            local input = {}
--            if gconf.cnt == 100 then break end
            for i, id in ipairs(input_order) do
                if data[id] == nil then
                    nerv.error("input data %s not found", id)
                end
                table.insert(input, data[id])
            end
            local output = {nerv.CuMatrixFloat(gconf.batch_size, 1)}
            err_output = {input[1]:create()}
            network:propagate(input, output)
            if bp then
                network:back_propagate(err_input, err_output, input, output)
                network:update(err_input, input, output)
            end
            -- collect garbage in-time to save GPU memory
            collectgarbage("collect")
        end
        print_stat(layer_repo)
        nerv.CuMatrix.print_profile()
        nerv.CuMatrix.clear_profile()
        if (not bp) and prefix ~= nil then
            nerv.info("writing back...")
            local fname = string.format("%s_cv%.3f.nerv",
                            prefix, get_accuracy(layer_repo))
            network:get_params():export(fname, nil)
        end
        return get_accuracy(layer_repo)
    end
    return iterative_trainer
end

dofile(arg[1])
start_halving_inc = 0.5
halving_factor = 0.6
end_halving_inc = 0.1
min_iter = 1
max_iter = 20
min_halving = 5
gconf.batch_size = 256
gconf.buffer_size = 81920

local pf0 = gconf.initialized_param
local trainer = build_trainer(pf0)
--local trainer = build_trainer("c3.nerv")
local accu_best = trainer(nil, gconf.cv_scp, false)
local do_halving = false

nerv.info("initial cross validation: %.3f", accu_best)
for i = 1, max_iter do
    nerv.info("[NN] begin iteration %d with lrate = %.6f", i, gconf.lrate)
    local accu_tr = trainer(nil, gconf.tr_scp, true)
    nerv.info("[TR] training set %d: %.3f", i, accu_tr)
    local accu_new = trainer(
                        string.format("%s_%s_iter_%d_lr%f_tr%.3f",
                            string.gsub(
                                (string.gsub(pf0[1], "(.*/)(.*)", "%2")),
                                "(.*)%..*", "%1"),
                            os.date("%Y%m%d%H%M%S"),
                            i, gconf.lrate,
                            accu_tr),
                        gconf.cv_scp, false)
    nerv.info("[CV] cross validation %d: %.3f", i, accu_new)
    -- TODO: revert the weights
    local accu_diff = accu_new - accu_best
    if do_halving and accu_diff < end_halving_inc and i > min_iter then
        break
    end
    if accu_diff < start_halving_inc and i >= min_halving then
        do_halving = true
    end
    if do_halving then
        gconf.lrate = gconf.lrate * halving_factor
    end
    if accu_new > accu_best then
        accu_best = accu_new
    end
--    nerv.Matrix.print_profile()
end
