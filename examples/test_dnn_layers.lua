require 'layer.affine'
require 'layer.sigmoid'
require 'layer.softmax_ce'

global_conf = {lrate = 0.8, wcost = 1e-6,
                momentum = 0.9, cumat_type = nerv.CuMatrixFloat}

pf = nerv.ChunkFile("affine.param", "r")
ltp = pf:read_chunk("a", global_conf)
bp = pf:read_chunk("b", global_conf)

-- print(bp.trans)

af = nerv.AffineLayer("test", global_conf, {["ltp"] = ltp,
                                            ["bp"] = bp,
                                            dim_in = {429},
                                            dim_out = {2048}})
sg = nerv.SigmoidLayer("test2", global_conf, {dim_in = {2048},
                                                dim_out = {2048}})
sm = nerv.SoftmaxCELayer("test3", global_conf, {dim_in = {2048, 2048},
                                                dim_out = {}})
af:init()
sg:init()
sm:init()

df = nerv.ChunkFile("input.param", "r")

label = nerv.CuMatrixFloat(10, 2048)
label:fill(0)
for i = 0, 9 do
    label[i][i] = 1.0
end

input1 = {df:read_chunk("input", global_conf).trans}
output1 = {nerv.CuMatrixFloat(10, 2048)}
input2 = output1
output2 = {nerv.CuMatrixFloat(10, 2048)}
input3 = {output2[1], label}
output3 = {}
err_input1 = {}
err_output1 = {nerv.CuMatrixFloat(10, 2048)}
err_input2 = err_output1
err_output2 = {nerv.CuMatrixFloat(10, 2048)}
err_input3 = err_output2
err_output3 = {input1[1]:create()}

for i = 0, 3 do
    -- propagate
    af:propagate(input1, output1)
    sg:propagate(input2, output2)
    sm:propagate(input3, output3)

    -- back_propagate
    sm:back_propagate(err_output1, err_input1, input3, output3)
    sg:back_propagate(err_output2, err_input2, input2, output2)
    af:back_propagate(err_output3, err_input3, input1, output1)

    -- update
    sm:update(err_input1, input3, output3)
    sg:update(err_input2, input2, output2)
    af:update(err_input3, input1, output1)


    print("output1")
    print(output1[1])
    print("output2")
    print(output2[1])
    print("err_output1")
    print(err_output1[1])
    print("err_output2")
    print(err_output2[1])
    nerv.printf("cross entropy: %.8f\n", sm.total_ce)
    nerv.printf("frames: %.8f\n", sm.total_frames)
end
print("linear")
print(af.ltp.trans)
print("linear2")
print(af.bp.trans)
