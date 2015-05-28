require 'layer.affine'
require 'layer.sigmoid'
require 'layer.softmax_ce'

global_conf = {lrate = 0.8, wcost = 1e-6, momentum = 0.9}

pf = nerv.ParamFile("affine.param", "r")
ltp = pf:read_param("a")
bp = pf:read_param("b")

-- print(bp.trans)

af = nerv.AffineLayer("test", global_conf, ltp, bp)
sg = nerv.SigmoidLayer("test2", global_conf)
sm = nerv.SoftmaxCELayer("test3", global_conf)

af:init()
sg:init()
sm:init()

df = nerv.ParamFile("input.param", "r")

label = nerv.CuMatrixFloat(10, 2048)
label:fill(0)
for i = 0, 9 do
    label[i][i] = 1.0
end

input1 = {[0] = df:read_param("input").trans}
output1 = {[0] = nerv.CuMatrixFloat(10, 2048)}
input2 = output1
output2 = {[0] = nerv.CuMatrixFloat(10, 2048)}
input3 = {[0] = output2[0], [1] = label}
output3 = nil
err_input1 = nil
err_output1 = {[0] = nerv.CuMatrixFloat(10, 2048)}
err_input2 = err_output1
err_output2 = {[0] = nerv.CuMatrixFloat(10, 2048)}
err_input3 = err_output2
err_output3 = {[0] = input1[0]:create()}

for i = 0, 3 do
    -- propagate
    af:propagate(input1, output1)
    sg:propagate(input2, output2)
    sm:propagate(input3, output3)


    -- back_propagate
    sm:back_propagate(err_output1, err_input1, input3, output3)
    sm:update(err_input1, input3, output3)

    sg:back_propagate(err_output2, err_input2, input2, output2)
    sg:update(err_input2, input2, output2)

    af:back_propagate(err_output3, err_input3, input1, output1)
    af:update(err_input3, input1, output1)


    print("output1")
    print(output1[0])
    print("output2")
    print(output2[0])
    print("err_output1")
    print(err_output1[0])
    print("err_output2")
    print(err_output2[0])
    nerv.utils.printf("cross entropy: %.8f\n", sm.total_ce)
    nerv.utils.printf("frames: %.8f\n", sm.total_frames)
end
print("linear")
print(af.ltp.trans)
print("linear2")
print(af.bp.trans)
