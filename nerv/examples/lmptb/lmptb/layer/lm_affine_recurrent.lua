local LMRecurrent = nerv.class('nerv.LMAffineRecurrentLayer', 'nerv.AffineRecurrentLayer') --breaks at sentence end, when </s> is met, input will be set to zero

--id: string
--global_conf: table
--layer_conf: table
--Get Parameters
function LMRecurrent:__init(id, global_conf, layer_conf)
    nerv.AffineRecurrentLayer.__init(self, id, global_conf, layer_conf)
    self.break_id = layer_conf.break_id --int, breaks recurrent input when the input (word) is break_id
    self.independent = layer_conf.independent --bool, whether break
end

function LMRecurrent:propagate(input, output)
    output[1]:mul(input[1], self.ltp_ih.trans, 1.0, 0.0, 'N', 'N')
    if (self.independent == true) then
        for i = 1, input[1]:nrow() do
            if (input[1][i - 1][self.break_id - 1] > 0.1) then --here is sentence break
                input[2][i - 1]:fill(0)
            end
        end
    end
    output[1]:mul(input[2], self.ltp_hh.trans, 1.0, 1.0, 'N', 'N')
    output[1]:add_row(self.bp.trans, 1.0)
end

