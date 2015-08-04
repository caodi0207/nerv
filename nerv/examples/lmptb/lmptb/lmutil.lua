local Util = nerv.class("nerv.LMUtil")

--list: table, list of string(word)
--vocab: nerv.LMVocab
--ty: nerv.CuMatrix
--Returns: nerv.CuMatrixFloat
--Create a matrix of type 'ty', size #list * vocab:size(). null_word will become a zero vector.
function Util.create_onehot(list, vocab, ty)
    local m = ty(#list, vocab:size())
    m:fill(0)
    for i = 1, #list, 1 do
        --index in matrix starts at 0
        if (list[i] ~= vocab.null_token) then 
            m[i - 1][vocab:get_word_str(list[i]).id - 1] = 1
        end
    end
    return m
end

function Util.wait(sec)
    local start = os.time()
    repeat until os.time() > start + sec
end

local Result = nerv.class("nerv.LMResult")

--global_conf: table
--vocab:nerv.LMVocab
function Result:__init(global_conf, vocab)
    self.gconf = global_conf
    self.vocab = vocab
end

--cla:string
--Initialize status of class cla
function Result:init(cla)
    self[cla] = {logp_all = 0, logp_unk = 0, cn_w = 0, cn_unk = 0, cn_sen = 0}
end

--cla:string
--w:string
--prob:float, the probability
function Result:add(cla, w, prob)
    self[cla].logp_all = self[cla].logp_all + math.log10(prob)
    if (self.vocab:is_unk_str(w)) then
        self[cla].logp_unk = self[cla].logp_unk + math.log10(prob)
        self[cla].cn_unk = self[cla].cn_unk + 1
    end
    if (w == self.vocab.sen_end_token) then
        self[cla].cn_sen = self[cla].cn_sen + 1
    else
        self[cla].cn_w = self[cla].cn_w + 1
    end
end

function Result:ppl_net(cla)  
    local c = self[cla]
    return math.pow(10, -(c.logp_all - c.logp_unk) / (c.cn_w - c.cn_unk + c.cn_sen))
end

function Result:ppl_all(cla)
    local c = self[cla]
    return math.pow(10, -(c.logp_all) / (c.cn_w + c.cn_sen))
end

function Result:status(cla)
    return "LMResult status of " .. cla .. ": " .. "<SEN_CN " .. self[cla].cn_sen .. "> <W_CN " .. self[cla].cn_w .. "> <PPL_NET " .. self:ppl_net(cla) .. "> <PPL_OOV " .. self:ppl_all(cla) .. "> <LOGP " .. self[cla].logp_all .. ">"
end
