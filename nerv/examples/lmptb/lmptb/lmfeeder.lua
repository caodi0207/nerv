require 'lmptb.lmvocab'

local Feeder = nerv.class("nerv.LMFeeder")

local printf = nerv.printf

--global_conf: table
--batch_size: int
--vocab: nerv.LMVocab
function Feeder:__init(global_conf, batch_size, vocab)
    self.gconf = global_conf
    self.fh = nil --file handle to read, nil means currently no file
    self.batch_size = batch_size
    self.log_pre = "[LOG]LMFeeder:"
    self.vocab = vocab
    self.streams = nil
end

--fn: string
--Initialize all streams
function Feeder:open_file(fn)
    if (self.fh ~= nil) then
        nerv.error("%s error: in open_file, file handle not nil.")
    end
    printf("%s opening file %s...\n", self.log_pre, fn)
    self.fh = io.open(fn, "r")
    self.streams = {}
    for i = 1, self.batch_size, 1 do
        self.streams[i] = {["store"] = {self.vocab.sen_end_token}, ["head"] = 1, ["tail"] = 1}
    end
end
   
--id: int
--Refresh stream id,  read a line from file
function Feeder:refresh_stream(id)
    if (self.streams[id] == nil) then
        nerv.error("stream %d does not exit.", id)
    end
    local st = self.streams[id]
    if (st.store[st.head] ~= nil) then return end
    if (self.fh == nil) then return end
    local list = self.vocab:read_line(self.fh)
    if (list == nil) then --file has end
        printf("%s file expires, closing.\n", self.log_pre)
        self.fh:close() 
        self.fh = nil 
        return 
    end
    for i = 1, #list, 1 do
        st.tail = st.tail + 1
        st.store[st.tail] = list[i]
    end
end

--Returns: nil/table
--If gets something, return a list of string, vocab.null_token indicates end of string
function Feeder:get_batch()
    local got_new = false
    local list = {}
    for i = 1, self.batch_size, 1 do
        self:refresh_stream(i)
        local st = self.streams[i]
        list[i] = st.store[st.head]
        if (list[i] == nil) then list[i] = self.vocab.null_token end
        if (list[i] ~= nil and list[i] ~= self.vocab.null_token)then
            got_new = true
            st.store[st.head] = nil
            st.head = st.head + 1
        end 
    end
    if (got_new == false) then
        return nil
    else
        return list
    end
end

--[[
do
    local test_fn = "/home/slhome/txh18/workspace/nerv-project/some-text"
    --local test_fn = "/home/slhome/txh18/workspace/nerv-project/nerv/examples/lmptb/PTBdata/ptb.train.txt"
    local vocab = nerv.LMVocab()
    vocab:build_file(test_fn)
    local batch_size = 3
    local feeder = nerv.LMFeeder({}, batch_size, vocab)
    feeder:open_file(test_fn)
    while (1) do
        local list = feeder:get_batch()
        if (list == nil) then break end
        for i = 1, batch_size, 1 do
            printf("%s(%d) ", list[i], vocab:get_word_str(list[i]).id) 
        end
        printf("\n")
    end
end
]]--
