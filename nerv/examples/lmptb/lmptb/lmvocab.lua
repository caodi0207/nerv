local Vocab = nerv.class("nerv.LMVocab")

local printf = nerv.printf

local mysplit = function(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

function Vocab:__init(global_conf)
    self.gconf = global_conf
    self.sen_end_token = "</s>"
    self.unk_token = "<unk>"
    self.null_token = "<null>" --indicating end of stream(in feeder)
    self.log_pre = "[LOG]LMVocab:"
    self.map_str = {} --map from str to word_entry
    self.map_id = {} --map from id to word_entry
    
    self:add_word(self.sen_end_token)
    self:add_word(self.unk_token)
end

--id: int
--w_str: string
--Returns: table
function Vocab:new_word_entry(id, w_str)
    return { ["id"] = id,
            ["str"] = w_str,
            ["cnt"] = 0,
        } 
end

--Returns: int
function Vocab:size()
    return #self.map_id
end

--w_str: string
--if w_str is not in vocab, then add it in, if it is already in, do nothing
function Vocab:add_word(w_str)
    if (self.map_str[w_str] ~= nil) then
        return 
    end
    local e = self:new_word_entry(self:size() + 1, w_str)
    self.map_id[self:size() + 1] = e
    self.map_str[w_str] = e
end

--Returns: table, the entry of the unk
function Vocab:get_unk_entry()
    if (self.map_str[self.unk_token] == nil) then
        nerv.error("unk entry not found.")
    end
    return self.map_str[self.unk_token]
end

--Returns: table, the entry of sentence end
function Vocab:get_sen_entry()
    if (self.map_str[self.sen_end_token] == nil) then
        nerv.error("sen end token not found")
    end
    return self.map_str[self.sen_end_token]
end

function Vocab:is_unk_str(w)
    if (key == self.null_token) then
        nerv.error("Vocab:get_word_str is called by the null token")
    end
    if (w == self.unk_token or self.map_str[w] == nil) then
        return true
    else
        return false
    end
end

--key: string
--Returns: table, the word_entry of this key
function Vocab:get_word_str(key)
    if (self.map_str[key] == nil) then
        return self:get_unk_entry()
    end
    if (key == self.null_token) then
        nerv.error("Vocab:get_word_str is called by the null token")
    end
    return self.map_str[key]
end

--key: int
--Returns: table
function Vocab:get_word_id(key)
    if (self.map_id[key] == nil) then
        nerv.error("id key %d does not exist.", key) 
    end
    return self.map_id(key)
end

--fh: file_handle
--Returns: a list of tokens(string) in the line, if there is no "</s>" at the end, the function will at it, if nothing to read, returns nil
function Vocab:read_line(fh)
    local l_str = fh:read("*line")
    if (l_str == nil) then return nil end
    local list = mysplit(l_str)
    if (list[(#list)] ~= self.sen_end_token) then
        list[#list + 1] = self.sen_end_token
    end
    return list
end

--fn: string
--Add all words in fn to the vocab
function Vocab:build_file(fn)
    printf("%s Vocab building on file %s...\n", self.log_pre, fn)
    local file = io.open(fn, "r")
    while (true) do
        local list = self:read_line(file)
        if (list == nil) then
            break
        else
            for i = 1, #list, 1 do
                self:add_word(list[i])          
            end
        end
    end
    file:close()
    printf("%s Building finished, vocab size now is %d.\n", self.log_pre, self:size())
end

--[[test
do
    local test_fn = "/home/slhome/txh18/workspace/nerv-project/some-text"
    local vocab = nerv.LMVocab()
    vocab:build_file(test_fn)
end
]]--
