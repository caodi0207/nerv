-- To define a readable and writable chunk, one must define a class with the
-- following methods: __init(id, global_conf), read(handle), write(handle),
-- get_info(), set_info(info) and an id attribute. This file demonstrates a
-- basic chunk implementation which manages the I/O of a matrix

local MatrixChunk = nerv.class("nerv.MatrixChunk")

function MatrixChunk:__init(id, global_conf)
    self.id = id
    self.info = {}
    self.gconf = global_conf
end

function MatrixChunk:read(handle)
    -- pass the read handle to the matrix method
    self.data = nerv.MMatrixFloat.load(handle)
end

function MatrixChunk:write(handle)
    -- pass the write handle to the matrix method
    self.data:save(handle)
end

function MatrixChunk:get_info()
    return self.info
end

function MatrixChunk:set_info(info)
    self.info = info
end

function MatrixChunk.create_from_matrix(id, mat)
    local ins = nerv.MatrixChunk(id)
    ins.data = mat
    return ins
end

mat = nerv.MMatrixFloat(3, 4)
for i = 0, 2 do
    for j = 0, 3 do
        mat[i][j] = i + j
    end
end

cd = nerv.MatrixChunk.create_from_matrix("matrix1", mat)

cf = nerv.ChunkFile("test.nerv", "w")
cf:write_chunk(cd)
cf:close()

cf2 = nerv.ChunkFile("test.nerv", "r")
cd2 = cf2:read_chunk("matrix1")
print(cd2.data)
