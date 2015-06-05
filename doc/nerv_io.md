#The Nerv IO Package#
Part of the [Nerv](../README.md) toolkit.

##Description##
The main class that the user uses to store and read parameter object to and from files is __nerv.ChunkFile__.  
In the file, a parameter object will be saved using a standard format. First is the length(in byte) of this object, then a table which includes some meta information of the object, and a data area. Below is an example text file.  
```
[0000000000202]
{type="nerv.ExampleP",info={message="just-a-try"},id="exampleP1"}
3 3
5.000000 5.000000 5.000000 
5.000000 5.000000 5.000000 
5.000000 5.000000 5.000000 
1 3
4.000000 4.000000 4.000000 
[0000000000202]
{type="nerv.ExampleP",info={message="just-a-try"},id="exampleP2"}
3 3
4.000000 4.000000 4.000000 
4.000000 4.000000 4.000000 
4.000000 4.000000 4.000000 
1 3
3.000000 3.000000 3.000000 
```

##Methods##
* __ChunkFile ChunkFile(string fn, string mode)__  
`mode` can be `r` or `w`, for reading or writing a file. The returned __ChunkFile__ will be ready to write or read objects which follows the __nerv.Param__ interface(using `write_chunk` and `read_chunk`). 
* __void ChunkFile.write_chunk(ChunkFile self, Param p)__  
Write `p` into the file. `p:write` will be called.
* __Param ChunkFile.read_chunk(ChunkFile self, string id, table global_conf)__  
Read the __Param__ object by id `id` from the file `self`. It will be constructed using `__init(id, global_conf)`. `p:read` will be called.

##Examples##
* An example showing how to use __ChunkFile__ to store and read parameter objects.
```
require 'io'
do
    local mt, mpt = nerv.class('nerv.ExampleP', 'nerv.Param')
    function nerv.ExampleP:__init(id, global_conf)
        self.id = id
        self.global_conf = global_conf
        self.matrix = nerv.MMatrixFloat(3, 3)
        for i = 0, 2, 1 do
            for j = 0, 2, 1 do
                self.matrix[i][j] = 3
            end
        end
        self.bias = nerv.MMatrixFloat(1, 3)
        for i = 0, 2, 1 do
            self.bias[i] = 2;
        end
        self:set_info({message = 'just-a-try'})
    end
    function nerv.ExampleP:addOne()
        for i = 0, 2, 1 do
            for j = 0, 2, 1 do
                self.matrix[i][j] = self.matrix[i][j] + 1
            end
        end
        for i = 0, 2, 1 do
            self.bias[i] = self.bias[i] + 1
        end
    end
    function nerv.ExampleP:read(pcdata)
        self.matrix = nerv.MMatrixFloat.load(pcdata)
        self.bias = nerv.MMatrixFloat.load(pcdata)
    end
    function nerv.ExampleP:write(pfhandle)
        self.matrix:save(pfhandle) 
        self.bias:save(pfhandle)
    end
end
global_conf = {}
do
    local f = nerv.ChunkFile('../tmp', 'w')
    local exampleP1 = nerv.ExampleP('exampleP1', global_conf)
    local exampleP2 = nerv.ExampleP('exampleP2', global_conf)
    exampleP1:addOne() 
    exampleP1:addOne()
    exampleP2:addOne()

    f:write_chunk(exampleP1)
    f:write_chunk(exampleP2)
end
do
    local f = nerv.ChunkFile('../tmp', 'r')
    local exampleP1 = f:read_chunk('exampleP1', global_conf)
    local exampleP2 = f:read_chunk('exampleP2', global_conf)
    print(exampleP1.matrix)
    print(exampleP2.matrix)
end
```

##Developer Notes##
* There are four classes in to deal with chunk data, which are __nerv.ChunkFile__, __nerv.ChunkFileHandle__, __nerv.ChunkInfo__, __nerv.ChunkData__. Below is the underlying C structs.
```
typedef struct ChunkFileHandle {
    FILE *fp;
} ChunkFileHandle;
typedef struct ChunkInfo {
    off_t offset, length;
} ChunkInfo;
typedef struct ChunkData {
    FILE *fp;
    char *data;
} ChunkData;
```
* In __Nerv.io__, a returned(by `ChunkFile.__init`) __nerv.ChunkFile__ will have a member `handle`, which is a __nerv.ChunkFileHandle__.  