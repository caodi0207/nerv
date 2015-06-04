#The Nerv IO Package#
Part of the [Nerv](../README.md) toolkit.

##Description##
There are four classes in to deal with chunk data, which are __nerv.ChunkFile__, __nerv.ChunkFileHandle__, __nerv.ChunkInfo__, __nerv.ChunkData__. Below is the underlying C structs.
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

##Methods##
* __ChunkFile ChunkFile.\_\_init(string mode, string fn)__  
`mode` can be `r` or `w`, for reading or writing a file.

##Developer Notes##
In __Nerv.io__, a returned(by `ChunkFile.__init`) __nerv.ChunkFile__ will have a member `handle`, which is a __nerv.ChunkFileHandle__.  