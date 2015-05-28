.PHONY: all clean luajit
OBJS := nerv.o luaT.o common.o \
		matrix/mmatrix.o matrix/cumatrix.o matrix/init.o matrix/cukernel.o \
		io/init.o io/param.o \
		examples/oop_example.o
LIBS := libnerv.so
LUA_LIBS := matrix/init.lua io/init.lua nerv.lua \
			pl/utils.lua pl/compat.lua \
			layer/init.lua layer/affine.lua layer/sigmoid.lua
INCLUDE := -I build/luajit-2.0/include/luajit-2.0/ -DLUA_USE_APICHECK
CUDA_BASE := /usr/local/cuda-6.5
CUDA_INCLUDE := -I $(CUDA_BASE)/include/
INCLUDE += $(CUDA_INCLUDE)
LDFLAGS := -L$(CUDA_BASE)/lib64/  -Wl,-rpath=$(CUDA_BASE)/lib64/ -lcudart -lcublas
CFLAGS := -Wall -Wextra
OBJ_DIR := build/objs
LUA_DIR := build/lua
SUBDIR := matrix io layer examples pl
NVCC := $(CUDA_BASE)/bin/nvcc
NVCC_FLAGS := -Xcompiler -fPIC,-Wall,-Wextra

OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))
OBJ_SUBDIR := $(addprefix $(OBJ_DIR)/,$(SUBDIR))
LUA_SUBDIR := $(addprefix $(LUA_DIR)/,$(SUBDIR))
LIBS := $(addprefix $(OBJ_DIR)/,$(LIBS))
LUA_LIBS := $(addprefix $(LUA_DIR)/,$(LUA_LIBS))

all: luajit $(OBJ_DIR) $(OBJ_SUBDIR) $(LIBS) $(LUA_DIR) $(LUA_SUBDIR) $(LUA_LIBS)
luajit:
	./build_luajit.sh
$(OBJ_DIR) $(LUA_DIR) $(OBJ_SUBDIR) $(LUA_SUBDIR):
	-mkdir -p $@
$(OBJ_DIR)/%.o: %.c $(patsubst /%.o,/%.c,$@)
	gcc -c -o $@ $< $(INCLUDE) -fPIC $(CFLAGS)
$(OBJ_DIR)/matrix/cukernel.o: matrix/cukernel.cu
	$(NVCC) -c -o $@ $< $(INCLUDE) $(NVCC_FLAGS)
$(LUA_DIR)/%.lua: %.lua
	cp $< $@
$(OBJ_DIR)/luaT.o:
	gcc -c -o $@ luaT/luaT.c $(INCLUDE) -fPIC
$(LIBS): $(OBJS)
	gcc -shared -o $@ $(OBJS) $(LDFLAGS)

$(OBJ_DIR)/matrix/cumatrix.o: matrix/generic/cumatrix.c matrix/generic/matrix.c
$(OBJ_DIR)/matrix/mmatrix.o: matrix/generic/mmatrix.c matrix/generic/matrix.c

clean:
	-rm -rf $(OBJ_DIR)
	-rm -rf $(LUA_DIR)
