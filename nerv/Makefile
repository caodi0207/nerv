.PHONY: build install clean
SHELL := /bin/bash
BUILD_DIR := $(CURDIR)/build
OBJS := nerv.o luaT.o common.o \
		matrix/mmatrix.o matrix/cumatrix.o matrix/init.o matrix/cukernel.o \
		io/init.o io/chunk_file.o \
		examples/oop_example.o
LIBS := libnerv.so
LUA_LIBS := matrix/init.lua io/init.lua init.lua \
			layer/init.lua layer/affine.lua layer/sigmoid.lua layer/softmax_ce.lua \
			layer/window.lua layer/bias.lua layer/combiner.lua layer/mse.lua \
			nn/init.lua nn/layer_repo.lua nn/param_repo.lua nn/layer_dag.lua \
			io/sgd_buffer.lua
INCLUDE := -I $(LUA_INCDIR) -DLUA_USE_APICHECK
CUDA_BASE := /usr/local/cuda-6.5
#CUDA_BASE := /usr/local/cuda-5.0
CUDA_INCLUDE := -I $(CUDA_BASE)/include/
INCLUDE += $(CUDA_INCLUDE)
LDFLAGS := -L$(CUDA_BASE)/lib64/  -Wl,-rpath=$(CUDA_BASE)/lib64/ -lcudart -lcublas
CFLAGS := -Wall -Wextra -O2
OBJ_DIR := $(BUILD_DIR)/objs
SUBDIR := matrix io layer examples nn
NVCC := $(CUDA_BASE)/bin/nvcc
NVCC_FLAGS := -Xcompiler -fPIC,-Wall,-Wextra

LUA_DIR = $(INST_LUADIR)/nerv
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))
OBJ_SUBDIR := $(addprefix $(OBJ_DIR)/,$(SUBDIR))
LUA_SUBDIR := $(addprefix $(LUA_DIR)/,$(SUBDIR))
LIBS := $(addprefix $(INST_LIBDIR)/,$(LIBS))
LUA_LIBS := $(addprefix $(LUA_DIR)/,$(LUA_LIBS))

build: $(OBJ_DIR) $(OBJ_SUBDIR) $(OBJS)
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

$(OBJ_DIR)/matrix/cumatrix.o: matrix/generic/cumatrix.c matrix/generic/matrix.c matrix/generic/cukernel.cu
$(OBJ_DIR)/matrix/mmatrix.o: matrix/generic/mmatrix.c matrix/generic/matrix.c
$(OBJ_DIR)/matrix/cukernel.o: matrix/generic/cukernel.cu

.PHONY: speech

speech:
	-mkdir -p build/objs/speech/tnet_io
	$(MAKE) -C speech/ BUILD_DIR=$(BUILD_DIR) LIB_DIR=$(LIB_DIR) OBJ_DIR=$(CURDIR)/build/objs/speech/ LUA_DIR=$(LUA_DIR)

clean:
	-rm -rf $(OBJ_DIR)

install: $(LIBS) $(LUA_DIR) $(LUA_SUBDIR) $(LUA_LIBS)