.PHONY: build install clean
SHELL := /bin/bash

BUILD_DIR := $(CURDIR)/build
LIB_PATH := $(LUA_BINDIR)/../lib
INC_PATH := $(LUA_BINDIR)/../include/nerv
LUA_DIR = $(INST_LUADIR)/nerv
OBJ_DIR := $(BUILD_DIR)/objs
ISUBDIR := io matrix luaT
SUBDIR := matrix io layer examples nn lib/io lib/luaT lib/matrix

INC_SUBDIR := $(addprefix $(INC_PATH)/,$(ISUBDIR))
OBJ_SUBDIR := $(addprefix $(OBJ_DIR)/,$(SUBDIR))
LUA_SUBDIR := $(addprefix $(LUA_DIR)/,$(SUBDIR))

INCS := common.h matrix/matrix.h io/chunk_file.h luaT/luaT.h
CORE_OBJS := lib/common.o lib/io/chunk_file.o \
			lib/matrix/mmatrix.o lib/matrix/cumatrix.o lib/matrix/cukernel.o
NERV_OBJS := nerv.o \
		matrix/mmatrix.o matrix/cumatrix.o matrix/init.o \
		io/init.o io/chunk_file.o \
		examples/oop_example.o
LUAT_OBJS := lib/luaT/luaT.o

INCS := $(addprefix $(INC_PATH)/,$(INCS))
CORE_OBJS := $(addprefix $(OBJ_DIR)/,$(CORE_OBJS))
NERV_OBJS := $(addprefix $(OBJ_DIR)/,$(NERV_OBJS))
LUAT_OBJS := $(addprefix $(OBJ_DIR)/,$(LUAT_OBJS))

OBJS := $(CORE_OBJS) $(NERV_OBJS) $(LUAT_OBJS)
LIBS := $(INST_LIBDIR)/libnerv.so $(LIB_PATH)/libnervcore.so $(LIB_PATH)/libluaT.so
LUA_LIBS := matrix/init.lua io/init.lua init.lua \
			layer/init.lua layer/affine.lua layer/sigmoid.lua layer/softmax_ce.lua layer/softmax.lua \
			layer/window.lua layer/bias.lua layer/combiner.lua layer/mse.lua layer/affine_recurrent.lua\
			nn/init.lua nn/layer_repo.lua nn/param_repo.lua nn/layer_dag.lua \
			io/sgd_buffer.lua 

INCLUDE := -I $(LUA_INCDIR) -DLUA_USE_APICHECK
CUDA_BASE := /usr/local/cuda-6.5
#CUDA_BASE := /usr/local/cuda-5.0
CUDA_INCLUDE := -I $(CUDA_BASE)/include/
INCLUDE += $(CUDA_INCLUDE)

LDFLAGS := -L$(CUDA_BASE)/lib64/ -Wl,-rpath=$(CUDA_BASE)/lib64/ -lcudart -lcublas
CFLAGS := -Wall -Wextra -O2
NVCC := $(CUDA_BASE)/bin/nvcc
NVCC_FLAGS := -Xcompiler -fPIC,-Wall,-Wextra

LUA_LIBS := $(addprefix $(LUA_DIR)/,$(LUA_LIBS))

build: $(OBJ_DIR) $(OBJ_SUBDIR) $(OBJS)

$(OBJ_DIR) $(LUA_DIR) $(OBJ_SUBDIR) $(LUA_SUBDIR) $(INC_SUBDIR):
	-mkdir -p $@
$(OBJ_DIR)/%.o: %.c $(patsubst /%.o,/%.c,$@)
	gcc -c -o $@ $< $(INCLUDE) -fPIC $(CFLAGS)
$(OBJ_DIR)/lib/matrix/cukernel.o: lib/matrix/cukernel.cu
	$(NVCC) -c -o $@ $< $(INCLUDE) $(NVCC_FLAGS)
$(LUA_DIR)/%.lua: %.lua
	cp $< $@

$(LIB_PATH)/libnervcore.so: $(CORE_OBJS)
	gcc -shared -o $@ $^ $(LDFLAGS)
$(LIB_PATH)/libluaT.so: $(LUAT_OBJS)
	gcc -shared -o $@ $^ $(LDFLAGS)
$(INST_LIBDIR)/libnerv.so: $(NERV_OBJS) $(LIB_PATH)/libnervcore.so $(LIB_PATH)/libluaT.so
	gcc -shared -o $@ $(NERV_OBJS) $(LDFLAGS) -Wl,-rpath=$(LIB_PATH) -L$(LIB_PATH) -lnervcore -lluaT

$(OBJ_DIR)/matrix/cumatrix.o: matrix/generic/cumatrix.c matrix/generic/matrix.c
$(OBJ_DIR)/matrix/mmatrix.o: matrix/generic/mmatrix.c matrix/generic/matrix.c

$(OBJ_DIR)/lib/matrix/cumatrix.o: lib/matrix/generic/cumatrix.c lib/matrix/generic/matrix.c lib/matrix/generic/cukernel.cu
$(OBJ_DIR)/lib/matrix/mmatrix.o: lib/matrix/generic/mmatrix.c lib/matrix/generic/matrix.c
$(OBJ_DIR)/lib/matrix/cukernel.o: lib/matrix/generic/cukernel.cu

clean:
	-rm -rf $(OBJ_DIR)

install: $(LIBS) $(LUA_DIR) $(LUA_SUBDIR) $(LUA_LIBS) $(INC_SUBDIR) $(INCS)

$(INC_PATH)/%.h: lib/%.h
	cp $< $@
