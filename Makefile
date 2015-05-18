.PHONY: all clean luajit
OBJS := oop_example.o nerv.o luaT.o common.o matrix/matrix.o matrix/cumatrix.o matrix/init.o matrix/cukernel.o
LIBS := libnerv.so
LUA_LIBS := matrix/init.lua nerv.lua
INCLUDE := -I build/luajit-2.0/include/luajit-2.0/ -DLUA_USE_APICHECK
CUDA_BASE := /usr/local/cuda-6.5
CUDA_INCLUDE := -I $(CUDA_BASE)/include/
INCLUDE += $(CUDA_INCLUDE)
LDFLAGS := -L$(CUDA_BASE)/lib64/  -Wl,-rpath=$(CUDA_BASE)/lib64/ -lcudart -lcublas
CFLAGS :=
OBJ_DIR := build/objs
LUA_DIR := build/lua
NVCC := $(CUDA_BASE)/bin/nvcc

OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))
LIBS := $(addprefix $(OBJ_DIR)/,$(LIBS))
LUA_LIBS := $(addprefix $(LUA_DIR)/,$(LUA_LIBS))

all: luajit $(OBJ_DIR) $(LIBS) $(LUA_DIR) $(LUA_LIBS)
luajit:
	./build_luajit.sh
$(OBJ_DIR):
	-mkdir -p $(OBJ_DIR)
	-mkdir -p $(OBJ_DIR)/matrix
	-mkdir -p $(LUA_DIR)/matrix
$(LUA_DIR):
	-mkdir -p $(LUA_DIR)
$(OBJ_DIR)/%.o: %.c
	gcc -c -o $@ $< $(INCLUDE) -fPIC $(CFLAGS)
$(OBJ_DIR)/matrix/%.o: matrix/%.c
	gcc -c -o $@ $< $(INCLUDE) -fPIC $(CFLAGS)
$(OBJ_DIR)/matrix/cukernel.o: matrix/cukernel.cu
	$(NVCC) -c -o $@ $< -Xcompiler -fPIC $(INCLUDE) $(CFLAGS)
$(LUA_DIR)/%.lua: %.lua
	cp $< $@
$(OBJ_DIR)/luaT.o:
	gcc -c -o $@ luaT/luaT.c $(INCLUDE) -fPIC
$(LIBS): $(OBJS)
	gcc -shared -o $@ $(OBJS) $(LDFLAGS)
clean:
	-rm -rf $(OBJ_DIR)
	-rm -rf $(LUA_DIR)
