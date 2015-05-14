.PHONY: all clean luajit
OBJS := oop_example.o nerv.o luaT.o
LIBS := libnerv.so
INCLUDE := -I build/luajit-2.0/include/luajit-2.0/ -DLUA_USE_APICHECK
LDFLAGS := -L luajit-2.0/build/lib/ -llua -lm
OBJ_DIR := build/objs
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))
LIBS := $(addprefix $(OBJ_DIR)/,$(LIBS))
all: luajit $(OBJ_DIR) $(LIBS)
luajit:
	./build_luajit.sh
$(OBJ_DIR):
	-mkdir -p $(OBJ_DIR)
$(OBJ_DIR)/%.o: %.c
	gcc -c -o $@ $< $(INCLUDE) -fPIC
$(OBJ_DIR)/luaT.o:
	gcc -c -o $@ luaT/luaT.c $(INCLUDE) -fPIC
$(LIBS): $(OBJS)
	gcc -shared -o $@ $(OBJS)
clean:
	-rm -rf $(OBJ_DIR)
