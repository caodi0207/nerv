.PHONY: all clean
OBJS := oop_example.o nerv.o luaT.o
INCLUDE := -I luajit-2.0/build/include/luajit-2.0/ -DLUA_USE_APICHECK
LDFLAGS := -L luajit-2.0/build/lib/ -llua -lm
OBJ_DIR := build/objs
OBJS := $(addprefix $(OBJ_DIR)/,$(OBJS))
all: libnerv.so
$(OBJS): $(OBJ_DIR)
$(OBJ_DIR):
	-mkdir -p $(OBJ_DIR)
$(OBJ_DIR)/%.o: %.c
	gcc -c -o $@ $< $(INCLUDE) -fPIC
$(OBJ_DIR)/luaT.o:
	gcc -c -o $@ luaT/luaT.c $(INCLUDE) -fPIC
libnerv.so: $(OBJS)
	gcc -shared -o $(OBJ_DIR)/$@ $(OBJS)
clean:
	-rm -rf $(OBJ_DIR)
