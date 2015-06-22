#!/bin/bash
LUAROCKS_PREFIX="$PREFIX"
LUAROCKS_SRC='luarocks/'
[[ -f "$LUAROCKS_PREFIX/bin/luarocks" ]] || (cd "$LUAROCKS_SRC"; ./configure --prefix=$LUAROCKS_PREFIX  --with-lua-include="$LUAROCKS_PREFIX/include/luajit-2.0/" --with-lua="$LUAROCKS_PREFIX" --lua-suffix='jit' --with-lua-lib="$LUAROCKS_PREFIX/lib/"; make clean && make build && make bootstrap )
