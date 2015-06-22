#!/bin/bash
LUAJIT_PREFIX="$PREFIX"
LUAJIT_SRC='luajit-2.0/'
[[ -f "$LUAJIT_PREFIX/bin/luajit" ]] || (cd "$LUAJIT_SRC"; make && make PREFIX="$LUAJIT_PREFIX" install)
