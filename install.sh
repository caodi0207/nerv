#!/bin/bash
PREFIX="${PWD}/build"
LUAJIT_SRC='luajit-2.0/'
LUAJIT_PREFIX="${PREFIX}/luaJIT"
(cd "$LUAJIT_SRC"; make && make PREFIX="$LUAJIT_PREFIX" install)
