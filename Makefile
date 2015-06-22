.PHONY: all clean install luajit luarocks nerv
SHELL := /bin/bash
PREFIX := $(CURDIR)/install/
all: luajit luarocks
luajit:
	PREFIX=$(PREFIX) ./tools/build_luajit.sh
luarocks:
	PREFIX=$(PREFIX) ./tools/build_luarocks.sh
install:
	cd nerv; $(PREFIX)/bin/luarocks make
