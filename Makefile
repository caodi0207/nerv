.PHONY: all clean install luajit luarocks speech
SHELL := /bin/bash
PREFIX := $(CURDIR)/install/
all: luajit luarocks
luajit:
	PREFIX=$(PREFIX) ./tools/build_luajit.sh
luarocks:
	PREFIX=$(PREFIX) ./tools/build_luarocks.sh
install:
	cd nerv; $(PREFIX)/bin/luarocks make
	ln -sfv $(PREFIX)/lib/lua/5.1/libnerv.so $(PREFIX)/lib/ # FIXME: bad trick
speech:
	cd speech; $(PREFIX)/bin/luarocks make
