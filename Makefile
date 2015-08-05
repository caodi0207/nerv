.PHONY: all clean install luajit luarocks speech
SHELL := /bin/bash
PREFIX := $(CURDIR)/install/
all: luajit luarocks install
luajit:
	PREFIX=$(PREFIX) ./tools/build_luajit.sh
luarocks:
	PREFIX=$(PREFIX) ./tools/build_luarocks.sh
install:
	cd nerv; $(PREFIX)/bin/luarocks make
speech:
	cd speech/speech_utils; $(PREFIX)/bin/luarocks make
	cd speech/htk_io; $(PREFIX)/bin/luarocks make
clean:
	cd nerv && make clean
