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
	cd speech/htk_io; $(PREFIX)/bin/luarocks make
clean:
	rm -r $(CURDIR)/install
