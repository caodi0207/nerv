CFLAG += -I ../install/include/luajit-2.0/  -I ../install/include/nerv/
LDFLAG += -L../install/lib/ -lluajit-5.1 -Wl,-rpath=../install/lib/ -lluaT -lnervcore
GCC := gcc

.PHONY: FORCE

FORCE: ../install/bin/luarocks
	echo "#!/bin/bash" > run.sh
	$< path >> run.sh
	echo "./main" >> run.sh
main: main.o
	$(GCC) -o $@ $<  $(LDFLAG)

main.o: main.c
	$(GCC) $(CFLAG) -o $@ $< -c
