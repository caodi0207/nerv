CFLAG += -I ../install/include/luajit-2.0/  -I ../install/include/nerv/
LDFLAG += -L../install/lib/ -lluajit-5.1 -Wl,-rpath=../install/lib/ -lluaT -lnervcore
GCC := gcc

main: main.o
	$(GCC) -o $@ $<  $(LDFLAG)

main.o: main.c
	$(GCC) $(CFLAG) -o $@ $< -c
