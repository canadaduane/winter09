OBJS = misc.o
CC = gcc
NVCC = nvcc
DEBUG = --device-emulation --debug
CFLAGS = -O3 $(DEBUG) -c
LFLAGS = -O3 $(DEBUG)

alignment: $(OBJS) alignment.cu
	$(NVCC) $(LFLAGS) $(OBJS) alignment.cu -o alignment

misc.o: misc.c misc.h
	$(NVCC) $(CFLAGS) misc.c

clean:
	\rm -f *.o *~ align
