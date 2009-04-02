OBJS = misc.o
CC = gcc
NVCC = nvcc
DEBUG = --device-emulation
CFLAGS = -O3 $(DEBUG) -c
LFLAGS = -O3 $(DEBUG)

align: $(OBJS) alignment.cu
	$(NVCC) $(LFLAGS) $(OBJS) alignment.cu -o alignment

misc.o: misc.c
	$(NVCC) $(CFLAGS) misc.c

clean:
	\rm -f *.o *~ align
