OBJS = mandelbrot_omp.o int_array.o misc.o
CC = gcc-4.2
MPICC = mpicc
DEBUG = 
CFLAGS = -Wall -O3 $(DEBUG) -c
LFLAGS = -Wall -O3 $(DEBUG)

mandelbrot: $(OBJS) main.c
	$(MPICC) -cc=gcc-4.2 -fopenmp $(LFLAGS) $(OBJS) main.c -o mandelbrot

mandelbrot_omp.o: mandelbrot_omp.h mandelbrot_omp.c
	$(CC) -fopenmp $(CFLAGS) mandelbrot_omp.c

misc.o: misc.h misc.c
	$(CC) $(CFLAGS) misc.c

int_array.o: int_array.h int_array.c
	$(CC) $(CFLAGS) int_array.c

clean:
	\rm -f *.o *~ mandelbrot
