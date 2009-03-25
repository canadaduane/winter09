OBJS = mandelbrot.o mandelbrot_omp.o misc.o
CC = gcc-4.2
MPICC = mpicc
DEBUG = 
CFLAGS = -Wall -O3 $(DEBUG) -c
LFLAGS = -Wall -O3 $(DEBUG)

mandelbrot: $(OBJS) main.c
	$(MPICC) $(LFLAGS) $(OBJS) main.c -o mandelbrot

mandelbrot.o: mandelbrot.h mandelbrot.c
	$(CC) $(CFLAGS) mandelbrot.c

misc.o: misc.h misc.c
	$(CC) $(CFLAGS) misc.c

mandelbrot_omp.o: mandelbrot_omp.h mandelbrot_omp.c
	$(CC) -fopenmp $(CFLAGS) mandelbrot_omp.c

clean:
	\rm -f *.o *~ mandelbrot
