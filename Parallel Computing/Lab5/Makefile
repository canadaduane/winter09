OBJS = misc.o median.o
CC = mpicc
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

pquicksort: $(OBJS) pquicksort.c
	$(CC) $(LFLAGS) $(OBJS) pquicksort.c -o pquicksort

misc.o: misc.h misc.c
	$(CC) $(CFLAGS) misc.c

median.o: median.h median.c
	$(CC) $(CFLAGS) median.c

clean:
	\rm -f *.o *~ hotplate
