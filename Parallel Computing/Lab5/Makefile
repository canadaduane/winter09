OBJS = misc.o
CC = mpicc
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

pquicksort: $(OBJS) pquicksort.c
	$(CC) $(LFLAGS) $(OBJS) pquicksort.c -o pquicksort

misc.o: misc.h misc.c
	$(CC) $(CFLAGS) misc.c

clean:
	\rm -f *.o *~ hotplate
