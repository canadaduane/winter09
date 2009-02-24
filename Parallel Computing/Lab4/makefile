OBJS = misc.o hotplate.o
CC = mpicc
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

hotplate: $(OBJS) main.c
	$(CC) $(LFLAGS) $(OBJS) main.c -o hotplate

misc.o: misc.h misc.c
	$(CC) $(CFLAGS) misc.c

hotplate.o: hotplate.c hotplate.h
	$(CC) $(CFLAGS) hotplate.c

clean:
	\rm -f *.o *~ hotplate
