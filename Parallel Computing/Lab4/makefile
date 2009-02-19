OBJS = misc.o hotplate.o
CC = mpic++
DEBUG = -g
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

hotplate: $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o hotplate

misc.o: misc.h misc.c
	$(CC) $(CFLAGS) misc.c

hotplate.o: hotplate.c
	$(CC) $(CFLAGS) hotplate.c

clean:
	\rm -f *.o *~ hotplate
