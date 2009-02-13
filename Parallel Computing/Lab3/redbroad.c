#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

int Dimensions = -1;
int NumNodes = -1;
int MyID = -1;

int* Message;
int MessageSize = -1;

/* Look for a command-line switch and get the next integer value */
int shell_arg_int( int argc, char* argv[], char* arg_switch, int default_value)
{
    int i;
    for ( i = 0; i < argc - 1; i++ )
    {
        if ( strcmp( argv[ i ], arg_switch ) == 0 )
        {
            return atoi( argv[ i + 1 ] );
        }
    }
    return default_value;
}

void print_message(int* msg, int size)
{
    int i;
    for ( i = 0; i < size; i++ ) printf( "%d ", msg[i] );
    printf( "\n" );
    
}

// vnode = ((iproc + nproc) - root) % nproc
void reduce(int root, int* boxes)
{
    int i, j;
    int vMyID = ((MyID + (1<<Dimensions)) - root) % (1<<Dimensions);
    int mask = 0;
    MPI_Request request;
    MPI_Status status;
    
    int* sum = calloc( sizeof(int), MessageSize );
    int* tmp = calloc( sizeof(int), MessageSize );
    memcpy( sum, boxes, sizeof(int) * MessageSize );
    
    for ( i = 0; i < Dimensions; i++)
    {
        int other = vMyID ^ (1<<i);
        int vOther = (other + root) % (1<<Dimensions);
        printf("(%d) mine: %d, other: %d\n", i, vMyID, vOther);
        if ( (vMyID & mask) == 0 && vOther < NumNodes )
        {
            if ( (vMyID & (1<<i)) != 0)
            {
                printf("%d send\n", vMyID);
                MPI_Isend(sum, MessageSize, MPI_INT, vOther, 0, MPI_COMM_WORLD, &request);
            }
            else
            {
                printf("%d recv\n", vMyID);
                MPI_Recv(tmp, MessageSize, MPI_INT, vOther, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                for ( j = 0; j < MessageSize; j++) sum[j] += tmp[j];
            }
        }
        mask = mask ^ (1<<i);
    }
    
    memcpy( boxes, sum, sizeof(int) * MessageSize );
    
    free(sum);
    free(tmp);
}

void broadcast( int root, int* boxes )
{
    int i, j;
    int vMyID = ((MyID + (1<<Dimensions)) - root) % (1<<Dimensions);
    MPI_Request request;
    MPI_Status status;
    
    /* Set mask to 0b for 1 dim, 01b for 2 dim, 011b for 3 dim, etc. */
    int mask = (1 << (Dimensions-1)) - 1;

    /* Set dimensional bit to 1b for 1 dim, 10b for 2 dim, 100b for 3 dim, etc.*/
    int flip = (1 << (Dimensions-1));
    
    for ( i = 0; i < Dimensions; i++ )
    { /* Loop through each dimension */
        
        /* The sender (if we are a recipient), or recipient (if we are a sender) */
        int other = vMyID ^ flip;
        int vOther = (other + root) % (1<<Dimensions);

        if ( ( vMyID & mask ) == 0 && vMyID < NumNodes && vOther < NumNodes)
        { /* Bits indicate it's our turn to send or receive */
            
            if ( ( vMyID & flip ) == 0 )
            { /* The bits indicate it's our turn to send ... */
                MPI_Isend(boxes, MessageSize, MPI_INT, vOther, 0, MPI_COMM_WORLD, &request);
            }
            else
            { /* ... or the bits indicate it's our turn to receive */
                MPI_Recv(boxes, MessageSize, MPI_INT, vOther, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            }
            
        }
        else
        {
            // printf ( "[Node %d skipping iteration %d] flip: %x, mask: %x\n", self->id, i, flip, mask );
        }
        flip = flip >> 1;
        mask = mask ^ flip;
    }
}

/* Return the current time in seconds, using a double precision number. */
double when()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}



int main(int argc, char *argv[])
{
    double start = when(), finish = 0.0;
    int i;
    int root = 0;
    char host[255], message[55];
    MPI_Status status;
    
    // Initialize message array
    root = shell_arg_int(argc, argv, "--root", 0);
    MessageSize = shell_arg_int(argc, argv, "--msg-size", 10);
    Message = calloc( sizeof(int), MessageSize );
    for (i = 0; i < MessageSize; i++) Message[i] = i;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
    Dimensions = (int)ceil(sqrt((double)NumNodes));
    
    if (MyID == root)
    {
        printf("Nodes: %d\n", NumNodes);
        printf("Dimensionality: %d\n", Dimensions);
        printf("Vector size: %d\n", MessageSize);
        printf("Root node: %d\n", root);
        printf("Initial vector values for each node: ");
        print_message( Message, MessageSize );
    }
    
    gethostname(host, 253);
    // printf("I am proc %d of %d running on %s\n", MyID, NumNodes, host);
    
    reduce(root, Message);
    // broadcast(root, Message);
    
    MPI_Finalize();
    
    
    finish = when();

    if (MyID == root)
    {
        printf ("Final vector values for all nodes: " );
        print_message( Message, MessageSize );
        printf("Completed in %f seconds.\n", finish - start);
    }

    free(Message);
}
