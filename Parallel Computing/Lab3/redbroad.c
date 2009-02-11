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
    
    printf("MyID: %d, vMyID: %d\n", MyID, vMyID);
    
    for ( i = 0; i < Dimensions; i++)
    {
        int other = vMyID ^ (1<<i);
        int vOther = (other + root) % (1<<Dimensions);
        
        printf( "%d, %d, mask: %x\n", vMyID, vOther, mask);
        if ( (vMyID & mask) == 0 && vMyID < NumNodes && vOther < NumNodes )
        {
            if ( (vMyID & (1<<i)) != 0)
            {
                printf("%d send: ", vMyID);
                print_message(sum, MessageSize);
                MPI_Isend(sum, MessageSize, MPI_INT, vOther, 0, MPI_COMM_WORLD, &request);
            }
            else
            {
                MPI_Recv(tmp, MessageSize, MPI_INT, vOther, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                printf("%d recv: ", vMyID);
                print_message(tmp, MessageSize);
                for ( j = 0; j < MessageSize; j++) sum[j] += tmp[j];
            }
        }
        mask = mask ^ (1<<i);
    }
    
    memcpy( boxes, sum, sizeof(int) * MessageSize );
    
    free(sum);
    free(tmp);
}



int main(int argc, char *argv[])
{
    int i;
    char host[255], message[55];
    MPI_Status status;
    
    // Initialize message array
    MessageSize = shell_arg_int(argc, argv, "--msg-size", 10);
    Message = calloc( sizeof(int), MessageSize );
    for (i = 0; i < MessageSize; i++)
    {
        Message[i] = i;
    }
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NumNodes);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyID);
    Dimensions = (int)ceil(sqrt((double)NumNodes));
    
    if (MyID == 0)
    {
        printf("Nodes: %d\n", NumNodes);
        printf("Dimensionality: %d\n", Dimensions);
    }
    
    gethostname(host, 253);
    printf("I am proc %d of %d running on %s\n", MyID, NumNodes, host);
    
    reduce(0, Message);
    
    printf ("Final values for %d: ", MyID);
    print_message( Message, MessageSize );

    MPI_Finalize();
    
    free(Message);
}
