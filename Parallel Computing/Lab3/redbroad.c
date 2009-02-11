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
void reduce(int* initial_values)
{
    int i;
    int mask = 0;
    MPI_Request request;
    MPI_Status status;
    
    int* sum = calloc( sizeof(int), MessageSize );
    int* tmp = calloc( sizeof(int), MessageSize );
    memcpy( sum, initial_values, sizeof(int) * MessageSize );
    
    for ( i = 0; i < Dimensions; i++)
    {
        int other = MyID ^ (1<<i);
        if ( (MyID & mask) == 0 && other < NumNodes )
        {
            if ( (MyID & (1<<i)) != 0)
            {
                printf( "%d send: ", MyID);
                print_message( sum, MessageSize );
                MPI_Isend(sum, MessageSize, MPI_INT, other, 0, MPI_COMM_WORLD, &request);
            }
            else
            {
                int j;
                MPI_Recv(tmp, MessageSize, MPI_INT, other, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                printf( "%d recv: ", MyID);
                print_message( tmp, MessageSize );
                for ( j = 0; j < MessageSize; j++)
                {
                    sum[j] += tmp[j];
                }
                printf( "%d sum: ", MyID);
                print_message( sum, MessageSize );
            }
        }
        mask = mask ^ (1<<i);
    }
    
    free(sum);
    free(tmp);
    
    // Procedure SingleNodeAccum(d, MyID, m, X, sum)
    //     for j = 0 to m-1 sum[j] = X[j];
    //     mask = 0
    //     for i = 0 to d-1
    //  if ((MyID AND mask) == 0)
    //      if ((MyID AND 2^i) != 0
    //      msg_dest = MyID XOR 2^i
    //      send(sum, msg_dest)
    //      else
    //      msg_src = MyID XOR 2^i
    //      recv(sum, msg_src)
    //      for j = 0 to m-1
    //          sum[j] += X[j]
    //      endif
    //  endif
    //  mask = mask XOR 2^i
    //     endfor
    // end
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
    
    gethostname(host,253);
    printf("I am proc %d of %d running on %s\n", MyID, NumNodes, host);
    
    reduce(Message);
    
    // printf("%d sending messages\n", MyID);
    // sprintf(message, "%d: Hello", MyID);
    // for (i = 0; i < NumNodes; i++)
    // {
    //     if (i != MyID)
    //         MPI_Send(message, 35, MPI_CHAR, i, 0, MPI_COMM_WORLD);
    // }
    // printf("%d receiving messages\n", MyID);
    // for (i = 0; i < NumNodes; i++)
    // {
    //     if (i != MyID)
    //     {
    //         MPI_Recv(message, 35, MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
    //         printf("%d recv \"%s\"\n",MyID, message);
    //     }
    // }

    MPI_Finalize();
    
    free(Message);
}
