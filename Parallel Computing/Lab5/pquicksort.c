#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "misc.h"
#include "median.h"

#define MSG_SIZE    0
#define MSG_NUMBERS 1

typedef struct DIV {
    int* less_than;
    int less_size;
    int* greater_than;
    int greater_size;
} Div;

int nproc, iproc;

// Divides an array of ints into those that are less-than (or equal) and those that are greater-than
Div divide(int* numbers, int size, int value)
{
    Div d;
    int i = 0, j = size-1;
    int a;
    while( 1 ) {
        while( i < size && numbers[i] <= value ) i++;
        while( j > 0    && numbers[j] >  value ) j--;
        if ( i < j )
        {
            a = numbers[i];
            numbers[i++] = numbers[j];
            numbers[j++] = a;
        }
        else
        {
            break;
        }
    }
    d.less_than = numbers;
    d.less_size = i;
    d.greater_than = numbers + i;
    d.greater_size = size - i;
    return d;
}

void show_div(Div d)
{
    int i;
    for( i = 0; i < size; i++) printf("%d, ", numbers[i]);
    printf("\n");
    printf("\tLess than (%d): ", d.less_size);
    for( i = 0; i < d.less_size; i++) printf("%d, ", d.less_than[i]);
    printf("\n\tGreater than (%d): ", d.greater_size);
    for( i = 0; i < d.greater_size; i++) printf("%d, ", d.greater_than[i]);
    printf ("\n");
}

void reduce( int nproc, int iproc, int* numbers, int size )
{
    MPI_Request request;
    MPI_Status status;
    MPI_Comm new_comm;
    int dim = (int) ceil(sqrt( (double) nproc ));
    int median;
    int i;
    
    // Allocate buffer for receiving values
    int tmp_max_size = size;
    int tmp_size = 0;
    int* tmp_numbers = calloc( tmp_max_size, sizeof(int) );
    
    fprintf(stderr, "Node %d has %d random numbers.\n", iproc, all_size);
    
    for ( i = 0; i < dim; i++ )
    {
        int dest = iproc ^ (1 << i);
        
        median = median_of_three( numbers, size );
        divided = divide( numbers, size, median );
        
        MPI_Isend(&size,      sizeof(int), MPI_INT,
                  dest, MSG_SIZE,    MPI_COMM_WORLD, &request);
        MPI_Isend(numbers,    sizeof(int), MPI_INT,
                  dest, MSG_NUMBERS, MPI_COMM_WORLD, &request);
        
        MPI_Recv(&tmp_size,   sizeof(int), MPI_INT,
                 dest, MSG_SIZE,    MPI_COMM_WORLD, &status);
        if (tmp_size > tmp_max_size) 
            tmp_numbers = realloc( tmp_numbers, tmp_size * sizeof(int) );
        MPI_Recv(tmp_numbers, sizeof(int), MPI_INT,
                 dest, MSG_NUMBERS, MPI_COMM_WORLD, &status);
        
        // other = iproc & 0x01
        // MPI_Comm_split(MPI_COMM_WORLD, color, 0, &new_comm);
    }

}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    // Seed the random number generator with our processor ID
    srandom(iproc);
    
    {
        double start = 0.0, finish = 0.0;
        int all_size = shell_arg_int(argc, argv, "--size", 16);
        int *all_numbers = alloc_n_random( all_size );
        
        if (iproc == 0) fprintf( stderr, "Sorting %d numbers total...\n", nproc * all_size);
        
        printf("Node %d started...\n", iproc);
        
        start = when();
        reduce( nproc, iproc, all_numbers, all_size );
        finish = when();
        
        // All nodes:
        printf("Node %d completed %d iterations in %f seconds.\n", iproc, iter, finish - start);
    }

    
    MPI_Finalize();
    
    return 0;
}
