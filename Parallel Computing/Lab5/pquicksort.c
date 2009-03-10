#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "misc.h"
#include "median.h"

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

int main(int argc, char *argv[])
{
    int size_param;
    int iter = 0;
    double start = 0.0, finish = 0.0;
    size_param = shell_arg_int(argc, argv, "--size", 8);
    int size = 1 << (size_param - 1);
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    if (iproc == 0)
        fprintf( stderr, "Sorting %d numbers total...\n", nproc * size);
    
    // Seed the random number generator with our processor ID
    srandom(iproc);
    
    printf("Node %d started...\n", iproc);
    
    start = when();
    
    {
        MPI_Status status;
        int *numbers;
        
        numbers = alloc_n_random( size );
        fprintf(stderr, "Node %d has %d random numbers.\n", iproc, size);
        
        // fprintf(stderr, "Node %d numbers: ", iproc);
        // for (iter = 0; iter < size; iter++) fprintf(stderr, "%d, ", numbers[iter]);
        fprintf(stderr, "Node %d's median: %d\n", iproc, median(numbers, size));
        
        
    }
    
    finish = when();
    
    MPI_Finalize();
    
    // All nodes:
    printf("Node %d completed %d iterations in %f seconds.\n", iproc, iter, finish - start);
    
    return 0;
}
