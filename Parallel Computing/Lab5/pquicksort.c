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
    int median;
    Array lesser;
    Array greater;
} Div;

int nproc, iproc;

/* qsort integer comparison: returns negative if b > a and positive if a > b */
int int_cmp(const void *a, const void *b)
{
    const int *ia = (const int *)a; // casting pointer types
    const int *ib = (const int *)b;
    return *ia  - *ib; 
}

// Divides an array of ints into those that are less-than (or equal) and those that are greater-than
Div divide(Array numbers, int median)
{
    Div d;
    d.median = median;
    int i = 0, j = numbers.size-1;
    int a;
    while( i < j ) {
        while( i < numbers.size && numbers.ptr[i] <= median ) i++;
        while( j > 0            && numbers.ptr[j] >  median ) j--;
        if ( i < j )
        {
            a = numbers.ptr[i];
            numbers.ptr[i++] = numbers.ptr[j];
            numbers.ptr[j--] = a;
        }
        else
        {
            break;
        }
    }
    d.lesser.ptr = numbers.ptr;
    d.lesser.size = i;
    d.greater.ptr = numbers.ptr + i;
    d.greater.size = numbers.size - i;
    return d;
}

void show_div(int iproc, Div d)
{
    int i;
    fprintf(stderr, "Division for iproc %d:", iproc);
    fprintf(stderr, "\n\tLess than %d (%d): ", d.median, d.lesser.size);
    for( i = 0; i < d.lesser.size; i++) fprintf(stderr, "%d, ", d.lesser.ptr[i]);
    fprintf(stderr, "\n\tGreater than %d (%d): ", d.median, d.greater.size);
    for( i = 0; i < d.greater.size; i++) fprintf(stderr, "%d, ", d.greater.ptr[i]);
    fprintf (stderr, "\n");
}

Array hypercube( int nproc, int iproc, Array numbers )
{
    MPI_Request request;
    MPI_Status status;
    MPI_Comm comm = MPI_COMM_WORLD;
    int vproc = iproc;
    int median;
    int i;

    // Number of dimensions e.g. 
    int dim = floor_log2( nproc );
    // A temporary location to receive the "incoming message length"
    int tmp_size = 0;
    
    // Give ourselves a big buffer to work with
    int allocated_size = numbers.size * 10;
    numbers.ptr = realloc(numbers.ptr, allocated_size * sizeof(int));
    
    // fprintf(stderr, "Node %d has %d random numbers.\n", iproc, original_size);
    
    for ( i = 0; i < dim; i++ )
    {
        int dest = vproc ^ 1;
        Div divided;
        
        median = median_of_three( numbers );
        MPI_Bcast (&median, 1, MPI_INT, 0, comm);
        
        // fprintf(stderr, "Node %d dividing %d numbers using median %d...\n", vproc, size, median);
        divided = divide( numbers, median );
        // show_div(vproc, divided);
        
        if (vproc % 2 == 0)
        {
            // fprintf(stderr, "Node %d sending to %d, size %d (greater)\n", vproc, dest, divided.greater_size);
            MPI_Isend(&(divided.greater.size), 1, MPI_INT,
                      dest, MSG_SIZE,    comm, &request);
            MPI_Isend(  divided.greater.ptr,  divided.greater.size, MPI_INT,
                      dest, MSG_NUMBERS, comm, &request);
            // New size is just the size of our "lesser" half
            numbers.size = divided.lesser.size;
        }
        else
        {
            // fprintf(stderr, "Node %d sending to %d, size %d (lesser)\n", vproc, dest, divided.lesser_size);
            MPI_Isend(&(divided.lesser.size), 1, MPI_INT,
                      dest, MSG_SIZE,    comm, &request);
            MPI_Isend(  divided.lesser.ptr,  divided.lesser.size, MPI_INT,
                      dest, MSG_NUMBERS, comm, &request);
            // New size is just the size of our "greater" half
            numbers.size = divided.greater.size;
            // Move the array of greater numbers back to the memory location origin
            memmove( numbers.ptr, divided.greater.ptr, divided.greater.size );
        }
        
        // fprintf(stderr, "Node %d receiving from %d\n", vproc, dest);
        MPI_Recv(&tmp_size,      1, MPI_INT,
                 dest, MSG_SIZE,    comm, &status);
        if (numbers.size + tmp_size > allocated_size)
        {
            // int new_size = size + tmp_size + 1;
            // fprintf(stderr, "Node %d reallocating from %d to %d\n", iproc, allocated_size, new_size);
            // numbers = realloc( numbers, new_size * sizeof(int) );
            // if (numbers == NULL)
            // {
                fprintf(stderr, "Cannot allocate more memory.\n");
                exit(1);
            // }
            // else
            // {
            //     allocated_size = new_size;
            // }
        }
        // fprintf(stderr, "Node %d about to receive %d ints starting at %d...\n", iproc, tmp_size, numbers.size);
        MPI_Recv(numbers.ptr + numbers.size, tmp_size, MPI_INT,
                 dest, MSG_NUMBERS, comm, &status);
        numbers.size += tmp_size;
        
        // Inherit half of the communicator
        MPI_Comm_split(comm, vproc % 2, vproc >> 1, &comm);
        vproc = vproc >> 1;
    }
    
    // fprintf(stderr, "Final numbers for node %d: (%d)\n\t", iproc, size);
    // for( i = 0; i < size; i++) fprintf(stderr, "%d, ", numbers[i]);

    return numbers;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    // Seed the random number generator with our processor ID
    srandom(iproc);
    
    {
        // int i;
        double start = 0.0, finish = 0.0;
        Array numbers, semisorted;
        
        numbers.size = shell_arg_int(argc, argv, "--size", 16);
        numbers.ptr = alloc_n_random( numbers.size );
        
        if (iproc == 0)
        {
            fprintf( stderr, "Sorting %d numbers total...\n", nproc * numbers.size);

            fprintf(stderr, "Node %d started...\n", iproc);
            // for (i = 0; i < all_size; i++) fprintf(stderr, "%d, ", all_numbers[i]);
            // fprintf(stderr, "\n");
        }
        
        start = when();
        semisorted = hypercube( nproc, iproc, numbers );
        qsort(semisorted.ptr, semisorted.size, sizeof(int), int_cmp);
        
        finish = when();
        
        // All nodes:
        fprintf(stderr, "Node %d completed in %f seconds.\n", iproc, finish - start);
    }

    
    MPI_Finalize();
    
    return 0;
}
