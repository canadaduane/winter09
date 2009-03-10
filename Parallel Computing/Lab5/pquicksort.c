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
    int* lesser;
    int lesser_size;
    int* greater;
    int greater_size;
} Div;

int nproc, iproc;

// Divides an array of ints into those that are less-than (or equal) and those that are greater-than
Div divide(int* numbers, int size, int median)
{
    Div d;
    d.median = median;
    int i = 0, j = size-1;
    int a;
    while( i < j ) {
        while( i < size && numbers[i] <= median ) i++;
        while( j > 0    && numbers[j] >  median ) j--;
        if ( i < j )
        {
            a = numbers[i];
            numbers[i++] = numbers[j];
            numbers[j--] = a;
        }
        else
        {
            break;
        }
    }
    d.lesser = numbers;
    d.lesser_size = i;
    d.greater = numbers + i;
    d.greater_size = size - i;
    return d;
}

void show_div(int iproc, Div d)
{
    int i;
    fprintf(stderr, "Division for iproc %d:", iproc);
    fprintf(stderr, "\n\tLess than %d (%d): ", d.median, d.lesser_size);
    for( i = 0; i < d.lesser_size; i++) fprintf(stderr, "%d, ", d.lesser[i]);
    fprintf(stderr, "\n\tGreater than %d (%d): ", d.median, d.greater_size);
    for( i = 0; i < d.greater_size; i++) fprintf(stderr, "%d, ", d.greater[i]);
    fprintf (stderr, "\n");
}

void hypercube( int nproc, int original_iproc, int* original_numbers, int original_size )
{
    MPI_Request request;
    MPI_Status status;
    MPI_Comm comm = MPI_COMM_WORLD;
    int iproc = original_iproc;
    // Number of dimensions e.g. 
    int dim = floor_log2( nproc );
    // The allocated space of our original_numbers array (grows as needed)
    int max_size = original_size;
    // A temporary location to receive the "incoming message length"
    int tmp_size = 0;
    // The offset between the start of the original_numbers pointer and our "useful data"
    int offset = 0;
    // The "current size" of our "useful data"
    int size = original_size;
    // A pointer to the "useful data"
    int* numbers = original_numbers;
    int median;
    int i;
    
    // fprintf(stderr, "Node %d has %d random numbers.\n", iproc, original_size);
    
    for ( i = 0; i < dim; i++ )
    {
        int dest = iproc ^ 1;
        Div divided;
        
        median = median_of_three( numbers, size );
        MPI_Bcast (&median, 1, MPI_INT, 0, comm);
        
        // fprintf(stderr, "Node %d dividing %d numbers using median %d...\n", iproc, size, median);
        divided = divide( numbers, size, median );
        // show_div(iproc, divided);
        
        if (iproc % 2 == 0)
        {
            offset  = (int)(divided.lesser - original_numbers);
            size    = divided.lesser_size;
            numbers = divided.lesser;
            // fprintf(stderr, "Node %d sending to %d, size %d (greater)\n", iproc, dest, divided.greater_size);
            MPI_Isend(&(divided.greater_size), 1, MPI_INT,
                      dest, MSG_SIZE,    comm, &request);
            MPI_Isend(  divided.greater,  divided.greater_size, MPI_INT,
                      dest, MSG_NUMBERS, comm, &request);
        }
        else
        {
            offset  = (int)(divided.greater - original_numbers);
            size    = divided.greater_size;
            numbers = divided.greater;
            // fprintf(stderr, "Node %d sending to %d, size %d (lesser)\n", iproc, dest, divided.lesser_size);
            MPI_Isend(&(divided.lesser_size), 1, MPI_INT,
                      dest, MSG_SIZE,    comm, &request);
            MPI_Isend(  divided.lesser,  divided.lesser_size, MPI_INT,
                      dest, MSG_NUMBERS, comm, &request);
        }
        
        // fprintf(stderr, "Node %d receiving from %d\n", iproc, dest);
        MPI_Recv(&tmp_size,      1, MPI_INT,
                 dest, MSG_SIZE,    comm, &status);
        // if (tmp_size > max_size)
        // {
            int* new_pointer =
                realloc( original_numbers,
                        (offset + size + tmp_size) * sizeof(int) );
            if (new_pointer != original_numbers)
            {
                numbers = new_pointer + offset;
                original_numbers = new_pointer;
            }
        // }
        MPI_Recv(numbers + size, tmp_size, MPI_INT,
                 dest, MSG_NUMBERS, comm, &status);
        size += tmp_size;
        
        // Inherit half of the communicator
        MPI_Comm_split(comm, iproc % 2, iproc >> 1, &comm);
        iproc = iproc >> 1;
    }
    
    fprintf(stderr, "Final numbers for node %d: (%d)\n\t", original_iproc, size);
    for( i = 0; i < size; i++) fprintf(stderr, "%d, ", numbers[i]);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    // Seed the random number generator with our processor ID
    srandom(iproc);
    
    {
        int i;
        double start = 0.0, finish = 0.0;
        int all_size = shell_arg_int(argc, argv, "--size", 16);
        int *all_numbers = alloc_n_random( all_size );
        
        if (iproc == 0)
        {
            fprintf( stderr, "Sorting %d numbers total...\n", nproc * all_size);

            fprintf(stderr, "Node %d started...\n", iproc);
            // for (i = 0; i < all_size; i++) fprintf(stderr, "%d, ", all_numbers[i]);
            // fprintf(stderr, "\n");
        }
        
        start = when();
        hypercube( nproc, iproc, all_numbers, all_size );
        finish = when();
        
        // All nodes:
        fprintf(stderr, "Node %d completed in %f seconds.\n", iproc, finish - start);
    }

    
    MPI_Finalize();
    
    return 0;
}
