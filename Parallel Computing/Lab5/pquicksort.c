#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "misc.h"

int nproc, iproc;

int main(int argc, char *argv[])
{
    int size_param;
    int iter = 0;
    double start = 0.0, finish = 0.0;
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    size_param = shell_arg_int(argc, argv, "--size", 8);
    
    printf("Node %d started...\n", iproc);

    start = when();

    {
        MPI_Status status;
        int size = 1 << (size - 1);
        
        fprintf(stderr, "Size: %d\n", size);
        
        // quicksort
    }

    finish = when();
    
    MPI_Finalize();
    
    // All nodes:
    printf("Node %d completed %d iterations in %f seconds.\n", iproc, iter, finish - start);
    
    return 0;
}
