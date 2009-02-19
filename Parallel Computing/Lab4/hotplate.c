#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "misc.h"

int nproc, iproc;

int main(int argc, char *argv[])
{
    int i;
    double start = 0.0, finish = 0.0;
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    if (iproc == 0)
    {
        printf("Number of nodes: %d\n", nproc);
    }
    // All nodes:
    {
        printf("Node %d started...\n", iproc);
    }
    
    start = when();
    // Do our job:
    
    // Done job
    finish = when();
    
    MPI_Finalize();
    
    // All nodes:
    {
        printf( "Node %d completed in %f seconds.\n", iproc, finish - start );
    }
    
}
