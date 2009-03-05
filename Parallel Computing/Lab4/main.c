#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#include "misc.h"
#include "hotplate.h"

#define WIDTH 768
#define HEIGHT 768
#define MSG_SLICE_START 0
#define MSG_SLICE_END 1
#define MSG_DONE 2

int nproc, iproc;
Hotplate *plate;

int main(int argc, char *argv[])
{
    int iter = 0;
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
        // int i;
        MPI_Status status;
        printf("Node %d started...\n", iproc);
        start = when();
        // Do our job:
        
        plate = hp_initialize(WIDTH, HEIGHT);
        plate->iproc = iproc;
        plate->nproc = nproc;
        hp_slice(plate, nproc, iproc);

        hp_fill(plate, 50.0);
        hp_hline(plate, 0,        0, WIDTH-1,    0.0);
        hp_hline(plate, HEIGHT-1, 0, WIDTH-1,  100.0);
        hp_vline(plate, 0,        0, HEIGHT-1,   0.0);
        hp_vline(plate, WIDTH-1,  0, HEIGHT-1,   0.0);
        hp_etch_hotspots(plate);
        hp_copy_to_source(plate);
        
        int done = FALSE;
        int recv_done;
        while (!done)
        {
            // fprintf(stderr, "[%d] Node: %d\n", iter, iproc);
            
            // Send top row up
            if (iproc > 0)
                MPI_Send(hp_slice_start_row(plate), plate->width, MPI_FLOAT, iproc-1, MSG_SLICE_START, MPI_COMM_WORLD);
            // Send bottom row down
            if (iproc < nproc-1)
                MPI_Send(hp_slice_end_row(plate), plate->width, MPI_FLOAT, iproc+1, MSG_SLICE_END, MPI_COMM_WORLD);
            
            // Receive top row from above
            if (iproc > 0)
                MPI_Recv(hp_slice_start_row(plate) - plate->width, plate->width, MPI_FLOAT, iproc-1, MSG_SLICE_END, MPI_COMM_WORLD, &status);
            // Receive bottom row from below
            if (iproc < nproc-1)
                MPI_Recv(hp_slice_end_row(plate) + plate->width, plate->width, MPI_FLOAT, iproc+1, MSG_SLICE_START, MPI_COMM_WORLD, &status);
            
            hp_slice_heat(plate);
            hp_etch_hotspots(plate);
            
            done = hp_is_steady_state(plate);
            MPI_Allreduce (&done, &recv_done, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
            done = recv_done;
            
            if (done)
                printf("%d reached steady state (%d)\n", iproc, iter);
            else
                iter++;
            
            hp_swap(plate);
        }
        
        hp_destroy(plate);
        
        // Done job
        finish = when();
    }
    
    
    MPI_Finalize();
    
    // All nodes:
    {
        printf("Node %d completed %d iterations in %f seconds.\n", iproc, iter, finish - start);
        // printf("%d cells had a value greater than 50.0.\n", gt_count);
    }
    
    return 0;
}
