#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "misc.h"
#include "parallel_mandelbrot.h"

int nproc, iproc;

int main( int argc, char** argv )
{
    int x = shell_arg_int(argc, argv, "-x", 0);
    int y = shell_arg_int(argc, argv, "-y", 0);
    int w = shell_arg_int(argc, argv, "-w", 500);
    int h = shell_arg_int(argc, argv, "-h", 500);
    double mag = shell_arg_float(argc, argv, "-m", 2.0);
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    if (iproc == 0)
    {
        printf("Starting parallel mandelbrot calculation...");
        printf("x: %d, y: %d, dimensions: %d x %d, magnification: %0.2f\n", x, y, w, h, mag);
    }
    
    {
        float start, finish;
        
        fprintf(stderr, "Node %d started...\n", iproc);
        
        start = when();
        {
            int rem = w % nproc;
            int my_w = w / nproc + (iproc < rem ? 1 : 0);
            int my_x = w / nproc * iproc + (iproc < rem ? iproc : rem);
            
            // fprintf(stderr, "%d -- my_w: %d, my_x: %d\n", iproc, my_w, my_x);
            
            parallel_mandelbrot()
        }
        finish = when();
        
        // All nodes:
        fprintf(stderr, "Node %d completed in %f seconds.\n", iproc, finish - start);
    }

    
    MPI_Finalize();
    
    return 0;
}
