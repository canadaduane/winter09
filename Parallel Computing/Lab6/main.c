#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "int_array.h"
#include "misc.h"
#include "mandelbrot_omp.h"

int nproc, iproc;
void show_array(IntArray a, int w);

int main( int argc, char** argv )
{
    double mag = shell_arg_float(argc, argv, "-m", 2.0);
    
    // Configure global variables for mandelbrot size
    int width  = shell_arg_int(argc, argv, "-w", 500);
    int height = shell_arg_int(argc, argv, "-h", 500);
    
    mandelbrot_width  = (double)width;
    mandelbrot_height = (double)height;
    
    MPI_Init(&argc, &argv);
    
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
    
    if (iproc == 0)
    {
        printf("Starting parallel mandelbrot calculation...");
    }
    
    {
        float start, finish;
        
        fprintf(stderr, "Node %d started...\n", iproc);
        
        start = when();
        {
            int rem  = width % nproc;
            int my_w = width / nproc + (iproc < rem ? 1 : 0);
            int my_x = width / nproc * iproc + (iproc < rem ? iproc : rem);
            // Allocate space for result
            IntArray result = ia_alloc2(my_w, height);
            
            fprintf(stderr, "%d -- x: %d, y: 0, dim: %d x %d\n", iproc, my_x, my_w, height);
            
            mandelbrot_omp(my_x, my_x + my_w, 0, mandelbrot_height, mag, result);
            if (iproc == 1)
            {
                fprintf(stderr, "%d --\n", iproc);
                show_array(result, my_w);
            }
        }
        finish = when();
        
        // All nodes:
        fprintf(stderr, "Node %d completed in %f seconds.\n", iproc, finish - start);
    }

    
    MPI_Finalize();
    
    return 0;
}

void show_array(IntArray a, int w)
{
    int i;
    for (i = 0; i < a.size; i++)
    {
        fprintf(stderr, "%d ", a.ptr[i]);
        if (i % w == w - 1) fprintf(stderr, "\n");
    }
}