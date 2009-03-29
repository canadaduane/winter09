#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#include "int_array.h"
#include "misc.h"
#include "mandelbrot_omp.h"

const float Left   = -1.5;
const float Right  =  1.5;
const float Top    = -1.5;
const float Bottom = -1.5;
const float Width  = 3.0;
const float Height = 3.0;

int nproc, iproc;

void show_array(IntArray a, int w);

int main( int argc, char** argv )
{
    // double mag = shell_arg_float(argc, argv, "-m", 2.0);
    
    // Configure global variables for mandelbrot size
    int img_width  = shell_arg_int(argc, argv, "-w", 40);
    int img_height = shell_arg_int(argc, argv, "-h", 40);
    mandelbrot_iters = shell_arg_int(argc, argv, "-i", 1000);
    
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
            int rem  = img_width % nproc;
            int my_img_w = img_width / nproc + (iproc < rem ? 1 : 0);
            int my_img_x = img_width / nproc * iproc + (iproc < rem ? iproc : rem);
            float my_mdb_x = Left + (Width / img_width) * my_img_x;
            float my_mdb_w = (Width / img_width) * my_img_w;
            // Allocate space for result
            IntArray result = ia_alloc2(my_img_w, img_height);
            
            fprintf(stderr, "%d -- x: %d, y: 0, dim: %d x %d\n", iproc, my_img_x, my_img_w, img_height);
            
            mandelbrot_omp(my_img_x, my_img_w,     // Image coords
                           0, img_height,
                           
                           my_mdb_x, my_mdb_w,     // Mandelbrot coords
                           Top, Height,
                           
                           result);
            
            if (iproc == 0)
            {
                fprintf(stderr, "%d --\n", iproc);
                show_array(result, my_img_w);
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