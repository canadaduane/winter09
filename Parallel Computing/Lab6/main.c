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
void write_image(IntArray a, int w);

int main( int argc, char** argv )
{
    int* final_result;
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
        final_result = malloc(img_width * img_height * sizeof(int));
    }
    
    {
        float start, finish;
        int* sizes;
        int* offsets;
        
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
            // Allocate sizes array for root node
            if (iproc == 0)
            {
                sizes   = malloc(nproc * sizeof(int));
                offsets = malloc(nproc * sizeof(int));
            }
            // Let the root node know how big each result is
            MPI_Gather(&result.size, 1, MPI_INT,
                       sizes, 1, MPI_INT,
                       0, MPI_COMM_WORLD);
            
            if (iproc == 0)
            {
                int i;
                offsets[0] = 0;
                for (i = 0; i < nproc; i++)
                {
                    offsets[i + 1] = offsets[i] + sizes[i];
                }
            }
            MPI_Gatherv(result.ptr, result.size, MPI_INT,
                        final_result, sizes, offsets, MPI_INT,
                        0, MPI_COMM_WORLD);
            
            if (iproc == 0)
            {
                int i;
                IntArray a;
                
                a.ptr = final_result;
                a.size = img_width * img_height;
                
                write_image(a, img_width);
            }
        }
        finish = when();
        
        if (iproc == 0)
        {
            free(final_result);
        }
        
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

void write_image(IntArray a, int w)
{
    int y, x, h = a.size / w;
    FILE* out = fopen("image.ppm", "w");
    fprintf(out, "P3 %d %d 255\n", w, h);
    for (y = 0; y < h; y++)
    {
        for (x = 0; x < w; x++)
        {
            int c = a.ptr[y * w + x];
            fprintf(out, "%d\t%d\t%d\t\t", c, c, c);
        }
        fprintf(out, "\n");
    }
}
