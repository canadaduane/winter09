#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "int_array.h"

int mandelbrot_iters = 1000;
double log_max = 2.7089236388165242e-2; // log((double)1000)/255;

int mandelbrot(float x_not, float y_not)
{
    // float cx = (x_not / mandelbrot_width - 0.5) / mag * 3.0 - 0.75;
    // float cy = (y_not / mandelbrot_height - 0.5) / mag * 3.0;
    float x = 0.0, y = 0.0;
    float x_sq, y_sq;
    int i = 0;
    while ((x_sq = x*x) + (y_sq = y*y) <= 100.0 && i++ < mandelbrot_iters)
    {
        float xtemp = x_sq - y_sq + x_not;
        y = 2 * x * y + y_not;
        x = xtemp;
    }
    return (int)(log((double)i) / log_max);
}

void mandelbrot_omp(int   img_x,   int   img_w,
                    int   img_y,   int   img_h,
                    float mdb_x,   float mdb_w,
                    float mdb_y,   float mdb_h,
                    IntArray arr)
{
    int x, y;
    float mx, my = mdb_y;
    float xi = mdb_w / img_w, yi = mdb_h / img_h;
    
    for (y = 0; y < img_h; y++)
    {
        mx = mdb_x;
        for (x = 0; x < img_w; x++)
        {
            arr.ptr[y * img_w + x] = mandelbrot(mx, my);
            mx += xi;
        }
        my += yi;
    }
}