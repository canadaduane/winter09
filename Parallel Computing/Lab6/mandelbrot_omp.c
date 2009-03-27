#include <stdio.h>
#include <omp.h>

#include "int_array.h"

float  mandelbrot_width  = 500;
float  mandelbrot_height = 500;
int    mandelbrot_iters  = 1000;

int mb(float x_not, float y_not, float mag)
{
    float cx = (x_not / mandelbrot_width - 0.5) / mag * 3.0 - 0.75;
    float cy = (y_not / mandelbrot_height - 0.5) / mag * 3.0;
    float x = 0.0, y = 0.0;
    float x_sq, y_sq;
    int i = 0;
    while ((x_sq = x*x) + (y_sq = y*y) <= 100.0 && i++ < mandelbrot_iters)
    {
        float xtemp = x_sq - y_sq + cx;
        y = 2 * x * y + cy;
        x = xtemp;
    }
    if (i >= mandelbrot_iters) return 0;
    else           return 1;
}

void mandelbrot_omp(float x_min, float x_max, float x_inc,
                    float y_min, float y_max, float y_inc,
                    float mag, IntArray arr)
{
    float w = x_max - x_min;
    float x, y;
    for (y = y_min; y < y_max; y += y_inc)
    {
        for (x = x_min; x < x_max; x += x_inc)
        {
            arr.ptr[(y - y_min) * w + (x - x_min)] = mb(x, y, mag);
        }
    }
}