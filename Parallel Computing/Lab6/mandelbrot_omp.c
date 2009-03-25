#include <stdio.h>
#include <omp.h>

#include "int_array.h"

double mandelbrot_width  = 500;
double mandelbrot_height = 500;
int    mandelbrot_iters  = 1000;

int mb(double x_not, double y_not, double mag)
{
    double cx = (x_not / mandelbrot_width - 0.5) / mag * 3.0 - 0.75;
    double cy = (y_not / mandelbrot_height - 0.5) / mag * 3.0;
    double x = 0.0, y = 0.0;
    double x_sq, y_sq;
    int i = 0;
    while ((x_sq = x*x) + (y_sq = y*y) <= 100.0 && i++ < mandelbrot_iters)
    {
        double xtemp = x_sq - y_sq + cx;
        y = 2 * x * y + cy;
        x = xtemp;
    }
    if (i >= mandelbrot_iters) return 0;
    else           return 1;
}

void mandelbrot_omp(int x_min, int x_max, int y_min, int y_max, double mag, IntArray arr)
{
    int w = x_max - x_min;
    int x, y;
    for (y = y_min; y < y_max; y++)
    {
        for (x = x_min; x < x_max; x++)
        {
            arr.ptr[(y - y_min) * w + (x - x_min)] = mb((double)x, (double)y, mag);
        }
    }
}