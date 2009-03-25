#include <stdio.h>
#include <omp.h>

#include "mandelbrot.h"
#include "int_array.h"

double x_max = 500;
double y_max = 500;
int i_max = 1000;

int mb(double x_not, double y_not, double mag)
{
    double cx = (x_not / x_max - 0.5) / mag * 3.0 - 0.75;
    double cy = (y_not / y_max - 0.5) / mag * 3.0;
    double x = 0.0, y = 0.0;
    double x_sq, y_sq;
    int i = 0;
    while ((x_sq = x*x) + (y_sq = y*y) <= 100.0 && i++ < i_max)
    {
        double xtemp = x_sq - y_sq + cx;
        y = 2 * x * y + cy;
        x = xtemp;
    }
    if (i >= i_max) return 0;
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
            arr.ptr[y * w + x] = mb((double)x, (double)y, mag);
        }
    }
}