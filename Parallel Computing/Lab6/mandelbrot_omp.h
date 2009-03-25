#ifndef MANDELBROT_OMP_H
#define MANDELBROT_OMP_H

#include "int_array.h"

extern double mandelbrot_width;
extern double mandelbrot_height;
extern int    mandelbrot_iters;

void mandelbrot_omp(int x_min, int x_max, int y_min, int y_max, double mag, IntArray arr);

#endif