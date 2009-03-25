#ifndef MANDELBROT_OMP_H
#define MANDELBROT_OMP_H

#include "int_array.h"

extern double x_max;
extern double y_max;
extern int i_max;

void mandelbrot_omp(int x_min, int x_max, int y_min, int y_max, double mag, IntArray arr);

#endif