#ifndef MANDELBROT_OMP_H
#define MANDELBROT_OMP_H

#include "int_array.h"

extern int    mandelbrot_iters;

void mandelbrot_omp(int   img_x,   int   img_w,
                    int   img_y,   int   img_h,
                    float mdb_x,   float mdb_w,
                    float mdb_y,   float mdb_h,
                    IntArray arr,
                    int procs);

#endif