#ifndef MEDIAN_H
#define MEDIAN_H

typedef struct ARRAY {
    int* ptr;
    int size;
} Array;

int median(int *numbers, int size);
int median_of_first( int* numbers, int size );
int median_of_three( Array numbers );
int median_random( Array numbers );

#endif