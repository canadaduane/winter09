#ifndef MEDIAN_H
#define MEDIAN_H

typedef struct ARRAY {
    int* ptr;
    int size;
} Array;

int median( Array numbers );
int median_of_first( Array numbers );
int median_of_three( Array numbers );
int median_random( Array numbers );

#endif