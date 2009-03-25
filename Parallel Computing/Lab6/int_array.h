#ifndef INT_ARRAY_H
#define INT_ARRAY_H

typedef struct INT_ARRAY {
    int* ptr;
    int size;
} IntArray;

IntArray ia_alloc2(int x, int y);

#endif