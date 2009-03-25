#include <stdlib.h>

#include "int_array.h"

IntArray ia_alloc2(int x, int y)
{
    IntArray a;
    a.ptr = malloc(x*y*sizeof(int));
    a.size = x*y;
    return a;
}