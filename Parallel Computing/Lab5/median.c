#include <stdlib.h>
#include <stdio.h>

#include "median.h"

/*
 * The following code is public domain.
 * Algorithm by Torben Mogensen, implementation by N. Devillard.
 * This code in public domain.
 * 
 * http://ndevilla.free.fr/median/median/index.html
 * 
 * NOTE: Only works on values of magnitude 1/2 the maximum allowable for that type
 */

int median(int *m, int n)
{
    int i, less, greater, equal;
    int min, max, guess, maxltguess, mingtguess;

    min = max = m[0] ;
    for (i=1 ; i<n ; i++) {
        if (m[i]<min) min=m[i];
        if (m[i]>max) max=m[i];
    }
    while (1) {
        guess = (min+max)/2;
        less = 0; greater = 0; equal = 0;
        maxltguess = min ;
        mingtguess = max ;
        for (i=0; i<n; i++) {
            if (m[i]<guess) {
                less++;
                if (m[i]>maxltguess) maxltguess = m[i] ;
            } else if (m[i]>guess) {
                greater++;
                if (m[i]<mingtguess) mingtguess = m[i] ;
            } else equal++;
        }
        if (less <= (n+1)/2 && greater <= (n+1)/2) break ; 
        else if (less>greater) max = maxltguess ;
        else min = mingtguess;
    }
    if (less >= (n+1)/2) return maxltguess;
    else if (less+equal >= (n+1)/2) return guess;
    else return mingtguess;
}


int median_of_first( int* numbers, int size )
{
    if (size > 0)
        return numbers[0];
    else
        return 0;
}

int median_of_three( Array numbers )
{
    if (numbers.size > 0)
    {
        if (numbers.size > 2)
        {
            int a = 0;
            int m = numbers.size/2;
            int z = numbers.size-1;
            if (numbers.ptr[a] < numbers.ptr[m])
            {
                if (numbers.ptr[m] < numbers.ptr[z])      return numbers.ptr[m];
                else
                {
                    if (numbers.ptr[a] < numbers.ptr[z])  return numbers.ptr[z];
                    else                                  return numbers.ptr[a];
                }
            }
            else
            {
                if (numbers.ptr[m] > numbers.ptr[z])      return numbers.ptr[m];
                else
                {
                    if (numbers.ptr[a] > numbers.ptr[z])  return numbers.ptr[z];
                    else                                  return numbers.ptr[a];
                }
            }
        }
        else
        {
            return numbers.ptr[0];
        }
    }
    else
    {
        return 0;
    }
}

int median_random( Array numbers )
{
    if (numbers.size > 0)
        return numbers.ptr[random() % numbers.size];
    else
        return 0;
}