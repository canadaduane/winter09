#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "hotplate.h"

Hotplate* hp_initialize(int w, int h)
{
    Hotplate* self = malloc(sizeof(Hotplate));
    assert(self != NULL);
    self->width = w;
    self->height = h;
    
    /* Allocate matrix 1 */
    self->matrix_data1 = malloc(sizeof(float) * w * h);
    assert(self->matrix_data1 != NULL);
    self->src_matrix = self->matrix_data1;
    
    /* Allocate matrix 2 */
    self->matrix_data2 = malloc(sizeof(float) * w * h);
    assert(self->matrix_data2 != NULL);
    self->dst_matrix = self->matrix_data2;
    
    return self;
}

void hp_destroy(Hotplate* self)
{
    assert(self != NULL);
    assert(self->matrix_data1 != NULL);
    assert(self->matrix_data2 != NULL);
    free(self->matrix_data1);
    free(self->matrix_data2);
    free(self);
}

void hp_slice(Hotplate* self, int slices, int slice_index)
{
    int hgt = self->height / slices;
    int rem = self->height % slices;
    if (rem == 0)
    {
        self->start_y = (slice_index + 0) * hgt;
        self->end_y   = self->start_y + hgt - 1;
    }
    else
    {
        if (slice_index < rem)
        {
            self->start_y = (slice_index + 0) * (hgt + 1);
            self->end_y   = self->start_y + hgt;
        }
        else
        {
            self->start_y =
                              (rem + 0) * (hgt + 1) +
                (slice_index - rem + 0) * (hgt + 0);
            self->end_y   = self->start_y + hgt - 1;
        }
    }
}

void hp_slice_heat(Hotplate* self)
{
    float new_heat_value = 0.0;
    int x, y;
    int start_y = self->start_y;
    int end_y = self->end_y;
    
    if (start_y <= 0)
        start_y = 1;
    if (end_y   >= self->height - 1)
        end_y = self->height - 1;
    
    printf("[%d] heat: %d to %d\n", self->iproc, start_y, end_y);
    for (y = start_y; y < end_y; y++)
    {
        for (x = 1; x < self->width - 1; x++)
        {
            new_heat_value = hp_get_total_heat_unsafe(self, x, y);
            hp_set_unsafe(self, x, y, new_heat_value);
        }
    }
}

float* hp_slice_start_row(Hotplate* self)
{
    return self->src_matrix + (self->start_y * self->width);
}

float* hp_slice_end_row(Hotplate* self)
{
    return self->src_matrix + (self->end_y * self->width);
}

void hp_swap(Hotplate* self)
{
    if (self->src_matrix == self->matrix_data1)
    {
        self->src_matrix = self->matrix_data2;
        self->dst_matrix = self->matrix_data1;
    }
    else
    {
        self->src_matrix = self->matrix_data1;
        self->dst_matrix = self->matrix_data2;
    }
}

/* Fill all parts of the dst_matrix (within the slice range) with a specific value */
void hp_fill(Hotplate* self, float value)
{
    int x, y;
    for (y = self->start_y; y <= self->end_y; y++)
    {
        for (x = 0; x < self->width; x++)
        {
            hp_set_unsafe(self, x, y, value);
        }
    }
}

/* Copy the dst_matrix to the src_matrix */
void hp_copy_to_source(Hotplate* self)
{
    int bytes = sizeof(float) * self->width * self->height;
    memcpy(self->src_matrix, self->dst_matrix, bytes);
}

/* Determine if the source and destination matrices are close enough to be considered "steady state" */
int hp_is_steady_state(Hotplate* self)
{
    float avg_nearby = 0.0f;
    int x, y;
    int start_y = self->start_y;
    int end_y = self->end_y;
    
    if (self->steady_state) return TRUE;
    
    if (start_y <= 0)
        start_y = 1;
    if (end_y   >= self->height - 1)
        end_y = self->height - 1;

    // int steady_state = TRUE;
    for (y = start_y; y < end_y; y++)
    {
        for (x = 1; x < self->width - 1; x++)
        {
            avg_nearby = hp_get_neighbors_sum_unsafe(self, x, y) / 4.0f;
            if (
                // If the current temp is greater than the average temp by a certain threshold, then
                // we haven't reached a steady state yet...
                fabs(hp_get_unsafe(self, x, y) - avg_nearby) >= 0.1 &&
                // Make sure this is not a special hostpot
                !hp_is_hotspot(self, x, y))
            {
                return FALSE;
            }
        }
    }
    /* Store result so that we don't have to recalculate every time after reaching steady state */
    self->steady_state = TRUE;
    return TRUE;
}

/* Count the # of cells with a value greater than _value_ */
int hp_cells_greater_than(Hotplate* self, float value)
{
    int count = 0;
    int x, y;
    for (y = 0; y < self->height; y++)
    {
        for (x = 0; x < self->width; x++)
        {
            if (hp_get_unsafe(self, x, y) > value) count++;
        }
    }
    return count;
}

/* Print the hotplate state as ASCII values to the terminal */
void hp_dump(Hotplate* self, int characters, int max_x, int max_y)
{
    int x, y;
    char intensity[] = {'.', '=', 'x', '%', 'M'};
    if ( max_x == 0 ) max_x = self->width;
    if ( max_y == 0 ) max_y = self->height;
    
    // printf("Dimensions: %d x %d\n", self->width, self->height);
    for (y = 0; y < max_y; y++)
    {
        for (x = 0; x < max_x; x++)
        {
            if (characters == TRUE)
            {
                int value = hp_get_unsafe(self, x, y) / 20;
                if (value > 4) value = 4;
                printf("%c", intensity[value]);
            }
            else
            {
                float value = hp_get_unsafe(self, x, y);
                printf("[%.01f]", value);
            }
        }
        printf("\n");
    }
}

/* Etch the specific hot spots from the CS 484 Hotplate lab */
void hp_etch_hotspots(Hotplate* self)
{
    hp_hline(self, 400, 0, 330, 100.0);
    hp_set(self, 500, 200, 100.0);
}
