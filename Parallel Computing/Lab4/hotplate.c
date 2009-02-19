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

/* Fill all parts of the dst_matrix with a specific value */
void hp_fill(Hotplate* self, float value)
{
    int square = self->width * self->height;
    int i;
    for (i = 0; i < square; i++)
        self->dst_matrix[i] = value;
}

/* Copy the dst_matrix to the src_matrix */
void hp_copy_to_source(Hotplate* self)
{
    int bytes = sizeof(float) * self->width * self->height;
    memcpy(self->src_matrix, self->dst_matrix, bytes);
}

static inline float hp_get(Hotplate* self, int x, int y)
{
    assert(x >= 0 && x < self->width);
    assert(y >= 0 && y < self->height);
    return hp_get_unsafe(self, x, y);
}

static inline float hp_get_unsafe(Hotplate* self, int x, int y)
{
    return self->src_matrix[y * self->width + x];
}

static inline void hp_set(Hotplate* self, int x, int y, float value)
{
    assert(x >= 0 && x < self->width);
    assert(y >= 0 && y < self->height);
    hp_set_unsafe(self, x, y, value);
}

static inline void hp_set_unsafe(Hotplate* self, int x, int y, float value)
{
    self->dst_matrix[y * self->width + x] = value;
}

static inline void hp_hline(Hotplate* self, int y, int x1, int x2, float value)
{
    int x;
    assert(y >= 0 && y < self->height);
    assert(x1 >= 0 && x1 < self->width);
    assert(x2 >= 0 && x2 < self->width);
    for (x = x1; x <= x2; x++)
        hp_set_unsafe(self, x, y, value);
}

static inline void hp_vline(Hotplate* self, int x, int y1, int y2, float value)
{
    int y;
    assert(x >= 0 && x < self->width);
    assert(y1 >= 0 && y1 < self->height);
    assert(y2 >= 0 && y2 < self->height);
    for (y = y1; y <= y2; y++)
        hp_set_unsafe(self, x, y, value);
}

/* Get the heat sum of a cell's four neighbors */
static inline float hp_get_neighbors_sum_unsafe(Hotplate* self, int x, int y)
{
    return
        hp_get_unsafe(self, x, y - 1) +
        hp_get_unsafe(self, x, y + 1) +
        hp_get_unsafe(self, x - 1, y) +
        hp_get_unsafe(self, x + 1, y);
}

/* Get the heat at the given cell, with the heat values of its 4 neighbors in consideration */
static inline float hp_get_total_heat_unsafe(Hotplate* self, int x, int y)
{
    float neighbors = hp_get_neighbors_sum_unsafe(self, x, y);
    return (neighbors + 4.0f * hp_get_unsafe(self, x, y)) / 8.0f;
}

void hp_calculate_heat_transfer_single(Hotplate* self)
{
    float new_heat_value = 0.0;
    int x, y;
    // printf("Process calculating y: %d to %d (lines = %d).\n", 1, self->height - 1, self->height - 2);
    for (y = 1; y < self->height - 1; y++)
    {
        for (x = 1; x < self->width - 1; x++)
        {
            new_heat_value = hp_get_total_heat_unsafe(self, x, y);
            hp_set_unsafe(self, x, y, new_heat_value);
        }
    }
}

void hp_calculate_heat_transfer_parallel(Hotplate* self)
{
    int i;
    HotplateThread threads[NUM_THREADS];
    for (i = 0; i < NUM_THREADS; i++)
    {
        hpt_initialize_static( threads + i, i, self, hp_calculate_heat_transfer_parallel_thread );
    }
    
    for (i = 0; i < NUM_THREADS; i++)
    {
        hpt_join( threads + i );
        hpt_destroy_static( threads + i );
    }
}

void* hp_calculate_heat_transfer_parallel_thread( void* v )
{
    HotplateThread* thread = (HotplateThread*)v;
    int lines = thread->hotplate->height / NUM_THREADS;
    int x, y;
    int x_min = 1, x_max = thread->hotplate->width - 1;
    int y_min = 1 + (lines * thread->id);
    int y_max = y_min + lines;
    
    /* Make sure the last thread takes care of the remainder of the integer division of the hotplate */
    if (thread->id == NUM_THREADS - 1) y_max = thread->hotplate->height - 1;
    
    // printf("Thread %d calculating y: %d to %d (lines = %d).\n", thread->id, y_min, y_max, lines);
    
    float new_heat_value;
    for (y = y_min; y < y_max; y++)
    {
        for (x = x_min; x < x_max; x++)
        {
            new_heat_value = hp_get_total_heat_unsafe(thread->hotplate, x, y);
            hp_set_unsafe(thread->hotplate, x, y, new_heat_value);
        }
    }
}

/* Transfer heat according to algorithm */
void hp_calculate_heat_transfer(Hotplate* self, int iter)
{
    // printf("Start iteration %d...\n", iter);
    // hp_calculate_heat_transfer_single(self);
    hp_calculate_heat_transfer_parallel(self);
    // printf("end iteration %d.\n", iter);
}