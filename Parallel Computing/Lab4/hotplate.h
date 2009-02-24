#include <assert.h>

#ifndef HOTPLATE_H
#define HOTPLATE_H

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

typedef struct HOT_PLATE_STRUCT
{
    /* Dimensions of hotplate */
    int width;
    int height;
    /* Each process needs to know the start and end of its "slice" */
    int start_y;
    int end_y;
    /* Pointers to matrix data */
    float* src_matrix;
    float* dst_matrix;
    /* Actual matrix data */
    float* matrix_data1;
    float* matrix_data2;
} Hotplate;

Hotplate* hp_initialize(int w, int h);
void hp_destroy(Hotplate* self);
void hp_slice(Hotplate* self, int slices, int slice_index);
void hp_slice_heat(Hotplate* self);
float* hp_slice_start_row(Hotplate* self);
float* hp_slice_end_row(Hotplate* self);
void hp_swap(Hotplate* self);
void hp_fill(Hotplate* self, float value);
void hp_copy_to_source(Hotplate* self);
int hp_is_steady_state(Hotplate* self);
int hp_cells_greater_than(Hotplate* self, float value);
void hp_dump(Hotplate* self, int characters, int max_x, int max_y);
void hp_etch_hotspots(Hotplate* self);

static inline void hp_set(Hotplate* self, int x, int y, float value);
static inline void hp_set_unsafe(Hotplate* self, int x, int y, float value);
static inline float hp_get(Hotplate* self, int x, int y);
static inline float hp_get_unsafe(Hotplate* self, int x, int y);
static inline void hp_hline(Hotplate* self, int y, int x1, int x2, float value);
static inline void hp_vline(Hotplate* self, int x, int y1, int y2, float value);
static inline float hp_get_neighbors_sum_unsafe(Hotplate* self, int x, int y);
static inline float hp_get_total_heat_unsafe(Hotplate* self, int x, int y);
static inline int hp_is_hotspot(Hotplate* self, int x, int y);


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

static inline int hp_is_hotspot(Hotplate* self, int x, int y)
{
	if (x == 500 && y == 200)
		return TRUE;
	if (y == 400 && (x >= 0 && x <= 330))
		return TRUE;
	return FALSE;
}

#endif
