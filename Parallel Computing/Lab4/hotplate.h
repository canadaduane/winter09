#ifndef HOTPLATE_H
#define HOTPLATE_H

typedef struct HOT_PLATE_STRUCT
{
    int width;
    int height;
    /* Pointers to matrix data */
    float* src_matrix;
    float* dst_matrix;
    /* Actual matrix data */
    float* matrix_data1;
    float* matrix_data2;
} Hotplate;

Hotplate* hp_initialize(int w, int h);
void hp_destroy(Hotplate* self);
void hp_swap(Hotplate* self);
void hp_fill(Hotplate* self, float value);
void hp_copy_to_source(Hotplate* self);
static inline void hp_set(Hotplate* self, int x, int y, float value);
static inline void hp_set_unsafe(Hotplate* self, int x, int y, float value);
static inline float hp_get(Hotplate* self, int x, int y);
static inline float hp_get_unsafe(Hotplate* self, int x, int y);
static inline void hp_hline(Hotplate* self, int y, int x1, int x2, float value);
static inline void hp_vline(Hotplate* self, int x, int y1, int y2, float value);
static inline float hp_get_neighbors_sum_unsafe(Hotplate* self, int x, int y);
static inline float hp_get_total_heat_unsafe(Hotplate* self, int x, int y);
void hp_calculate_heat_transfer(Hotplate* self, int iter);
void hp_calculate_heat_transfer_single(Hotplate* self);
void hp_calculate_heat_transfer_parallel(Hotplate* self);
int hp_is_steady_state(Hotplate* self);
int hp_cells_greater_than(Hotplate* self, float value);
void hp_dump(Hotplate* self, int characters);
void hp_etch_hotspots(Hotplate* self);
static inline int hp_is_hotspot(Hotplate* self, int x, int y);

#endif