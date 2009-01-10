#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define FALSE 0
#define TRUE 1

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

double when();
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
void hp_calculate_heat_transfer(Hotplate* self);
int hp_is_steady_state(Hotplate* self);
int hp_cells_greater_than(Hotplate* self, float value);
void hp_dump(Hotplate* self);

void hp_etch_hotspots(Hotplate* self);
static inline int hp_is_hotspot(Hotplate* self, int x, int y);

int main(int argc, char **argv)
{
	double start = when(), finish = 0.0;
	int size = 768, i;
	int gt_count = 0;

	/* Initialize the hotplate with specific pixel temperatures */
	Hotplate* hp = hp_initialize(size, size);
	hp_fill(hp, 50.0);
	hp_hline(hp, 0,      0, size-1,   0.0);
	hp_hline(hp, size-1, 0, size-1, 100.0);
	hp_vline(hp, 0,      0, size-1,   0.0);
	hp_vline(hp, size-1, 0, size-1,   0.0);
	hp_etch_hotspots(hp);
	
	hp_copy_to_source(hp);
	
	for(i = 0; i < 500; i++) // should be 359
	{
		hp_calculate_heat_transfer(hp);
		hp_etch_hotspots(hp);
		if (hp_is_steady_state(hp)) break;
		hp_swap(hp);
		// printf("%d ", i); fflush(stdout);
	}
	
	// hp_dump(hp);
	
	finish = when();
	gt_count = hp_cells_greater_than(hp, 50.0f);
	hp_destroy(hp);
	printf("Completed %d iterations in %f seconds.\n", i, finish - start);
	printf("%d cells had a value greater than 50.0.\n", gt_count);
}

/* Return the current time in seconds, using a double precision number. */
double when()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

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

/* Transfer heat according to algorithm */
void hp_calculate_heat_transfer(Hotplate* self)
{
	float new_heat_value = 0.0;
	int x, y;
	for (y = 1; y < self->height - 1; y++)
	{
		for (x = 1; x < self->width - 1; x++)
		{
			new_heat_value = hp_get_total_heat_unsafe(self, x, y);
			hp_set_unsafe(self, x, y, new_heat_value);
		}
	}
}

/* Determine if the source and destination matrices are close enough to be considered "steady state" */
int hp_is_steady_state(Hotplate* self)
{
	float avg_nearby = 0.0f;
	int x, y;
	for (y = 1; y < self->height - 1; y++)
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
				// printf("Steady state failed at: %d, %d", x, y);
				return FALSE;
			}
		}
	}
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
void hp_dump(Hotplate* self)
{
	int x, y;
	char intensity[] = {'.', '=', 'x', '%', 'M'};
	printf("Dimensions: %d x %d\n", self->width, self->height);
	for (y = 0; y < self->height; y++)
	{
		for (x = 0; x < self->width; x++)
		{
			int value = hp_get_unsafe(self, x, y) / 20;
			if (value > 4) value = 4;
			printf("%c", intensity[value]);
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

static inline int hp_is_hotspot(Hotplate* self, int x, int y)
{
	if (x == 500 && y == 200)
		return TRUE;
	if (y == 400 && (x >= 0 && x <= 330))
		return TRUE;
	return FALSE;
}