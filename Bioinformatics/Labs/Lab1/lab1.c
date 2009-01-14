#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

/* ===== Contig Class Declaration ===== */

typedef struct CONTIG_STRUCT
{
	int length;
	char* sequence;
} Contig;

Contig* ctig_initialize(char* s);
void ctig_destroy(Contig* self);
void ctig_set_sequence(Contig* self, char* s);

/* ===== NwMatrix Class Declaration ===== */

typedef struct NW_MATRIX
{
	/* This is the N x M matrix for the Needleman-Wunsch algorithm   */
	int width, height;
	int* matrix;
} NwMatrix;

const int NW_MATRIX_DEFAULT_WIDTH  = 5000;
const int NW_MATRIX_DEFAULT_HEIGHT = 5000;

NwMatrix* nw_initialize();
void nw_destroy(NwMatrix* self);
static inline int nw_get_unsafe(NwMatrix* self, int x, int y);
static inline void nw_set_unsafe(NwMatrix* self, int x, int y, int value);
void nw_grow_matrix(NwMatrix* self, int w, int h);
int nw_compute_score(NwMatrix* self, Contig* c1, Contig* c2);

/* * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                   Main Program                    */
/* * * * * * * * * * * * * * * * * * * * * * * * * * */

int main(int argc, char* argv[])
{
	// if (argc < 3)
	// {
	// 	printf("Expecting 2 or more arguments.\n");
	// 	return -1;
	// }
	
	// Main call to algorithm
	{
		Contig* c1;
		Contig* c2;
		NwMatrix *m;
		int score;
		
		c1 = ctig_initialize("agct");
		c2 = ctig_initialize("agct");

		m = nw_initialize();
		score = nw_compute_score(m, c1, c2);
		
		printf("Score: %d\n", score);
		
		nw_destroy(m);
		
		ctig_destroy(c2);
		ctig_destroy(c1);
	}
}


/* ===== Contig Class Definitions ===== */
Contig* ctig_initialize(char* s)
{
	Contig* self = malloc(sizeof(Contig));
	assert(self != NULL);
	ctig_set_sequence(self, s);
	
	return self;
}

void ctig_destroy(Contig* self)
{
	assert(self != NULL);
	assert(self->sequence != NULL);
	free(self->sequence);
	free(self);
}

void ctig_set_sequence(Contig* self, char* s)
{
	assert(s != NULL);
	self->length = strlen(s);
	
	if (self->sequence == NULL)
		free(self->sequence);
	
	self->sequence = malloc(self->length);
	
	memcpy(self->sequence, s, self->length);
}

/* ===== NwMatrix Class Definitions ===== */

NwMatrix* nw_initialize()
{
	NwMatrix* self = malloc(sizeof(NwMatrix));
	assert(self != NULL);
	
	self->width = NW_MATRIX_DEFAULT_WIDTH;
	self->height = NW_MATRIX_DEFAULT_HEIGHT;
	
	/* Allocate Matrix */
	self->matrix = malloc(sizeof(int) * self->width * self->height);
	assert(self->matrix != NULL);
	
	return self;
}

void nw_destroy(NwMatrix* self)
{
	assert(self != NULL);
	assert(self->matrix != NULL);
	free(self->matrix);
	free(self);
}

static inline int nw_get_unsafe(NwMatrix* self, int x, int y)
{
	return self->matrix[y * self->width + x];
}

static inline void nw_set_unsafe(NwMatrix* self, int x, int y, int value)
{
	self->matrix[y * self->width + x] = value;
}

/* Increase the size of the matrix, if necessary */
void nw_grow_matrix(NwMatrix* self, int w, int h)
{
	int cursize = self->width * self->height;
	int newsize = w * h;
	if (cursize < newsize)
	{
		assert(self->matrix != NULL);
		free(self->matrix);
		self->width = w;
		self->height = h;
		/* Allocate Matrix */
		self->matrix = malloc(sizeof(int) * self->width * self->height);
		assert(self->matrix != NULL);
	}
}

int nw_compute_score(NwMatrix* self, Contig* c1, Contig* c2)
{
	assert(c1 != NULL);
	assert(c1->length > 0);
	assert(c2 != NULL);
	assert(c2->length > 0);
	
	/* Algorithm */
	{
		int s_match = 1, s_differ = 0, s_indel = -1;
		int i, j;
		nw_grow_matrix(self, c1->length + 1, c2->length + 1);
		
		#pragma omp sections private(i, j)
		{
			#pragma omp section
			for (i = 0; i < c1->length + 1; i++)
				nw_set_unsafe(self, i, 0, i * s_indel);
			
			#pragma omp section
			for (j = 0; j < c2->length + 1; j++)
				nw_set_unsafe(self, 0, j, j * s_indel);
		}
		
		for (j = 1; j < c2->length + 1; j++)
		{
			for (i = 1; i < c1->length + 1; i++)
			{
				int a = nw_get_unsafe(self, i - 1, j - 1) + (c1->sequence[i] == c2->sequence[j] ? s_match : s_differ);
				int b = nw_get_unsafe(self, i - 1, j) + s_indel;
				int c = nw_get_unsafe(self, i, j - 1) + s_indel;
				int max = (a >= b ? (a >= c ? a : c) : b);
				nw_set_unsafe(self, i, j, max);
			}
		}
		return nw_get_unsafe(self, c1->length, c2->length);
	}
}