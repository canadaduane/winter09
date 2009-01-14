#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
// #include <omp.h>

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
int nw_compute_score(NwMatrix* self, Contig* c1, Contig* c2);

/* ***** Main Program ***** */
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
		
		c1 = ctig_initialize("one");
		c2 = ctig_initialize("own");

		m = nw_initialize();
		score = nw_compute_score(m, c1, c2);
		
		printf("Score: %d\n", score);
		
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

int nw_compute_score(NwMatrix* self, Contig* c1, Contig* c2)
{
	assert(c1 != NULL);
	assert(c2 != NULL);

	return 0;
}