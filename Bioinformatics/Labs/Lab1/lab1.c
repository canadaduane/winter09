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

Contig* ctig_intitialize(char* s);
void ctig_destroy(Contig* self);
void ctig_set_sequence(char* s);

/* ===== NwMatrix Class Declaration ===== */

typedef struct NW_MATRIX
{
	int width;
	int height;
	int* matrix;
} NwMatrix;

NwMatrix* nw_initialize(int w, int h);
void nw_destroy(NwMatrix* self);

/* ***** Main Program ***** */
int main(int argc, char** argv)
{
	
}


/* ===== Contig Class Definitions ===== */
Contig* ctig_intitialize(char* s)
{
	Contig* self = malloc(sizeof(Contig));
	assert(self != NULL);
	ctig_set_sequence(s);
	
	return self;
}

void ctig_destroy(Contig* self)
{
	assert(self != NULL);
	assert(self->sequence != NULL);
	free(self->sequence);
	free(self);
}

void ctig_set_sequence(char* s)
{
	assert(s != NULL);
	self->length = strlen(s);
	
	if (self->sequence == NULL)
		free(self->sequence);
	
	self->sequence = malloc(self->length);
	
	memcpy(self->sequence, s, self->length);
}

/* ===== NwMatrix Class Definitions ===== */

NwMatrix* nw_initialize(int w, int h)
{
	NwMatrix* self = malloc(sizeof(NwMatrix));
	assert(self != NULL);
	self->width = w;
	self->height = h;
	
	/* Allocate Matrix */
	self->matrix = malloc(sizeof(int) * w * h);
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
