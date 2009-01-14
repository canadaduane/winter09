#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "readline.h"

const int TRUE = 1;
const int FALSE = 0;

/* ===== Contig Class Declaration ===== */

typedef struct CONTIG_STRUCT
{
	int length;
	char* sequence;
} Contig;

Contig* ctig_initialize(void);
Contig* ctig_initialize_with_string(char* s);
void ctig_destroy(Contig* self);
void ctig_set_sequence(Contig* self, char* s);

/* ===== ContigList Class Declaration ===== */

typedef struct CONTIG_LIST_STRUCT
{
	int length;
	Contig** list;
} ContigList;

ContigList* ctig_list_initialize(void);
void ctig_list_destroy(ContigList* self);
void ctig_list_add(ContigList* self, Contig* ctig);
void ctig_list_grow(ContigList* self, int len);
void ctig_list_load_fasta(ContigList* self, char* filename);
void ctig_list_print(ContigList* self);

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
void nw_grow(NwMatrix* self, int w, int h);
int nw_compute_score(NwMatrix* self, Contig* c1, Contig* c2);

/* * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                   Main Program                    */
/* * * * * * * * * * * * * * * * * * * * * * * * * * */

int main(int argc, char* argv[])
{
	// if (argc < 2)
	// {
	// 	printf("Usage: seq [dna-seq-fasta.out]\n");
	// 	return -1;
	// }
	
	{
		ContigList *list = ctig_list_initialize();
		ctig_list_load_fasta(list, "sequences-1.fasta.out");
		fflush(stdout);
		ctig_list_print(list);
	}
	exit(0);
	
	// Main call to algorithm
	{
		Contig* c1;
		Contig* c2;
		NwMatrix *m;
		int score;
		
		c1 = ctig_initialize_with_string("agct");
		c2 = ctig_initialize_with_string("agct");

		m = nw_initialize();
		score = nw_compute_score(m, c1, c2);
		
		printf("Score: %d\n", score);
		
		nw_destroy(m);
		
		ctig_destroy(c2);
		ctig_destroy(c1);
	}
}


/* ===== Contig Class Definitions ===== */

Contig* ctig_initialize()
{
	Contig* self = calloc(1, sizeof(Contig));
	assert(self != NULL);
	
	self->length = 0;
	self->sequence = calloc(0, sizeof(char));
	
	return self;
}

Contig* ctig_initialize_with_string(char* s)
{
	
	Contig* self = ctig_initialize();
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

/* Sets the object length and copies the full string (including the \0) */
void ctig_set_sequence(Contig* self, char* s)
{
	assert(s != NULL);
	self->length = strlen(s);
	
	if (self->sequence == NULL)
		free(self->sequence);
	
	self->sequence = malloc(self->length + 1);
	
	memcpy(self->sequence, s, self->length + 1);
}

/* ===== ContigList Class Definitions ===== */

ContigList* ctig_list_initialize(void)
{
	ContigList* self = calloc(1, sizeof(ContigList));
	assert(self != NULL);
	
	self->length = 0;

	/* Allocate Dummy Pointer List */
	self->list = calloc(0, sizeof(Contig*));
	assert(self->list != NULL);
	
	return self;
}

void ctig_list_grow(ContigList* self, int new_len)
{
	assert(self->list != NULL);
	if (new_len > self->length)
	{
		self->length = new_len;
		self->list = realloc(self->list, sizeof(Contig*) * self->length);
	}
}

void ctig_list_destroy(ContigList* self)
{
	assert(self != NULL);
	assert(self->list != NULL);
	free(self->list);
	free(self);
}

void ctig_list_add(ContigList* self, Contig* ctig)
{
	ctig_list_grow(self, self->length + 1);
	self->list[self->length - 1] = ctig;
}

void ctig_list_load_fasta(ContigList* self, char* filename)
{
	FILE* in = fopen(filename, "r");
	char* line = NULL;
	int seq_line_next = FALSE;
	
	while((line = readline(in)) && !feof(in))
	{
		if (seq_line_next == TRUE)
		{
			// printf("Line: %s\n", line);
			Contig* new_contig = ctig_initialize();
			new_contig->sequence = line;
			new_contig->length = strlen(line);
			ctig_list_add(self, new_contig);
			seq_line_next = FALSE;
		}
		else if (line[0] == '>')
		{
			seq_line_next = TRUE;
		}
		// printf("Readline: %s\n", line);
	}
	
	fclose(in);
}

void ctig_list_print(ContigList* self)
{
	int i;
	for (i = 0; i < self->length; i++)
	{
		assert(i < self->length);
		assert(self->list[i] != NULL);
		printf("%d: %s\n", i, self->list[i]->sequence);
	}
}

/* ===== NwMatrix Class Definitions ===== */

NwMatrix* nw_initialize()
{
	NwMatrix* self = calloc(1, sizeof(NwMatrix));
	assert(self != NULL);
	
	self->width = NW_MATRIX_DEFAULT_WIDTH;
	self->height = NW_MATRIX_DEFAULT_HEIGHT;
	
	/* Allocate Matrix */
	self->matrix = calloc(self->width * self->height, sizeof(int));
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
void nw_grow(NwMatrix* self, int w, int h)
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
		self->matrix = calloc(self->width * self->height, sizeof(int));
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
		nw_grow(self, c1->length + 1, c2->length + 1);
		
		#pragma omp sections private(i, j)
		{
			#pragma omp section
			for (i = 0; i < c1->length + 1; i++)
				nw_set_unsafe(self, i, 0, i * s_indel);
			
			#pragma omp section
			for (j = 0; j < c2->length + 1; j++)
				nw_set_unsafe(self, 0, j, j * s_indel);
		}
		
		#pragma omp parallel for private(i, j)
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