#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include "readline.h"
#include "minmax.h"

const int TRUE = 1;
const int FALSE = 0;

/* ===== Contig Class Declaration ===== */

typedef struct CONTIG_STRUCT
{
	int length;
	char* sequence;
} Contig;

typedef struct CONTIG_MATCH_STRUCT
{
	int i_start, i_end;
	int j_start, j_end;
	int overlap;
	int position;
	Contig* c1;
	Contig* c2;
} ContigMatch;

const int MATCH_THRESHOLD = 5;

Contig* ctig_initialize(void);
Contig* ctig_initialize_with_string(char* s);
void ctig_destroy(Contig* self);
void ctig_set_sequence(Contig* self, char* s);
void ctig_exact_match(ContigMatch* match, Contig* self, Contig* other);
Contig* ctig_exact_merge(ContigMatch* match);

/* ===== ContigList Class Declaration ===== */

typedef struct CONTIG_LIST_STRUCT
{
	int length;
	Contig** list;
} ContigList;

ContigList* ctig_list_initialize(void);
void ctig_list_destroy(ContigList* self);
void ctig_list_add(ContigList* self, Contig* ctig);
Contig* ctig_list_add_sequence(ContigList* self, char* sequence, int length);
void ctig_list_grow(ContigList* self, int len);
void ctig_list_load_fasta(ContigList* self, char* filename);
void ctig_list_print(ContigList* self);
ContigList* ctig_list_cross_match(ContigList* self);

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
		ContigList *result = NULL;
		
		ctig_list_load_fasta(list, "sequences-3.fasta.out");
		ctig_list_print(list);
		
		result = ctig_list_cross_match(list);
		ctig_list_print(result);
	}
	exit(0);
	
	// Main call to algorithm
	{
		ContigMatch* match = calloc(1, sizeof(ContigMatch));
		Contig* c1 = NULL;
		Contig* c2 = NULL;
		Contig* c3 = NULL;
		NwMatrix *m = NULL;
		int overlap = 0;
		
		c1 = ctig_initialize_with_string("ACTGCCCCTTCAA");
		c2 = ctig_initialize_with_string("CCCCTTCAAG");

		// m = nw_initialize();
		
		ctig_exact_match(match, c1, c2);
		printf("Match | overlap: %d, position: %d\n", match->overlap, match->position);
		printf("  c1: %s\n  c2: %s\n", match->c1->sequence, match->c2->sequence);
		printf("  i_start: %d, i_end: %d\n", match->i_start, match->i_end);
		printf("  j_start: %d, j_end: %d\n", match->j_start, match->j_end);
		printf("\n");
		
		if (match->overlap > 0)
		{
			c3 = ctig_exact_merge(match);
			printf("Merge: %s\n", c3->sequence);
		}
		else
		{
			printf("Nothing to merge\n");
		}
		
		// nw_destroy(m);
		
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

void ctig_exact_match(ContigMatch* match, Contig* self, Contig* other)
{
	int h, i, j;

	match->c1 = self;
	match->c2 = other;
	match->overlap = 0;
	match->position = -1;
	
	for (h = 0; h < self->length + other->length; h++)
	{
		int match_overlap = 0;
		int i_start = max(0, h - other->length);
		int i_end   = min3(h, self->length, self->length + other->length - h);
		int j_start = max(0, other->length - h);
		int j_end   = min(other->length, self->length + other->length - h);
		// printf("h: %d, i_start: %d, i_end: %d, j_start: %d, j_end: %d\n", h, i_start, i_end, j_start, j_end);
		for (i = i_start; i < i_end; i++)
		{
			for (j = j_start; j < j_end; j++)
			{
				// printf("h: %d, i: %d, j: %d | %c =?= %c\n", h, i, j, self->sequence[i], other->sequence[j]);
				if (self->sequence[i] == other->sequence[j])
				{
					match_overlap ++;
					i ++;
				}
				else
				{
					match_overlap = 0;
					goto failed_match;
				}
			}
		}
		if (match_overlap > match->overlap)
		{
			match->i_start = i_start;
			match->i_end = i_end;
			match->j_start = j_start;
			match->j_end = j_end;
			match->overlap = match_overlap;
			match->position = h;
		}
		failed_match:;
	}
}

Contig* ctig_exact_merge(ContigMatch* match)
{
	int max_length =
		match->c1->length +
		match->c2->length;
	assert( match->position > 0 && match->position < max_length );
	int actual_length = max_length - match->overlap;
	char* sequence = calloc(actual_length + 1, sizeof(char));
	char* seq1 = NULL;
	char* seq2 = NULL;
	int seq1_len = 0;
	int seq2_len = 0;
	int k = 0;

	if (match->i_start > match->j_start)
	{	// i_start begins the sequence
		seq1 = match->c1->sequence;
		seq1_len = match->i_end;
	}
	else
	{ // j_start begins the sequence
		seq1 = match->c2->sequence;
		seq1_len = match->j_end;
	}
	
	if (match->c1->length - match->i_end > match->c2->length - match->j_end)
	{ // use c1 as second half
		seq2 = match->c1->sequence + match->i_end;
		seq2_len = match->c1->length - match->i_end;
	}
	else
	{ // use c2 as second half
		seq2 = match->c2->sequence + match->j_end;
		seq2_len = match->c2->length - match->j_end;
	}
	
	memcpy(sequence, seq1, seq1_len);
	memcpy(sequence + seq1_len, seq2, seq2_len);
	sequence[seq1_len + seq2_len + 1] = '\0';
	
	Contig* new_contig = ctig_initialize();
	new_contig->sequence = sequence;
	new_contig->length = seq1_len + seq2_len;

	return new_contig;
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
		self->list = realloc(self->list, sizeof(Contig*) * new_len);
	}
	self->length = new_len;
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

Contig* ctig_list_add_sequence(ContigList* self, char* sequence, int length)
{
	Contig* new_contig = ctig_initialize();
	new_contig->sequence = sequence;
	new_contig->length = length;
	ctig_list_add(self, new_contig);
	return new_contig;
}

void ctig_list_load_fasta(ContigList* self, char* filename)
{
	FILE* in = fopen(filename, "r");
	char* line = NULL;
	char* complete_line = calloc(1, sizeof(char));
	int line_length = 0;
	
	int seq_line_next = FALSE;
	int seq_line_done = FALSE;
	
	strcpy(complete_line, "");
	while((line = readline(in)) && !feof(in))
	{
		if (seq_line_next == TRUE)
		{
			// printf("Readline: %s\n", line);
			// printf(" c1: %d, c2: %d\n", line[0], line[1]);
			if (line[0] == '\0') // Blank Line
			{
				seq_line_done = TRUE;
			}
			else
			{
				line_length += strlen(line);
				complete_line = realloc(complete_line, line_length + 1);
				strcat(complete_line, line);
			}
		}
		else if (line[0] == '>') // Start of DNA data section
		{
			seq_line_next = TRUE;
		}
		
		if (seq_line_done == TRUE)
		{
			ctig_list_add_sequence(self, complete_line, line_length);
			seq_line_next = FALSE;
			seq_line_done = FALSE;
			
			/* Allocate the beginning of a new buffer for the complete_line pointer */
			complete_line = calloc(1, sizeof(char));
			line_length = 0;
			strcpy(complete_line, "");
		}
	}

	if (line_length > 0)
		ctig_list_add_sequence(self, complete_line, line_length);
	
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

ContigList* ctig_list_cross_match(ContigList* self)
{
	Contig* merged = NULL;
	ContigList* src = self;
	ContigList* dst = ctig_list_initialize();
	ContigMatch match, best_match;
	int prev_list_length = -1;
	int i = 0, j = 0;
	int* marked = calloc(0, sizeof(int));
	int best_match_index = -1;
	
	while (prev_list_length != src->length)
	{
		prev_list_length = src->length;
		
		/* Reallocate the 'marked' boolean array for the src ContigList */
		free(marked); marked = calloc(src->length, sizeof(int));
		
		for (i = 0; i < src->length; i++)
		{
			best_match.overlap = 0;
			best_match_index = -1;
			if (!marked[i])
			{
				for (j = 0; j < src->length; j++)
				{
					if (!marked[j] && i != j)
					{
						ctig_exact_match( &match, src->list[i], src->list[j] );
						if (match.overlap >= MATCH_THRESHOLD && match.overlap > best_match.overlap)
						{
							best_match = match;
							best_match_index = j;
						}
					}
				}
				if (best_match.overlap > 0 && best_match_index >= 0)
				{
					marked[best_match_index] = TRUE;
					// printf("Merging %d and %d\n", i, best_match_index);
					merged = ctig_exact_merge(&best_match);
					// printf("  %s\n", merged->sequence);
					src->list[i] = merged;
				}
			}
		}
		
		/* Reallocate the 'destination' ContigList */
		dst = ctig_list_initialize();

		/* Move marked Contigs to the dst list, then swap src and dst */
		for (i = 0; i < src->length; i++)
		{
			if (!marked[i])
				ctig_list_add(dst, src->list[i]);
			else
				ctig_destroy(src->list[i]);
		}
		ctig_list_destroy(src);
		src = dst;
	}
	
	return src;
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
