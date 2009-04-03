#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>

#include "misc.h"

#define BLOCK_SIZE 16

typedef struct GRID
{
    int w; // Width of grid
    int h; // Height of grid
    int* box;
    int success;
} Grid;

Grid* grid_new();
Grid* grid_init( Grid* g, int w, int h );
Grid* grid_init_file( Grid* g, char* filename );
Grid* grid_free( Grid* g );
Grid* grid_clear( Grid* g );
Grid* grid_copy( Grid* a, Grid* b );
Grid* grid_copy_to_device( Grid* g );
Grid* grid_copy_from_device( Grid* g );
Grid* grid_set_seq_row( Grid* g, char* seq, int w );
Grid* grid_set_seq_col( Grid* g, char* seq, int h );
Grid* grid_show( Grid* g );
Grid* grid_alignment( Grid* g );

Grid* grid_new()
{
    Grid* g = (Grid*)malloc( sizeof( Grid ) );
    
    g->w = 0;
    g->h = 0;
    g->box = NULL;
    g->success = true;
    
    return g;
}

Grid* grid_init( Grid* g, int w, int h )
{
    assert( g->success == true );
    
    // Free memory if necessary
    grid_free( g );
    
    g->w = w + 2;
    g->h = h + 2;
    g->box = (int *)malloc( g->w * g->h * sizeof( int ) );
    
    return g;
}

Grid* grid_init_file( Grid* g, char* filename )
{
    assert( g->success == true );
    assert( filename != NULL );
    assert( strlen(filename) > 0 );
    
    FILE* input = fopen( filename, "r" );
    if( input != NULL )
    {
        int i = -1;
        char* line;
        char* seq[2];
        int seq_size[2] = {0, 0};
        
        seq[0] = (char *)malloc( 1 );
        seq[1] = (char *)malloc( 1 );
        while( (line = readline( input )) )
        {
            int length = strlen( line );
            if( length == 0 )
            {
                // Skip blank lines
            }
            else if( line[0] == '>' )
            {
                // Begin sequence after this line
                i++;
            }
            else if( i >= 0 && i <= 1 )
            {
                // Grab contents and add it to our seq
                int j;
                for( j = 0; j < length; j++ )
                {
                    if( line[j] != ' ' && line[j] != '\t' && line[j] != '\n' )
                    {
                        seq[i][ seq_size[i]++ ] = line[j];
                        seq[i] = (char*)realloc( seq[i], seq_size[i] + 1 );
                    }
                }
            }
            free( line );
        }
        // Be nice and add a trailing null char so sequences can be printed
        seq[0][ seq_size[0] ] = '\0';
        seq[1][ seq_size[1] ] = '\0';
        
        // Grid sequences point to newly loaded sequences
        grid_init( g, seq_size[0], seq_size[1] );
        grid_set_seq_row( g, seq[0], seq_size[0] );
        grid_set_seq_col( g, seq[1], seq_size[1] );
        
        fclose( input );
    }
    else
    {
        g->success = false;
    }
    
    return g;
}

Grid* grid_free( Grid* g )
{
    assert( g->success == true );
    
    if( g->box != NULL )
    {
        free( g->box );
        g->box = NULL;
    }
    
    return g;
}

Grid* grid_clear( Grid* g )
{
    assert( g->success == true );
    assert( g->box != NULL );
    
    int size = g->w * g->h * sizeof(int);
    memset( g->box, 0, size );
    
    return g;
}

// Copy grid 'a' to grid 'b'
Grid* grid_copy( Grid* a, Grid* b )
{
    assert( a->success == true && b->success == true );
    assert( a->w == b->w && a->h == b->h );
    assert( b->box != NULL );
    
    int size = a->w * a->h * sizeof(int);
    memcpy( b->box, a->box, size );
    
    return a;
}

// Copies a Grid object to the device and returns a DEVICE pointer to the copy
Grid* grid_copy_to_device( Grid* g )
{
    assert( g->success == true );
    assert( g->box != NULL );
    
    // Create a temp Grid object where we will setup a Device pointer to the box data
    Grid tmp;
    tmp.w = g->w;
    tmp.h = g->h;
    tmp.success = true;
    
    // Allocate room for the object AND the object's box data
    Grid* grid_d;
    int size = sizeof( int ) * tmp.w * tmp.h;
    cudaMalloc( (void**)& grid_d, sizeof( Grid ) );
    cudaMalloc( (void**)& tmp.box, size );
    
    // Copy the object and the box data to the device
    cudaMemcpy( grid_d, &tmp, sizeof( Grid ), cudaMemcpyHostToDevice);
    cudaMemcpy( tmp.box, g->box, size, cudaMemcpyHostToDevice);
    
    // Return the DEVICE pointer
    return grid_d;
}

Grid* grid_copy_from_device( Grid* g )
{
    assert( g->success == true );
    assert( g->box != NULL );
    
    // Copy the object from the device
    Grid* grid_h = (Grid*)malloc( sizeof( Grid ) );
    cudaMemcpy( grid_h, g, sizeof( Grid ), cudaMemcpyDeviceToHost);
    
    // Copy the box data from the device
    int size = sizeof( int ) * grid_h->w * grid_h->h;
    int* box = (int*)malloc( size );
    cudaMemcpy( box, grid_h->box, size, cudaMemcpyDeviceToHost);
    grid_h->box = box;
    
    // Return the HOST pointer
    return grid_h;
}

Grid* grid_set_seq_row( Grid* g, char* seq, int w )
{
    assert( g->w >= w );
    
    for( int i = 2; i < g->w; i++ )
        g->box[i] = seq[i-2];
    
    return g;
}

Grid* grid_set_seq_col( Grid* g, char* seq, int h )
{
    assert( g->h >= h );
    
    int i = 2 * g->w;
    int j;
    for( j = 0; j < h; j++, i+=g->w )
        g->box[i] = seq[j];
    
    return g;
}

// Show small grids as text output.  NOTE: Will not work for values > 48
Grid* grid_show( Grid* g )
{
    printf( "Show Grid: %d x %d\n", g->w, g->h );
    int i, j;
    for( i = 0; i < g->h; i++ )
    {
        for( j = 0; j < g->w; j++ )
        {
            int c = g->box[ i * g->w + j ];
            if( i == 0 || j == 0 )
            {
                if( c == 0) printf( "    " );
                else printf("  %c ", c);
            }
            else printf("%3d ", c);
        }
        printf("\n");
    }
    return g;
}

Grid* grid_alignment( Grid* g )
{
    // Costs
    int indel = 3;
    int subst = 1;
    int match = 0;
    
    // Initialize corner
    g->box[1 * g->w + 1] = 0;
    
    // Prepare first horizontal line
    for( int i = 2; i < g->w; i++ )
        g->box[1 * g->w + i] = (i - 1) * indel; 
    
    // Prepare first vertical line
    for( int i = 2; i < g->h; i++ )
        g->box[i * g->w + 1] = (i - 1) * indel; 
    
    // Setup for diagonal alignment solution
    int min = min2(g->w, g->h) - 2;
    int max = max2(g->w, g->h) - 2;
    int col = 1, row = g->w;
    
    // Increase diagonally
    for( int k = 0; k < min; k++ )
    {
        for( int i = 0; i <= k; i++ )
        {
            int x = (2 + k - i);
            int y = (2 + i) * row;
            int diag = g->box[(y - row) + (x - col)];
            int vert = g->box[(y - row) + (x)];
            int horz = g->box[(y) + (x - col)];
            int c1 = diag + (g->box[x] == g->box[y] ? match : subst);
            int c2 = vert + indel;
            int c3 = horz + indel;
            g->box[x + y] = min3(c1, c2, c3);
        }
    }
    
    // Translate (skip this for now, assume symmetry)
    
    // Decrease diagonally
    for( int k = 0; k < max - 1; k++ )
    {
        for( int i = 0; i < (max - k - 1); i++ )
        {
            int x = (1 + max - i);
            int y = (3 + k + i) * row;
            int diag = g->box[(y - row) + (x - col)];
            int vert = g->box[(y - row) + (x)];
            int horz = g->box[(y) + (x - col)];
            int c1 = diag + (g->box[x] == g->box[y] ? match : subst);
            int c2 = vert + indel;
            int c3 = horz + indel;
            g->box[x + y] = min3(c1, c2, c3);
        }
    }
    
    return g;
}

__global__ void cuda_grid_alignment( Grid* g )
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    printf("idx: %d\n", idx);
    printf("cuda dim: %d x %d\n", g->w, g->h);
    g->box[36] = 111;
}

int main( int argc, char** argv )
{
    char* filename = shell_arg_string( argc, argv, "-f", "small.fasta" );
    
    Grid* grid_h = grid_new();
    
    // grid_init( grid_h, 4, 4 );
    grid_init_file( grid_h, filename );
    printf("Grid Size: %d x %d\n", grid_h->w, grid_h->h);

    Grid* grid_d = grid_copy_to_device( grid_h );
    
    cuda_grid_alignment<<< 1, 1, 1 >>>( grid_d );
    
    Grid* grid_result = grid_copy_from_device( grid_d );
    
    // grid_alignment( grid_h );
    grid_show( grid_result );
}
