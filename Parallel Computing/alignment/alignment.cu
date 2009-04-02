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
            // printf("%d: %s\n", i, line);
            free( line );
        }
        // Be nice and add a trailing null char so sequences can be printed
        seq[0][ seq_size[0] ] = '\0';
        seq[1][ seq_size[1] ] = '\0';
        
        // Grid sequences point to newly loaded sequences
        grid_init( g, seq_size[0] + 2, seq_size[1] + 2 );
        grid_set_seq_row( g, seq[0], seq_size[0] );
        grid_set_seq_col( g, seq[1], seq_size[1] );
        
        printf("seq size 0: %d, len: %d, w: %d\n", seq_size[0], strlen(seq[0]), g->w);
        
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
    int size = sizeof( Grid ) * grid_h->w * grid_h->h;
    grid_h->box = (int*)malloc( size );
    cudaMemcpy( grid_h->box, g->box, size, cudaMemcpyDeviceToHost);
    
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

Grid* grid_show( Grid* g )
{
    int i, j;
    for( i = 0; i < g->h; i++ )
    {
        for( j = 0; j < g->w; j++ )
        {
            int c = g->box[ i * g->w + j ];
                 if( c == 0 ) printf("    ");
            else if( c > '9') printf("  %c ", c);
            else              printf("%3d ", c);
        }
        printf("\n\n");
    }
    return g;
}

int main( int argc, char** argv )
{
    char* filename = shell_arg_string( argc, argv, "-f", "default.fasta" );
    
    Grid* grid_h = grid_new();
    // Grid* grid_d;
    
    grid_init( grid_h, 4, 4 );
    printf("g->w: %d, g->h: %d\n", grid_h->w, grid_h->h);
    
    grid_set_seq_row( grid_h, "test", 4 );
    grid_set_seq_col( grid_h, "four", 4 );
    grid_show( grid_h );
    
    // grid_load_seq( grid_h, filename );
    // grid_alloc_for_seq( grid_h, mem_host );
    // grid_alloc_for_grid( grid_d, grid_h, mem_device );
    
    // grid_copy( grid_h, grid_d );
    
    
}
