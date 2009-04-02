#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>

#include "misc.h"

#define BLOCK_SIZE 16

// typedef enum BOOLEAN
// {
//     false,
//     true
// } Boolean;

typedef enum MEMORY_TYPE
{
    mem_host,
    mem_device
} MemType;

typedef struct GRID
{
    int w; // Width of grid
    int h; // Height of grid
    float* box;
    char* x; // Sequence across top of grid
    char* y; // Sequence on left of grid
    MemType mem;
    int success;
} Grid;

Grid grid_new();
Grid grid_alloc( Grid g, int w, int h, MemType mem );
Grid grid_alloc_for_seq( Grid g, MemType mem );
Grid grid_alloc_for_grid( Grid g, Grid other, MemType mem );
Grid grid_free( Grid g );
Grid grid_clear( Grid g );
Grid grid_copy( Grid a, Grid b );
Grid grid_load_seq( Grid g, char* filename );

Grid grid_new()
{
    Grid g;
    g.w = 0;
    g.h = 0;
    g.box = NULL;
    g.x = NULL;
    g.y = NULL;
    g.mem = mem_host;
    g.success = true;
    return g;
}

Grid grid_alloc( Grid g, int w, int h, MemType mem )
{
    assert( g.success == true );
    
    if( g.box != NULL ) grid_free( g );
    
    g.w = w;
    g.h = h;
    g.mem = mem;
    switch( g.mem )
    {
        case mem_device: cudaMalloc( (void**)& g.box, w * h * sizeof( float ));  break;
        case mem_host:   g.box = (float *)malloc( w * h * sizeof( float ) );     break;
    }
    return g;
}

Grid grid_alloc_for_seq( Grid g, MemType mem )
{
    return grid_alloc( g, strlen(g.x), strlen(g.y), mem );
}

Grid grid_alloc_for_grid( Grid g, Grid other, MemType mem )
{
    return grid_alloc( g, other.w, other.h, mem );
}

Grid grid_free( Grid g )
{
    assert( g.success == true );

    switch( g.mem )
    {
        case mem_device: cudaFree( g.box );  break;
        case mem_host:   free( g.box );      break;
    }
    return g;
}

Grid grid_clear( Grid g )
{
    assert( g.success == true );

    int size = g.w * g.h * sizeof(int);
    switch( g.mem )
    {
        case mem_device: cudaMemset( g.box, 0, size );  break;
        case mem_host:   memset( g.box, 0, size );      break;
    }
    return g;
}

// Copy grid 'a' to grid 'b'
Grid grid_copy( Grid a, Grid b )
{
    assert( a.success == true && b.success == true );
    assert( a.w == b.w && a.h == b.h );
    assert( b.box != NULL );
    
    int size = a.w * a.h * sizeof(int);
    
         if( a.mem == mem_host   && b.mem == mem_host )   memcpy( b.box, a.box, size );
    else if( a.mem == mem_host   && b.mem == mem_device ) cudaMemcpy( b.box, a.box, size, cudaMemcpyHostToDevice);
    else if( a.mem == mem_device && b.mem == mem_host )   cudaMemcpy( b.box, a.box, size, cudaMemcpyDeviceToHost);
    else if( a.mem == mem_device && b.mem == mem_device ) cudaMemcpy( b.box, a.box, size, cudaMemcpyDeviceToDevice);
    else
    {
        a.success = false;
        b.success = false;
    }
    
    return a;
}

Grid grid_load_seq( Grid g, char* filename )
{
    assert( g.success == true );
    assert( g.mem == mem_host );
    
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
        
        // Free up previously allocated sequences, if any
        if( g.x != NULL ) free( g.x );
        if( g.y != NULL ) free( g.y );
        
        // Grid sequences point to newly loaded sequences
        g.x = seq[0];
        g.y = seq[1];
        
        printf("seq size 0: %d, len: %d, w: %d\n", seq_size[0], strlen(seq[0]), g.w);
        
        fclose( input );
    }
    else
    {
        g.success = false;
    }
    
    return g;
}

int main( int argc, char** argv )
{
    char* filename = shell_arg_string( argc, argv, "-f", "default.fasta" );
    
    // Grid grid_h = grid_alloc( grid_new(), 64, 64, mem_host );
    // Grid grid_d = grid_alloc( grid_new(), 64, 64, mem_device );
    Grid grid_h = grid_new();
    Grid grid_d = grid_new();
    
    grid_load_seq( grid_h, filename );
    grid_alloc_for_seq( grid_h, mem_host );
    // grid_alloc_for_grid( grid_d, grid_h, mem_device );
    
    // grid_copy( grid_h, grid_d );
    
    
}
