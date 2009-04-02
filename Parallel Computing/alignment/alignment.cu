#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <cuda.h>

#define BLOCK_SIZE 16

typedef enum BOOLEAN
{
    false,
    true
} Boolean;

typedef enum MEMORY_TYPE
{
    mem_host,
    mem_device
} MemType;

typedef struct GRID
{
    float* box;
    int w;
    int h;
    MemType mem;
    int success;
} Grid;

Grid grid_new()
{
    Grid g;
    g.w = 0;
    g.h = 0;
    g.box = NULL;
    g.mem = mem_host;
    g.success = true;
    return g;
}

Grid grid_alloc( Grid g, int w, int h, MemType mem )
{
    assert( g.success == true );

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
    assert( g.success == true );
    assert( a.w == b.w && a.h == b.h );
    assert( b.box != NULL );
    
    int size = a.w * a.h * sizeof(int);
    
         if( a.mem == mem_host   && b.mem == mem_host )   memcpy( b.mem, a.mem, size );
    else if( a.mem == mem_host   && b.mem == mem_device ) cudaMemcpy( b.mem, a.mem, size, cudaMemcpyHostToDevice);
    else if( a.mem == mem_device && b.mem == mem_host )   cudaMemcpy( b.mem, a.mem, size, cudaMemcpyDeviceToHost);
    else if( a.mem == mem_device && b.mem == mem_device ) cudaMemcpy( b.mem, a.mem, size, cudaMemcpyDeviceToDevice);
    else
    {
        a.success = false;
        b.success = false;
    }
    
    return a;
}

Grid grid_load( Grid g, char* filename )
{
    assert( g.success == true );
    
    
}

int main( int argc, char** argv )
{
    char* filename = shell_arg_string( argc, argv, "-f", "default.fasta" );
    
    Grid grid_h = grid_alloc( grid_new(), 64, 64, mem_host );
    Grid grid_d = grid_alloc( grid_new(), 64, 64, mem_device );
    
    grid_copy( grid_load( grid_h, filename ), grid_d );
    
    
}
