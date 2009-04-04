#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Look for a command-line switch and get the next integer value */
int shell_arg_int( int argc, char* argv[], char* arg_switch, int default_value )
{
    int i;
    for ( i = 0; i < argc - 1; i++ )
    {
        if ( strcmp( argv[ i ], arg_switch ) == 0 )
        {
            return atoi( argv[ i + 1 ] );
        }
    }
    return default_value;
}

int shell_arg_present( int argc, char* argv[], char* arg_switch )
{
    int i;
    for ( i = 0; i < argc - 1; i++ )
    {
        if ( strcmp( argv[ i ], arg_switch ) == 0 )
        {
            return 1;
        }
    }
    return 0;
}

/* Look for a command-line switch and get the next double value */
float shell_arg_float( int argc, char* argv[], char* arg_switch, float default_value )
{
    int i;
    for ( i = 0; i < argc - 1; i++ )
    {
        if ( strcmp( argv[ i ], arg_switch ) == 0 )
        {
            return atof( argv[ i + 1 ] );
        }
    }
    return default_value;
}

char* shell_arg_string( int argc, char* argv[], char* arg_switch, char* default_value )
{
    int i;
    for ( i = 0; i < argc - 1; i++ )
    {
        if ( strcmp( argv[ i ], arg_switch ) == 0 )
        {
            return argv[ i + 1 ];
        }
    }
    return default_value;
}

/* Return the current time in seconds, using a double precision number. */
double when()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

/**
 * Returns the floor form of binary logarithm for a 32 bit integer.
 * -1 is returned if n is 0.
 */
int floor_log2(unsigned int n) {
  int pos = 0;
  if (n >= 1<<16) { n >>= 16; pos += 16; }
  if (n >= 1<< 8) { n >>=  8; pos +=  8; }
  if (n >= 1<< 4) { n >>=  4; pos +=  4; }
  if (n >= 1<< 2) { n >>=  2; pos +=  2; }
  if (n >= 1<< 1) {           pos +=  1; }
  return ((n == 0) ? (-1) : pos);
}

// Source: http://www.dreamincode.net/code/snippet2997.htm
/* 
    Returns NULL to indicate EOF ... or the address of a NEW C string.
    REMEMBER to FREE the memory when done with this string.
*/
char* readline(FILE* f)
{
    static int c = 0; /* static ... to remember if a previous call set EOF */
    static int lineCount = 0; /* to handle the empty file ... */
    int bufSize = 255, i = 0; /* adjust 255 to whatever is 'right' for your data */
    
    if( c == EOF ) { c = 0; return NULL; } /* reset c so can rewind and read again */
    
    char* line = (char*) calloc(bufSize, sizeof(char));
    
    while ( (c = fgetc(f)) != EOF && c != '\n' )
    {
        if( i >= bufSize )
        {
            bufSize += 256; /* adjust 256 to whatever is 'right' for your data */
            line = (char*) realloc(line, bufSize * sizeof(char));
        }
        line[i++] = c;
    }
    /* handle special case of empty file ...*/
    if( lineCount++ == 0 && c == EOF ) { free(line); return NULL; }
    
    line[i] = '\0'; /* confirm terminal 0 */
    return realloc( line, i+1 ); /* total len =  last index i ... + 1 more */
}

int min2( int a, int b )
{
    return (a < b ? a : b);
}

int min3( int a, int b, int c )
{
    return (a < b ? (a < c ? a : (b < c ? b : c)) : (b < c ? b : c));
}

int max2( int a, int b )
{
    return (a > b ? a : b);
}

int max3( int a, int b, int c )
{
    return (a > b ? (a > c ? a : (b > c ? b : c)) : (b > c ? b : c));
}
