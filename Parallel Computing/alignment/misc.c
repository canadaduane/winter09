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
