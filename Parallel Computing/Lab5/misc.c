#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Look for a command-line switch and get the next integer value */
int shell_arg_int( int argc, char* argv[], char* arg_switch, int default_value)
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

/* Return the current time in seconds, using a double precision number. */
double when()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}
