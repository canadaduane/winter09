#ifndef MISC_H
#define MISC_H

int shell_arg_int( int argc, char* argv[], char* arg_switch, int default_value );
int* alloc_n_random( int size );
double when( void );

#endif