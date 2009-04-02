#ifndef MISC_H
#define MISC_H

int    shell_arg_int( int argc, char* argv[], char* arg_switch, int default_value );
float  shell_arg_float( int argc, char* argv[], char* arg_switch, float default_value );
double when( void );
int    floor_log2(unsigned int n);

#endif