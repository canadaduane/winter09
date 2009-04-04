#ifndef MISC_H
#define MISC_H

extern "C" int    shell_arg_int( int argc, char* argv[], char* arg_switch, int default_value );
extern "C" int    shell_arg_present( int argc, char* argv[], char* arg_switch );
extern "C" float  shell_arg_float( int argc, char* argv[], char* arg_switch, float default_value );
extern "C" char*  shell_arg_string( int argc, char* argv[], char* arg_switch, char* default_value );
extern "C" double when( void );
extern "C" int    floor_log2( unsigned int n );
extern "C" char*  readline( FILE* f );
extern "C" int    min2( int a, int b );
extern "C" int    min3( int a, int b, int c );
extern "C" int    max2( int a, int b );
extern "C" int    max3( int a, int b, int c );

#endif