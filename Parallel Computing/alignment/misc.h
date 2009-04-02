#ifndef MISC_H
#define MISC_H

extern "C" int    shell_arg_int( int argc, char* argv[], char* arg_switch, int default_value );
extern "C" float  shell_arg_float( int argc, char* argv[], char* arg_switch, float default_value );
extern "C" char*  shell_arg_string( int argc, char* argv[], char* arg_switch, char* default_value );
extern "C" double when( void );
extern "C" int    floor_log2( unsigned int n );
extern "C" char*  readline( FILE* f );

#endif