#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

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

/* The HyperCubeNode Data Structure */
typedef struct HYPER_CUBE_NODE_STRUCT
{
	int id;
	int stored_sum;
	int num_nodes;
	int dimensions;
	pthread_t pthread;
	pthread_attr_t pattr;
	volatile int proceed;
} HyperCubeNode;

/* HyperCubeNode Construction / Destruction */
HyperCubeNode* hcn_initialize( int id );
HyperCubeNode* hcn_initialize_static( HyperCubeNode* self, int id );
void hcn_destroy( HyperCubeNode* self );
void hcn_destroy_static( HyperCubeNode* self );
/* HyperCubeNode Public Methods */
void hcn_thread_start( HyperCubeNode* self, void* (*action)( void* ) );
void hcn_thread_join( HyperCubeNode* self );
void* hcn_broadcast_action( void* hcn );

HyperCubeNode** nodes;

int main( int argc, char* argv[] )
{
	int i;
	int dimensions = shell_arg_int( argc, argv, "-d", 3 );
	int num_nodes  = shell_arg_int( argc, argv, "-n", 8 );
	char node_switch[] = "-0";
	
	printf( "Hypercube Broadcast\nDimensions: %d\nTotal Nodes: %d\n", dimensions, num_nodes );
	
	/* Initialize Nodes */
	nodes = calloc( num_nodes, sizeof( HyperCubeNode* ) );
	for ( i = 0; i < num_nodes; i++ )
	{
		nodes[i] = hcn_initialize( i );
		node_switch[1] = '0' + i;
		nodes[i]->stored_sum = shell_arg_int( argc, argv, node_switch, 10 );
		nodes[i]->dimensions = dimensions;
		nodes[i]->num_nodes = num_nodes;
		printf( "  Node %d starts with sum: %d\n", i, nodes[i]->stored_sum );
	}
	
	for ( i = 0; i < num_nodes; i++ )
	{
		hcn_thread_start( nodes[i], hcn_broadcast_action );
	}
	
	for ( i = 0; i < num_nodes; i++ )
	{
		hcn_thread_join( nodes[i] );
	}
	
	printf("Sum: %d\n", nodes[0]->stored_sum);
	
	/* Delete Nodes */
	for ( i = 0; i < num_nodes; i++ )
		hcn_destroy( nodes[i] );
	free( nodes );
}


/* HyperCubeNode Class */
HyperCubeNode* hcn_initialize( int id )
{
	return hcn_initialize_static( calloc( 1, sizeof( HyperCubeNode ) ), id );
}

HyperCubeNode* hcn_initialize_static( HyperCubeNode* self, int id )
{
	self->id = id;
	self->stored_sum = 0;
	self->dimensions = 0;
	self->num_nodes = 0;
	self->proceed = 0;
	pthread_attr_init( &(self->pattr) );
	pthread_attr_setdetachstate( &(self->pattr), PTHREAD_CREATE_JOINABLE );
	return self;
}

void hcn_destroy( HyperCubeNode* self )
{
	hcn_destroy_static( self );
	free( self );
}

void hcn_destroy_static( HyperCubeNode* self )
{
	return;
}

void hcn_thread_start( HyperCubeNode* self, void* (*action)( void* ) )
{
	int rc = pthread_create(
		&(self->pthread),  /* Store pthread state */
		&(self->pattr),    /* Start pthread with given attributes */
		action,            /* Start a new thread with the given action function */
		(void *)self       /* Pass the HyperCubeNode as the thread's first and only parameter */
	);
	
	if (rc) {
		printf("ERROR; return code from pthread_create() is %d\n", rc);
		exit(-1);
	}
}

void hcn_thread_join( HyperCubeNode* self )
{
	void* status;
	int rc = pthread_join( self->pthread, &status );
	if (rc)
	{
		printf("ERROR; return code from pthread_join() is %d\n", rc);
		exit(-1);
	}
}

void* hcn_broadcast_action( void* hcn )
{
	HyperCubeNode* self = (HyperCubeNode*)hcn;
	int d = self->dimensions;
	int n = self->num_nodes;
	int i;
	
	/* Set mask to 0b for 1 dim, 01b for 2 dim, 011b for 3 dim, etc. */
	// int mask = ((1 << d) - 1) ^ (1 << (d - 1));
	
	/* Set mask to */
	int mask = 1;
	
	/* Set dimensional bit to 1b for 1 dim, 10b for 2 dim, 100b for 3 dim, etc.*/
	int dim_bit = 1 << (d - 1);
	
	for ( i = 0; i < d; i++ )
	{ /* Loop through each dimension */
		printf(" i: %d, mask: %x, self->id: %d\n", i, mask, self->id);
		
		if ( mask & self->id == 0 )
		{ /* Bits indicate it's our turn to send or receive */
			
			/* The sender (if we are a recipient), or recipient (if we are a sender) */
			int other = ( self->id ^ dim_bit );
			
			if ( self->id & ( dim_bit >> (d - i) ) )
			{ /* The bits indicate it's our turn to send ... */
				printf(" %d sending to %d\n", self->id, other);
				nodes[other]->stored_sum += self->stored_sum;
				nodes[other]->proceed = 1;
			}
			else
			{ /* ... or the bits indicate it's our turn to receive */
				printf(" %d waiting to receive from %d\n", self->id, other);
				while( self->proceed == 0 );
				self->proceed = 0;
			}
			
		}
		
		mask = (mask << 1) | 1;
	}
}

// mask = 011b
// 	for (d = 0; d < n; d++)
// 	{
// 		if (mask & iproc == 0)
// 		{
// 			if (iproc & 2^(n-d-1) == 0)
// 			{
// 				send(_, iproc xor 2^(n-d-1))
// 			}
// 			else
// 			{
// 				recv(_, iproc xor 2^(n-d-1))
// 			}
// 		}
// 		mask = mask xor 2^(n-d-2)
// 	}
// 	