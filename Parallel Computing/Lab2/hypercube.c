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
	int state;
	int message_box;
	int message_received;
	int num_nodes;
	int dimensions;
	pthread_t pthread;
	pthread_attr_t pattr;
	pthread_cond_t message_cond;
	pthread_mutex_t message_mutex;
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
		nodes[i]->state = shell_arg_int( argc, argv, node_switch, 10 );
		nodes[i]->dimensions = dimensions;
		nodes[i]->num_nodes = num_nodes;
		printf( "  Node %d starts with sum: %d\n", i, nodes[i]->state );
	}
	
	for ( i = 0; i < num_nodes; i++ )
	{
		hcn_thread_start( nodes[i], hcn_broadcast_action );
	}
	
	for ( i = 0; i < num_nodes; i++ )
	{
		hcn_thread_join( nodes[i] );
	}
	
	printf("Sum: %d\n", nodes[0]->state);
	
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
	self->state = 0;
	self->message_box = 0;
	self->message_received = 0;
	self->dimensions = 0;
	self->num_nodes = 0;
	pthread_attr_init( &(self->pattr) );
	pthread_attr_setdetachstate( &(self->pattr), PTHREAD_CREATE_JOINABLE );
	pthread_cond_init( &(self->message_cond), NULL );
	pthread_mutex_init( &(self->message_mutex), NULL );
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
	
	pthread_cond_t* cond;
	pthread_mutex_t* mutex;
	
	for ( i = 0; i < d; i++ )
	{ /* Loop through each dimension */
		printf("Iteration %d, self->id: %d\n", i, self->id);
		
		/* The sender (if we are a recipient), or recipient (if we are a sender) */
		int other = ( self->id ^ ( 1 << i ) );
		
		if ( self->id < other )
		{
			cond = &(self->message_cond);
			mutex = &(self->message_mutex);
		}
		else
		{
			cond = &(nodes[other]->message_cond);
			mutex = &(nodes[other]->message_mutex);
		}
		
		printf( "   %d --(%d)-> %d\n", self->id, self->state, other );
		pthread_mutex_lock( mutex );
		{
			nodes[other]->message_box = self->state;
			nodes[other]->message_received = 1;
			pthread_cond_signal( cond );
		}
		pthread_mutex_unlock( mutex );

		printf( " W %d <- ? -- %d\n", self->id, other );
		pthread_mutex_lock( mutex );
		{
			while( !self->message_received )
				pthread_cond_wait( cond, mutex );
			// Reset 'message received' flag
			self->message_received = 0;
			printf( "   %d <-(%d)-- %d\n", self->id, self->message_box, other );
		}
		pthread_mutex_unlock( mutex );
		
		/* Perform the reduction operation: */
		self->state = self->state + self->message_box;

		printf( " F %d: %d\n", self->id, self->state);
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