#include<stdio.h>
#include<sys/time.h>
#include<cuda.h>
#define BLOCK_SIZE 16

typedef struct GRID
{
    float* box;
    int w;
    int h;
} Grid;

typedef enum MEMORY_TYPE
{
    mem_device,
    mem_host
} MemType;

Grid gridAlloc( int w, int h, MemType l )
{
    Grid g;
    g.w = w;
    g.h = h;
    switch( l )
    {
        case mem_device: cudaMalloc( (void**)& g.box, w * h * sizeof( float )); break;
        case mem_host:   g.box = (float *)malloc( w * h * sizeof( float ) ); break;
        default: exit(254);
    }
    return g;
}

/////////////////////////////////////////////////////////////////////////////////////////
//
// PieValueFun : The device kernel to calculate the PI value  
//
/////////////////////////////////////////////////////////////////////////////////////////
 
__global__ void PieValueFun(float *dIntervalLocalValue, float numInterval, int maxNumThread)
{
    int tindex = blockIdx.x * blockDim.x + threadIdx.x;
    int intervalCount ;
    
    float h = 1.0 / numInterval;
    float sum  = 0.0; 
    float x = 0.0;
    dIntervalLocalValue[tindex] = 0.0;
 
    for( intervalCount = tindex + 1; intervalCount <= numInterval; intervalCount += maxNumThread )
    {
          x = h * ( (float)intervalCount - 0.5 ); 
          sum += (4.0 /(1.0 + x*x)); 
    }
    dIntervalLocalValue[tindex] = h * sum;
    //printf("%d: I put in a %f\n", tindex, dIntervalLocalValue[tindex]);

}//end of PieValueFun device function

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory
*/
__global__ void reduceSum(float *g_idata, float *g_odata)
{
    extern __shared__ float sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        *g_odata = sdata[0];
        //printf("Reduced to %d\n", *g_odata);
    }
}

/* Return the current time in seconds, using a double precision number.       */
double When()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

int main(int argc, char* argv[])
{
	float             *dIntervalLocalValue;
	float             *hPieValue;
	int               numInterval;
	float	          *dPieValue;
	int 	          nblocks = 1;
	int               numThread = nblocks * BLOCK_SIZE;

	if(argc != 2 )
	{
		printf("\n Invalid Number of argument! ");
		printf("\n ./<executable name> <Number of Interval>\n\n");
		exit(-1);
	}
	else
		numInterval = atoi(argv[1]);

	//allocation host memory for retriving resultant  value from device.
	hPieValue = (float*) malloc( sizeof(float));

	//allocation of device memory
	cudaMalloc( (void**)&dIntervalLocalValue, numThread * sizeof(float));
	cudaMalloc( (void**)&dPieValue, 1 * sizeof(float));

	// defining thread grid and block
	dim3 dimGrid(nblocks);
	dim3 dimBlock(BLOCK_SIZE);                            

	double t0 = When();
	//calling device kernel
	PieValueFun<<<dimGrid, dimBlock>>>(dIntervalLocalValue,numInterval,nblocks*BLOCK_SIZE); 
	reduceSum<<< 1, numThread/2, numThread/2 * sizeof(float) >>>(dIntervalLocalValue, dPieValue);

	//retriving result from device
	cudaMemcpy((void*)hPieValue, (void*)dPieValue, sizeof(float) , cudaMemcpyDeviceToHost );
	double t = When() - t0;



	printf("\n ----------------------------------------------------------------------\n ");
	printf( "Number Of Interval : %d               PI = %f \n", numInterval,*hPieValue);
	printf( "Computed in %lf seconds", t);
	printf("\n ----------------------------------------------------------------------\n ");

	cudaFree(dIntervalLocalValue);
	free(hPieValue);
 }// end of main
