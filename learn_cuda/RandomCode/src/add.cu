#include <iostream>
#include <math.h>

// Size of array
#define N 1048576

// Kernel function to add the elements of two arrays
__global__ void add(float *a, float *b, float *c)
{
	// int id = blockDim.x * blockIdx.x + threadIdx.x;
	// if(id < N) c[id] = a[id] + b[id];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
        c[i] = a[i] + b[i];
}

// Main program
int main()
{
	// Number of bytes to allocate for N doubles
	size_t bytes = N * sizeof(float);

	// Allocate memory for arrays A, B, and C on host
	float *A = (float*)malloc(bytes);
	float *B = (float*)malloc(bytes);
	float *C = (float*)malloc(bytes);

	// Allocate memory for arrays d_A, d_B, and d_C on device
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, bytes);
	cudaMalloc(&d_B, bytes);
	cudaMalloc(&d_C, bytes);

	// Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = 1.0f;
		B[i] = 2.0f;
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

	// Set execution configuration parameters
	//		thr_per_blk: number of CUDA threads per grid block
	//		blk_in_grid: number of blocks in grid
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	// Launch kernel
	add<<< numBlocks, blockSize >>>(d_A, d_B, d_C);

	// Copy data from device array d_C to host array C
	cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

	// Verify results
    float tolerance = 1.0e-14;
	for(int i=0; i<N; i++)
	{
		if( fabs(C[i] - 3.0) > tolerance)
		{ 
			printf("\nError: value of C[%d] = %d instead of 3.0\n\n", i, C[i]);
			exit(1);
		}
	}	

	// Free CPU memory
	free(A);
	free(B);
	free(C);

	// Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
