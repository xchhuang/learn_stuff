// #include <iostream>
// #include <math.h>

// // function to add the elements of two arrays
// __global__ void add(int n, float *a, float *b, float *c)
// {
//     // int index = blockIdx.x * blockDim.x + threadIdx.x;
//     // int stride = blockDim.x * gridDim.x;
//     // for (int i = index; i < n; i += stride)
//     //     y[i] = x[i] + y[i];
    
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) 
//     {
//         c[i] = a[i] + b[i];
//     }
// }

// int main(void)
// {
//     int N = 1 << 20; // 1M elements
//     // std::cout << "N: " << N << std::endl;
//     float* a, *b, *c;
//     size_t size = N * sizeof(float);
//     cudaMallocManaged(&a, size);
//     cudaMallocManaged(&b, size);
//     cudaMallocManaged(&c, size);
    

//     // initialize x and y arrays on the host
//     for (int i = 0; i < N; i++) {
//         a[i] = 1.0f;
//         b[i] = 2.0f;
//         c[i] = 0.0f;
//     }

//     // Run kernel on 1M elements on the CPU
//     int blockSize = 256;
//     int numBlocks = (N + blockSize - 1) / blockSize;
//     // std::cout << "blockSize: " << blockSize << ", numBlocks: " << numBlocks << std::endl;
//     add <<< 1, 1 >>> (N, a, b, c);

//     cudaDeviceSynchronize();

//     // Check for errors (all values should be 3.0f)
//     // float maxError = 0.0f;
//     // for (int i = 0; i < N; i++)
//     //     maxError = fmax(maxError, fabs(c[i] - 3.0f));
//     // std::cout << "Max error: " << maxError << std::endl;

//     // Free memory
//     cudaFree(a);
//     cudaFree(b);
//     cudaFree(c);
    

//     return 0;
// }






#include <stdio.h>

// Size of array
#define N 1048576

// Kernel
__global__ void add_vectors(float *a, float *b, float *c)
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
	add_vectors<<< numBlocks, blockSize >>>(d_A, d_B, d_C);

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

	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", blockSize);
	printf("Blocks In Grid    = %d\n", numBlocks);
	printf("---------------------------\n\n");

	return 0;
}
