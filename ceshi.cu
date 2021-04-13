#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<math.h>

const int N = 1024 * 1024;
const int GPUTHREADNUM = 256;
const int GPUBLOCKNUM = 256; 
__global__ void kernelA(int *a,int *b,int *c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
	{
		c[idx] = (a[idx] + b[idx])/2;
	
	}
}

__global__ void kernelB(int *a)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
	{
	//	r[idx] = (c[idx] +a[idx])/2;
	a[idx] = a[idx] +1;
	}
}

int main()
{
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	int *host_a,*host_b,*host_c;
	int *dev_a,*dev_b,*dev_c;
	int i;
	cudaMalloc((void**)&dev_a,N*sizeof(int));
	cudaMalloc((void**)&dev_b,N*sizeof(int));
	cudaMalloc((void**)&dev_c,N*sizeof(int));
	//cudaMalloc((void**)&dev_r,N*sizeof(int));

	cudaHostAlloc((void **)&host_a,N * sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_b,N * sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_c,N * sizeof(int),cudaHostAllocDefault);
	for(i=0;i<N;i++)
  	{
		host_a[i] = N - i;
		host_b[i] = i;
	}
	cudaMemcpyAsync(dev_a,host_a,N * sizeof(int),cudaMemcpyHostToDevice,stream);
 	cudaMemcpyAsync(dev_b,host_b,N * sizeof(int),cudaMemcpyHostToDevice,stream);
	for(i=0;i<100;i++)
	{
		kernelA << <N/GPUBLOCKNUM , GPUTHREADNUM,0,stream >> >(dev_a,dev_b,dev_c);
		kernelB << <N/GPUBLOCKNUM,GPUTHREADNUM,0,stream >> >(dev_a);
	}
	cudaMemcpyAsync(host_c,dev_c,N * sizeof(int),cudaMemcpyDeviceToHost,stream);
	cudaStreamSynchronize(stream);
	for(i=0;i<10;i++)
	{
		printf("%d ",host_c[i]);
	}

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	cudaStreamDestroy(stream);
}
