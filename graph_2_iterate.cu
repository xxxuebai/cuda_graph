#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<math.h>

const int N = 1024 * 1024;
const int THREAD = 256;
const int BLOCK = 256; 
__global__ void kernelA(int *a,int *b,int *c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
	{
		c[idx] = (a[idx] + b[idx])/2;
                a[idx] = a[idx] + 1;
                b[idx] = b[idx] + 1;
	}
}

void cudaGraphSample(int *dev_a,int *dev_b,int *dev_c)
{
	cudaStream_t stream0;
	cudaGraph_t graph;
	cudaStreamCreate(&stream0);
	cudaStreamBeginCapture(stream0);
	kernelA << <N/BLOCK,THREAD,0,stream0 >> >(dev_a,dev_b,dev_c);
        kernelA << <N/BLOCK,THREAD,0,stream0 >> >(dev_a,dev_b,dev_c);
	cudaStreamEndCapture(stream0,&graph);
	cudaGraphExec_t graphExec;
	cudaGraphInstantiate(&graphExec,graph,NULL,NULL,0);
	for(int i=0;i<100;i++)
	{
		cudaGraphLaunch(graphExec,stream0);
	}
	cudaStreamSynchronize(stream0);
	cudaGraphExecDestroy(graphExec);
	cudaGraphDestroy(graph);
	cudaStreamDestroy(stream0);
}

int main()
{
	int *host_a,*host_b,*host_c;
	int *dev_a,*dev_b,*dev_c;
	int i;
	cudaMalloc((void**)&dev_a,N*sizeof(int));
	cudaMalloc((void**)&dev_b,N*sizeof(int));
	cudaMalloc((void**)&dev_c,N*sizeof(int));
	cudaHostAlloc((void **)&host_a,N * sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_b,N * sizeof(int),cudaHostAllocDefault);
	cudaHostAlloc((void **)&host_c,N * sizeof(int),cudaHostAllocDefault);
	for(i=0;i<N;i++)
  	{
		host_a[i] = N - i;
		host_b[i] = i;
	}
	cudaMemcpyAsync(dev_a,host_a,N * sizeof(int),cudaMemcpyHostToDevice);
 	cudaMemcpyAsync(dev_b,host_b,N * sizeof(int),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	cudaGraphSample(dev_a,dev_b,dev_c);
	cudaMemcpyAsync(host_c,dev_r,N * sizeof(int),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
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
}
