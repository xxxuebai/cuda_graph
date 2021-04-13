#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<math.h>
#include<vector>
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
		//r[idx] = (c[idx] +a[idx]) /2;
	a[idx] = a[idx] + 1;
	}
}
void cudaGraphSample(int *dev_a,int *dev_b,int *dev_c)
{
	int i;
	cudaStream_t streamForGraph;
	cudaGraph_t graph;
	std::vector<cudaGraphNode_t> nodeDependencies;
	cudaGraphNode_t kernelNode;
	cudaStreamCreate(&streamForGraph);

	cudaKernelNodeParams kernelNodeParams = {0};
	
	cudaGraphCreate(&graph,0);
	
	void *kernelArgs[3] = {(void *)&dev_a,(void *)&dev_b,(void *)dev_c};
	kernelNodeParams.func = (void *)kernelA;
	kernelNodeParams.gridDim = dim3(1,1,1);
	kernelNodeParams.blockDim = dim3(N/GPUBLOCKNUM,1,1);
	kernelNodeParams.sharedMemBytes = 0;
	kernelNodeParams.kernelParams =(void **) kernelArgs;

	kernelNodeParams.extra = NULL;

	cudaGraphAddKernelNode(&kernelNode,graph,nodeDependencies.data(),nodeDependencies.size(),&kernelNodeParams);
	
	//nodeDependencies.clear();
	nodeDependencies.push_back(kernelNode);

	kernelNodeParams.func = (void *) kernelB;
	kernelNodeParams.gridDim = dim3(1,1,1);
        kernelNodeParams.blockDim = dim3(N/GPUBLOCKNUM,1,1);
        kernelNodeParams.sharedMemBytes = 0;
	void *kernelArgs2[1] = {(void *)&dev_a};
        kernelNodeParams.kernelParams =(void **)kernelArgs2;
	 kernelNodeParams.extra = NULL;
	cudaGraphAddKernelNode(&kernelNode,graph,nodeDependencies.data(),nodeDependencies.size(),&kernelNodeParams);
	 nodeDependencies.clear();
        nodeDependencies.push_back(kernelNode);

	cudaGraphExec_t graphExec;
	cudaGraphInstantiate(&graphExec,graph,NULL,NULL,0);
	for(i=0;i<10;i++)
	{
		cudaGraphLaunch(graphExec,streamForGraph);
	}
	cudaStreamSynchronize(streamForGraph);
	cudaGraphExecDestroy(graphExec);
	cudaGraphDestroy(graph);
	cudaStreamDestroy(streamForGraph);
}

int main()
{
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
	cudaMemcpyAsync(dev_a,host_a,N * sizeof(int),cudaMemcpyHostToDevice);
 	cudaMemcpyAsync(dev_b,host_b,N * sizeof(int),cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	
		cudaGraphSample(dev_a,dev_b,dev_c);
	
	cudaMemcpyAsync(host_c,dev_c,N * sizeof(int),cudaMemcpyDeviceToHost);
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
