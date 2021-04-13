#include<cuda_runtime.h>
#include<iostream>
#include<stdio.h>
#include<math.h>

const int N = 1024 * 1024;
const int GPUTHREADNUM = 256;
const int GPUBLOCKNUM = 256; 
__global__ void kernelA(int *a,int *b,int *c)//核函数A 两个数组平均数
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
	{
		c[idx] = (a[idx] + b[idx])/2;
	}
}

__global__ void kernelB(int *c)//核函数B，A的结果+1
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
	{
		c[idx] = c[idx] +1;
	}
}
void cudaGraphSample(int *dev_a,int *dev_b,int *dev_c)//图任务建立
{
	cudaStream_t stream0,streamForGraph;//定义两个流
	cudaGraph_t graph;//定义一个图
	cudaStreamCreate(&stream0);//创建流
	cudaStreamCreate(&streamForGraph);//创建流
	cudaStreamBeginCapture(stream0);//开始流捕获模式
	kernelA << <N/GPUBLOCKNUM , GPUTHREADNUM,0,stream0 >> >(dev_a,dev_b,dev_c);//执行核函数A
        kernelB << <N/GPUBLOCKNUM,GPUTHREADNUM,0,stream0 >> >(dev_c);//执行核函数B
	cudaStreamEndCapture(stream0,&graph);//结束流捕获模式
	cudaGraphExec_t graphExec;//创建一个实例
	cudaGraphInstantiate(&graphExec,graph,NULL,NULL,0);//实例化
	cudaGraphLaunch(graphExec,streamForGraph);//启动图
	cudaStreamSynchronize(streamForGraph);//同步
	cudaGraphExecDestroy(graphExec);//销毁
	cudaGraphDestroy(graph);
	cudaStreamDestroy(streamForGraph);
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
	cudaMemcpyAsync(host_c,dev_c,N * sizeof(int),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
//	for(i=0;i<20;i++)
//	{
//		for(int j=0;j<N;j++)
//	{
		printf("%d ",host_c[0]);
//		}
//		printf("\n");
//		}
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
