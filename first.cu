/*
 源链接：
 https://blog.csdn.net/smartcat2010/article/details/105167981
 https://developer.nvidia.com/blog/cuda-graphs/
*/

//初始
#define NSTEP 1000
#define NKERNEL 20
 
// start CPU wallclock timer
for(int istep=0; istep<NSTEP; istep++){
  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
    shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
    cudaStreamSynchronize(stream);
  }
}
//end CPU wallclock time

/*
总共平均耗时9.6μs；kernel执行耗时2.9us；
缺点：启动kernel-->执行kernel-->等待执行完；

device和host是异步的，当CPU调用device函数后就返回了;
cudaMemcpy函数是个同步函数。
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count,cudaMemcpyKind kind, cudaStream_t stream = 0);
值得注意的就是最后一个参数，stream表示流，一般情况设置为默认流，这个函数和主机是异步的，执行后控制权立刻归还主机，
*/

//改进
// start wallclock timer
for(int istep=0; istep<NSTEP; istep++){
  for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
    shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
  }
  cudaStreamSynchronize(stream);
}
//end wallclock timer

/*
 总共平均耗时3.8μs；kernel执行耗时2.9us；
 优点：启动下一个kernel和执行上一个kernel，能够并行起来；
 缺点：每个kernel还得启动一次；
*/


//Graph优化版本：
bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;
for(int istep=0; istep<NSTEP; istep++){
  if(!graphCreated){
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
      shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    graphCreated=true;
  }
  cudaGraphLaunch(instance, stream);
  cudaStreamSynchronize(stream);
}
//总共平均耗时3.4μs；kernel执行耗时2.9us；
//优点：整个graph启动一次；头一次构建graph慢，但是后面的迭代就可以复用该graph了；

