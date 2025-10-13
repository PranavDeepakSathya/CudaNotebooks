#include <cuda_runtime.h> 
#include <cuda.h> 
#include<cooperative_groups.h> 
#include<stdlib.h>
#include<stdio.h>



__global__ void grid_sync_reduction(float* A_in, float*A_out)
{

}

int main ()
{
  int device = 0; 
  int supports_coop_launch = 0; 
  cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, device);
  printf("%d", supports_coop_launch);
  int num_TPB = 256; 
  int blocks_per_sm = 0; 
  //we can calculate on our device how many blocks can be co-scheduled on a single SM. 
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, grid_sync_reduction, num_TPB, 0);
  printf("%d", blocks_per_sm);
}
