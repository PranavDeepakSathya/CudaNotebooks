#include <cuda_runtime.h> 
#include <cuda.h> 
#include<cooperative_groups.h> 
#include<stdlib.h>
#include<stdio.h>
/*
instead of being a pussy ass bitch, lets thing about a basic smem reduction kernel. 
say N is our big problem size, and BN is what a singular block would work on. 
indeed the partition [0, BN-1], [BN, 2BN-1]... [(k-1)BN, N-1] would be a k blocks. 
And this partition is okay, because shared memory and shit is per block. 
So, we have num_TPB threads in a block, ITS MATHIN TIME 

*/


__global__ void grid_sync_reduction(float* A_in, float*A_out)
{

}

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            return 1;                                                             \
        }                                                                         \
    } while (0)

int main ()
{
  int device = 0; 
  int supports_coop_launch = 0; 
  int num_TPB = 256; 
  int blocks_per_sm = 0; 
  cudaDeviceProp deviceProp;
  int num_SMs_on_device = 0;
  int K_num_blocks = 0;

  // 1. Get Cooperative Launch Attribute
  CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, device));
  
  // 2. Get Device Properties
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));
  num_SMs_on_device = deviceProp.multiProcessorCount;
  
  // 3. Calculate Occupancy (Max Active Blocks Per SM)
  // This assumes 'grid_sync_reduction' is the actual kernel name.
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, (void*)grid_sync_reduction, num_TPB, 0));
  
  // 4. Calculate Total Co-Schedulable Blocks
  K_num_blocks = blocks_per_sm * num_SMs_on_device;

  // ----------------------------------------------------------------------
  // --- Pretty Print of all important numbers ---
  printf("\n--- CUDA Device Occupancy Analysis (Device %d) ---\n", device);
  printf("Device Name: %s\n", deviceProp.name);
  printf("----------------------------------------------------\n");
  printf("1. Supports Cooperative Launch:         %s (%d)\n", 
         supports_coop_launch ? "Yes" : "No", supports_coop_launch);
  printf("2. Kernel Threads Per Block (num_TPB):  %d\n", num_TPB);
  printf("3. Number of Streaming Multiprocessors: %d\n", num_SMs_on_device);
  printf("4. Max Active Blocks Per SM (Occupancy):%d\n", blocks_per_sm);
  printf("5. Total Co-Schedulable Blocks (K_num_blocks):\n");
  printf("   (SMs * Blocks/SM) = %d * %d = %d\n", num_SMs_on_device, blocks_per_sm, K_num_blocks);
  printf("----------------------------------------------------\n\n");
  
  return 0;
}
