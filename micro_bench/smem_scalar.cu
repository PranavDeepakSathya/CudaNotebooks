#include<stdio.h> 
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h> 

constexpr int N_iter = 100000;
constexpr int N_warps = 32;
constexpr int N_lanes = 32;
constexpr int N_blocks = 1;

__device__ __forceinline__ unsigned long long get_clock64() {
    unsigned long long clock_val;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock_val));
    return clock_val;
}

__global__ void scalar_smem_latency (unsigned long long *starts, unsigned long long *stops)
{
    __shared__ float A[N_warps][N_lanes]; 
    int warp_id = threadIdx.y; 
    int lane_id = threadIdx.x; 

    float* ptr = &A[warp_id][lane_id];
    int addr = (int)ptr & 0xFFFF;
    
    float* ptr_new = &A[warp_id + 3][lane_id + 11]; 
    int addr_new = (int)ptr & 0xFFFF;
    
    float r1 = {0.9}; 
    
    for (int u = 0; u < N_warps; u++)
    {
        for (int v = 0; v < N_lanes; v++)
        {
            A[u][v] = 0.3;
        }
    }

    unsigned long long s; 
    unsigned long long e;
    
    for (int i = 0; i < N_iter; i++)
    {
        s = get_clock64();
        
        asm volatile ("ld.volatile.shared.f32 %0, [%1]; \n"
                      : "=f"(r1)
                      : "r"(addr));
        
        asm volatile ("ld.volatile.shared.f32 %0, [%1]; \n"
                      : "=f"(r1)
                      : "r"(addr_new));

        
        e = get_clock64(); 
        
        starts[i + (warp_id*32 + lane_id)] = s; 
        stops[i + (warp_id*32 + lane_id)] = e; 
        
        printf("reg_value: %f \n ", r1);
    }
    
    
}


int main()
{
    unsigned long long *h_start, *h_stop;
    unsigned long long *d_start, *d_stop;
    size_t size = N_iter * sizeof(unsigned long long);
    cudaHostAlloc(&h_start, size, cudaHostAllocDefault);
    cudaHostAlloc(&h_stop, size, cudaHostAllocDefault);
    cudaMalloc(&d_start,size);
    cudaMalloc(&d_stop, size);
    
    scalar_smem_latency<<<1,1>>>(d_start, d_stop);
    
    cudaMemcpy(h_start, d_start, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stop, d_stop, size, cudaMemcpyDeviceToHost);
    
    /*
    """
    for (int i = 0; i < N_iter; i ++)
    {
        printf("start_cycle: %lld, stop_cycle: %lld \n", h_start[i], h_stop[i]);
    }
    """
    */
    for (int i = 0; i < N_iter; i++)
    {
        unsigned long long diff = h_stop[i] - h_start[i];
        printf("latency at %d : %lld \n", i, diff);
    }
    
    cudaFree(d_start);
    cudaFree(d_stop);
    cudaFreeHost(h_start);
    cudaFreeHost(h_stop);
}