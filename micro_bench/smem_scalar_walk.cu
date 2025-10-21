
#include <stdio.h> 
#include <stdlib.h>
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>

// Define a safe macro for CUDA error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

constexpr int N_iter = 1000;
constexpr int N_threads = 1;
constexpr int N_blocks = 1;

// 64-bit clock counter: This is correct for PTX `clock64`.
__device__ __forceinline__ unsigned long long get_clock64() {
    unsigned long long clock_val;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(clock_val));
    return clock_val;
}

// NOTE: Kernel signature requires 4 arguments.
__global__ void scalar_smem_latency (float* In, float* Out, unsigned long long *starts, unsigned long long *stops)
{
  int t = threadIdx.x; 
  __shared__ float S[N_iter]; 
  float r1; 
 
  unsigned long long s; 
  unsigned long long e; 
  
  for (int j = 0; j < N_iter; j++)

  { 
      if(j < 10)
      {
        printf("%f  ", In[t+j]);
      }

      S[t+j] = In[t+j];
  }
  
  __syncthreads();
  
  for (int j = 0; j < N_iter; j+=2)
  {
    float* ptr_0 = &S[t + j]; 
    int addr_0 = (int)ptr_0 & 0xFFFF;
    float* ptr_1 = &S[t + j + 1]; 
    int addr_1 = (int)ptr_1 & 0xFFFF;
    
    s = get_clock64();
      asm volatile ("ld.volatile.shared.f32 %0, [%1]; \n"
                      : "=f"(r1)
                      : "r"(addr_0));
      asm volatile ("ld.volatile.shared.f32 %0, [%1]; \n"
                      : "=f"(r1)
                      : "r"(addr_1));
      
    e = get_clock64();
    starts[t + j] = s; 
    stops[t + j] = e;
    Out[t + j + 1] = r1;

  }
  
}


int main()
{
    //# --- 1. Variable Declarations and Size Calculations ---
    unsigned long long *h_start, *h_stop;
    unsigned long long *d_start, *d_stop;
    
    //# Size for cycle arrays (N_iter measurements for 1 thread)
    size_t size_cycles = N_iter * sizeof(unsigned long long);
    
    //# Size for I/O float arrays
    size_t size_IO = N_iter * sizeof(float);
    float* h_In, *h_out;
    float* D_In, *D_out; // Device pointers for In and Out (required by kernel signature)

    //# --- 2. Host and Device Memory Allocation and Initialization ---
    
    //# Allocate Pinned Host memory for In and Out (for setup)
    CUDA_CHECK(cudaHostAlloc(&h_In, size_IO, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_out, size_IO, cudaHostAllocDefault)); 
    
    //# Corrected loop syntax: semicolon instead of comma in the condition
    for (int i = 0; i < N_iter; i++) 
    {
        h_In[i] = (float) (i+1) / (100.0); 
        h_out[i] = 0.0f; // #Use 0.0f for float literal
    }
    
    //# Allocate Pinned Host memory for start/stop cycles
    CUDA_CHECK(cudaHostAlloc(&h_start, size_cycles, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_stop, size_cycles, cudaHostAllocDefault));
    
    // Allocate Device memory for all 4 kernel arguments
    CUDA_CHECK(cudaMalloc(&d_start, size_cycles));
    CUDA_CHECK(cudaMalloc(&d_stop, size_cycles));
    CUDA_CHECK(cudaMalloc(&D_In, size_IO));
    CUDA_CHECK(cudaMalloc(&D_out, size_IO));

    // Copy input data to the device (D_In)
    CUDA_CHECK(cudaMemcpy(D_In, h_In, size_IO, cudaMemcpyHostToDevice));
    
    // --- 3. Kernel Launch ---

    printf("Launching kernel with N_blocks=%d and N_threads=%d...\n", N_blocks, N_threads);

    // Launch the kernel with all 4 required arguments
    scalar_smem_latency<<<N_blocks, N_threads>>>(D_In, D_out, d_start, d_stop);
    
    // Wait for the kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // --- 4. Data Transfer and Processing ---
    
    // Copy the cycle counters back to the host
    CUDA_CHECK(cudaMemcpy(h_start, d_start, size_cycles, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_stop, d_stop, size_cycles, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out, D_out, size_IO, cudaMemcpyDeviceToHost));

    float avg = 0.0;
    
    for (int i = 0; i < N_iter; i++)
    {
        // Safe subtraction using unsigned long long
        unsigned long long diff = h_stop[i] - h_start[i];
        printf("value of r1 at iteration %d: %f \n", i, h_out[i]);
        printf("Latency at iteration %d: %llu cycles \n", i, diff);
        avg += ((float)diff/2.0);

    }
    printf("average latency %f \n", avg/500.0);
    // --- 5. Cleanup ---
    
    cudaFree(d_start);
    cudaFree(d_stop);
    cudaFree(D_In);
    cudaFree(D_out);
    
    cudaFreeHost(h_start);
    cudaFreeHost(h_stop);
    cudaFreeHost(h_In);
    cudaFreeHost(h_out);
    
    return 0;
}