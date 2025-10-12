#include <cuda_runtime.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <cooperative_groups.h>

// --- Configuration Constants ---
// Define the dimensions for a simple 1D buffer operation
constexpr int DATA_SIZE = 1024 * 1024; 

// Kernel Launch Configuration (Modern Cluster Launch)
constexpr int n_blocks_per_grid = 128; 
constexpr int n_blocks_per_cluster = 2;
constexpr int n_threads_per_block = 1024;
// -------------------------------

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d in %s: %s\n",
                file, line, func, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/*
Never mind the rest of the code here, lets get to a basic sync pattern. 
*/
__global__ void generic_kernel(const float *In1, const float *In2, float *Out)
{


    
}


int main ()
{
    // --- 1. Host Memory Allocation ---
    size_t size_bytes = DATA_SIZE * sizeof(float); 

    float *In1_h = (float*) malloc(size_bytes);
    float *In2_h = (float*) malloc(size_bytes); 
    float *Out_h = (float*) malloc(size_bytes); 
    
    if (!In1_h || !In2_h || !Out_h)
    {
        fprintf(stderr, "Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize inputs (Placeholder: fill with simple data for testing)
    for (int i = 0; i < DATA_SIZE; i++)
    {
        In1_h[i] = (float)i*0.001f;
        In2_h[i] = (float)i * 0.002f;
        Out_h[i] = 0.0f; 
    }
    
    // --- 2. Device Memory Allocation ---
    float* In1_d, *In2_d, *Out_d; 
    CHECK_CUDA_ERROR(cudaMalloc(&In1_d, size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&In2_d, size_bytes)); 
    CHECK_CUDA_ERROR(cudaMalloc(&Out_d, size_bytes)); 

    // --- 3. Copy Inputs to Device ---
    CHECK_CUDA_ERROR(cudaMemcpy(In1_d, In1_h, size_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(In2_d, In2_h, size_bytes, cudaMemcpyHostToDevice));
    // Optional: Copy output buffer to device if required, otherwise zero-init is fine.
    CHECK_CUDA_ERROR(cudaMemcpy(Out_d, Out_h, size_bytes, cudaMemcpyHostToDevice)); 
    
    // --- 4. Configure Cluster Launch (Modern CUDA API) ---
    dim3 TPB(n_threads_per_block, 1, 1);
    dim3 BPG(n_blocks_per_grid, 1, 1);
    
    cudaLaunchConfig_t config = {0};
    config.gridDim = BPG;
    config.blockDim = TPB; 

    // Set Cluster Dimension (requires CUDA 11.8+ and supported GPU architecture)
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = n_blocks_per_cluster;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    // --- 5. Launch Kernel ---
    printf("Launching kernel...\n");
    
    // NOTE: For timing/profiling, wrap this launch with cudaEvent calls or ncu
    cudaLaunchKernelEx(&config, generic_kernel, In1_d, In2_d, Out_d);

    // Check for launch errors immediately
    CHECK_CUDA_ERROR(cudaGetLastError()); 

    // --- 6. Synchronize and Copy Results ---
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for kernel to finish

    CHECK_CUDA_ERROR(cudaMemcpy(Out_h, Out_d, size_bytes, cudaMemcpyDeviceToHost));
    
    // --- 7. Cleanup ---
    free(In1_h);
    free(In2_h);
    free(Out_h);
    
    CHECK_CUDA_ERROR(cudaFree(In1_d));
    CHECK_CUDA_ERROR(cudaFree(In2_d));
    CHECK_CUDA_ERROR(cudaFree(Out_d));

    printf("Kernel execution complete and data copied back.\n");
    return 0;
}
