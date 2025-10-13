#include <cuda_runtime.h>
#include <stdio.h> 
#include <stdlib.h> 
#include <cooperative_groups.h>
// Alternatively use an alias to avoid polluting the namespace with collective algorithms
namespace cg = cooperative_groups;

// --- Configuration Constants ---
// Define the dimensions for a simple 1D buffer operation
constexpr int DATA_SIZE = 1024 * 1024; 

// Kernel Launch Configuration (Modern Cluster Launch)
// We will let the API determine the optimal cluster size.
constexpr int n_blocks_per_grid = 256;         // Total blocks in the grid
// Note: n_blocks_per_cluster will now be calculated in main()
constexpr int n_threads_per_block = 1024;      // Threads per block
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

// =======================================================
// KERNEL DEFINITION
// =======================================================
__global__ void generic_kernel(const float *In1, const float *In2, float *Out)
{
    
    auto grid = cg::this_grid(); 
    // get the rank of the calling cluster in range [0, total_deployed_clusters)
    uint cluster_rank = grid.cluster_rank();
    //get the rank of the calling block in range [0, total_deployed_blocks) 
    uint block_rank = grid.block_rank(); 
    //get the rank of the calling thread in range [0, total_deployed_threads) 
    uint thread_rank = grid.thread_rank(); 
    
    // Only the first few threads should print to avoid flooding the console
    if (cluster_rank == 0 && block_rank == 0 && thread_rank < 32)
    {
        printf("cluster: %d, block: %d, thread: %d, \n", cluster_rank, block_rank, thread_rank);
    }
    
    // Core GPU logic remains unchanged (Vector Add Example)
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < DATA_SIZE)
    {
        Out[tid] = In1[tid] + In2[tid];
    }


    grid.sync();
}


int main ()
{
    // Determine optimal cluster size based on kernel resources
    int optimal_blocks_per_cluster = 0;
    int SM_count;
    
    // Get the SM count (can be used to size the grid appropriately later)
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&SM_count, cudaDevAttrMultiProcessorCount, 0));

    // Calculate the optimal cluster size for this specific kernel and block dimension.
    // The function returns the largest number of blocks that can be executed concurrently
    // on a single SM while maximizing occupancy.
    CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialClusterSize(
        &optimal_blocks_per_cluster,
        &SM_count, // Output: The number of active multiprocessors (SMs)
        (void*)generic_kernel, 
        n_threads_per_block, 
        0, // Shared memory per block (0 in this case)
        0 // Flags
    ));
    
    // Use the calculated optimal size, ensuring it doesn't exceed the total grid size
    // and is a factor of the total grid size.
    int final_blocks_per_cluster = optimal_blocks_per_cluster;
    
    // Ensure the grid dimension is divisible by the chosen cluster size.
    if (n_blocks_per_grid % final_blocks_per_cluster != 0) {
        // If not divisible, fall back to a safe power-of-2 factor that divides the grid.
        final_blocks_per_cluster = 1; // Default to 1 (no clustering) if optimization fails.
        // NOTE: In a real application, you would find the nearest divisor or adjust n_blocks_per_grid.
    }
    
    printf("Calculated Optimal Cluster Size (Blocks/Cluster): %d\n", final_blocks_per_cluster);
    printf("Total Grid Blocks: %d\n", n_blocks_per_grid);


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

    // Set Cluster Dimension using the calculated optimal size
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = final_blocks_per_cluster;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    // --- 5. Launch Kernel ---
    printf("Launching kernel with %d blocks per cluster...\n", final_blocks_per_cluster);
    
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
