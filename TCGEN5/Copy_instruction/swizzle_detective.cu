
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h> 

// Modern C++ CUDA headers for TMA
#include <cuda/barrier>
#include <cuda/ptx>

// Namespaces for cleaner code
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// --- Helper Macros ---
#define CUDA_CHECK(call) \
    { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(err); \
        } \
    }

// --- Driver API Helper ---
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    cudaDriverEntryPointQueryResult driver_status;
    void* ptr = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &driver_status));
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
}