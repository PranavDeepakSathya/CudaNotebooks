
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

// --- constants --- 

constexpr int byte_aligner = 128; 
constexpr uint32_t M = 32; 
constexpr uint32_t N = 32; 
constexpr uint32_t BM = 8; 
constexpr uint32_t BN = 8;
constexpr uint32_t GM = M/BM; 
constexpr uint32_t GN = N/BN;
constexpr uint32_t rank = 2; 

static_assert(BM*BN <= 1024, "too many threads");

constexpr uint64_t tensor_shape[rank] = {N,M}; //fast dimension first 
constexpr uint64_t tensor_stride[rank-1] = {N*sizeof(float)}; //number of bytes in the fast dimension is the stride
constexpr uint32_t smem_box_shape[rank] = {BN,BM}; 
constexpr uint32_t element_stride[rank] = {1,1};
constexpr size_t gmem_tensor_size = M*N*sizeof(float);

__global__ void tma_kernel(float* A, const __grid_constant__ CUtensorMap tensor_map)
{
    uint x = blockIdx.x*blockDim.x; 
    uint y = blockIdx.y*blockDim.y;
    __shared__ alignas(byte_aligner) float As[BM][BN];  // declared aligned shared memory 

    #pragma nv_diag_suppress static_var_with_dynamic_init // tells compiler hey chill the fuck out we know whats good.
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, BM*BN);
        cde::fence_proxy_async_shared_cta(); // special proxy that waits and ensures that the TMA engine has visibility to the bar. 
    }
    __syncthreads(); // ensure all thereads are all synced up before using the barrier (bootstrapped init)

    barrier::arrival_token token;
    if (threadIdx.x == 0)
    {
        // Loads the data from Global to Shared. 
        // Swizzling (128B) happens here during the write to SMEM banks.
        cde::cp_async_bulk_tensor_2d_global_to_shared(&As, &tensor_map, x, y, bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(As));
    } 
    else {
        token = bar.arrive();
    }

    bar.wait(std::move(token));
    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    uint smem_col = threadIdx.x; 
    uint smem_row = threadIdx.y; 
    uint gmem_row = y + smem_row;
    uint gmem_col = x + smem_col;
    A[gmem_row*N + gmem_col] = As[smem_row][smem_col];
}

int main()
{
    float* A_h, *A_d; 
    cudaHostAlloc(&A_h, gmem_tensor_size, cudaHostAllocDefault); 
    cudaMalloc(&A_d, gmem_tensor_size); 
    for (int i = 0; i < M*N; i++)
    {
        A_h[i] = (float) i; 
    }
    cudaMemcpy(A_d, A_h, gmem_tensor_size, cudaMemcpyHostToDevice); 
    auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();
    CUtensorMap tensor_map{};
    void *tensor_ptr = (void*)A_d; 
    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 
        rank,
        tensor_ptr, 
        tensor_shape,
        tensor_stride,
        smem_box_shape,
        element_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        printf("Tensor Map Encode Failed!\n");
        return -1;
    }

    dim3 grid(GN, GM); 
    dim3 block(BN, BM);
    tma_kernel<<<grid, block>>>(A_d, tensor_map);
    cudaDeviceSynchronize();
    cudaMemcpy(A_h, A_d, gmem_tensor_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) printf("%d, ", (int) A_h[i]);

}