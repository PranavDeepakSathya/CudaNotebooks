
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <stdint.h>
#include <time.h>

// --- Driver API error checking macro ---
#define CHECK_CUDA_DRIVER(call) \
    do { \
        CUresult err = call; \
        if (err != CUDA_SUCCESS) { \
            const char* str; \
            cuGetErrorString(err, &str); \
            printf("CUDA Driver error: %s in %s at line %d\n", str, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int N_ctas = 2;
constexpr int N_threads_per_block = 32;
constexpr int N_debug_vals = 4;

int main() {
    uint32_t* h_debug_val;
    size_t size = N_debug_vals * sizeof(uint32_t);

    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_debug_val;

    // --- 1. Initialize CUDA Driver ---
    CHECK_CUDA_DRIVER(cuInit(0));
    CHECK_CUDA_DRIVER(cuDeviceGet(&device, 0));
    CHECK_CUDA_DRIVER(cuCtxCreate(&context, 0, device));

    // --- 2. Load PTX directly from file ---
    CHECK_CUDA_DRIVER(cuModuleLoad(&module, "tcgen5alloc.ptx"));
    CHECK_CUDA_DRIVER(cuModuleGetFunction(&kernel, module, "alloc_dealloc_tmem_ptx"));

    // --- 3. Allocate Host and Device Memory ---
    h_debug_val = (uint32_t*)malloc(size);
    CHECK_CUDA_DRIVER(cuMemAlloc(&d_debug_val, size));

    // --- 4. Initialize Host Data and Copy to Device ---
    srand(time(NULL));
    printf("Host array before kernel (random values):\n");
    for (int i = 0; i < N_debug_vals; i++) {
        h_debug_val[i] = rand();
        printf("h_debug_val[%d] = %u\n", i, h_debug_val[i]);
    }

    CHECK_CUDA_DRIVER(cuMemcpyHtoD(d_debug_val, h_debug_val, size));

    // --- 5. Launch Kernel ---
    void* args[] = { &d_debug_val };

    CHECK_CUDA_DRIVER(cuLaunchKernel(
        kernel,
        N_ctas, 1, 1,
        N_threads_per_block, 1, 1,
        0,
        0,
        args,
        0
    ));

    CHECK_CUDA_DRIVER(cuCtxSynchronize());

    // --- 6. Copy Back and Print Results ---
    CHECK_CUDA_DRIVER(cuMemcpyDtoH(h_debug_val, d_debug_val, size));

    printf("\nKernel executed successfully.\n");
    printf("TMEM addresses from device:\n");
    for (int i = 0; i < N_ctas; i++) {
        printf("CTA %d, Alloc 0 TMEM address: %u\n", i, h_debug_val[2*i + 0]);
        printf("CTA %d, Alloc 1 TMEM address: %u\n", i, h_debug_val[2*i + 1]);
    }

    // --- 7. Cleanup ---
    CHECK_CUDA_DRIVER(cuMemFree(d_debug_val));
    CHECK_CUDA_DRIVER(cuModuleUnload(module));
    CHECK_CUDA_DRIVER(cuCtxDestroy(context));
    free(h_debug_val);

    return 0;
}
