#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

// cuda.h is for the Driver API (cu...)
#include <cuda.h>
// cuda_runtime.h is for the Runtime API (cuda...)
#include <cuda_runtime.h>

// --- A Robust Error-Checking Macro ---
// This will check both Runtime and Driver API calls
// and print the error message if something fails.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t runtimeErr = (call); \
        if (runtimeErr != cudaSuccess) { \
            fprintf(stderr, "CUDA Runtime Error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(runtimeErr)); \
            exit(EXIT_FAILURE); \
        } \
        CUresult driverErr = (CUresult)runtimeErr; \
        if (driverErr != CUDA_SUCCESS) { \
            const char* errName; \
            cuGetErrorName(driverErr, &errName); \
            const char* errString; \
            cuGetErrorString(driverErr, &errString); \
            fprintf(stderr, "CUDA Driver Error at %s:%d: %s (%s)\n", \
                    __FILE__, __LINE__, errString, errName); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


int main() {
    // === 1. Setup Constants ===
    const int N = 8; // Matrix dimension (8x8)
    const int num_elements = N * N;
    const size_t bytes = num_elements * sizeof(float);
    const std::string ptx_filename = "skeleton.ptx"; // Make sure this matches your file

    // --- Host Memory ---
    std::vector<float> h_A(num_elements);
    std::vector<float> h_B(num_elements);
    std::vector<float> h_C(num_elements);
    unsigned long long h_Clock = 0; // To store the result

    // --- Device Memory Pointers ---
    // Note: Driver API uses CUdeviceptr, which is just unsigned long long
    CUdeviceptr d_A, d_B, d_C, d_Clock;

    // === 2. Initialize CUDA & Load PTX Module ===
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    CUDA_CHECK( (CUresult)cuInit(0) );
    CUDA_CHECK( (CUresult)cuDeviceGet(&device, 0) );
    CUDA_CHECK( (CUresult)cuCtxCreate(&context, 0, device) );

    std::cout << "Loading PTX file: " << ptx_filename << std::endl;
    CUDA_CHECK( (CUresult)cuModuleLoad(&module, ptx_filename.c_str()) );
    CUDA_CHECK( (CUresult)cuModuleGetFunction(&kernel, module, "benchmark") );

    // === 3. Allocate and Initialize Memory ===
    
    // Initialize host data (e.g., A=1.0, B=2.0)
    for (int i = 0; i < num_elements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate on device
    CUDA_CHECK( (CUresult)cuMemAlloc(&d_A, bytes) );
    CUDA_CHECK( (CUresult)cuMemAlloc(&d_B, bytes) );
    CUDA_CHECK( (CUresult)cuMemAlloc(&d_C, bytes) );
    CUDA_CHECK( (CUresult)cuMemAlloc(&d_Clock, sizeof(unsigned long long)) );

    // Copy from host to device
    std::cout << "Copying data to GPU..." << std::endl;
    CUDA_CHECK( (CUresult)cuMemcpyHtoD(d_A, h_A.data(), bytes) );
    CUDA_CHECK( (CUresult)cuMemcpyHtoD(d_B, h_B.data(), bytes) );

    // === 4. Setup Kernel Launch ===

    // Kernel parameters must be passed as an array of POINTERS to the data.
    void* args[] = {
        &d_A,
        &d_B,
        &d_C,
        &d_Clock
    };

    // Grid: <<<1, 1>>>
    unsigned int gridDimX = 1;
    unsigned int gridDimY = 1;
    unsigned int gridDimZ = 1;

    // Block: <<<1, 1>>>
    unsigned int blockDimX = 1;
    unsigned int blockDimY = 1;
    unsigned int blockDimZ = 1;

    unsigned int sharedMemBytes = 0;
    CUstream stream = 0;

    // === 5. Launch Kernel and Synchronize ===
    std::cout << "Launching kernel..." << std::endl;
    CUDA_CHECK( (CUresult)cuLaunchKernel(
        kernel,
        gridDimX, gridDimY, gridDimZ,
        blockDimX, blockDimY, blockDimZ,
        sharedMemBytes,
        stream,
        args,  // Array of pointers to args
        NULL   // Extra options
    ));

    std::cout << "Waiting for kernel to finish..." << std::endl;
    CUDA_CHECK( (CUresult)cuCtxSynchronize() );
    std::cout << "Kernel finished." << std::endl;

    // === 6. Copy Results Back and Report ===
    CUDA_CHECK( (CUresult)cuMemcpyDtoH(h_C.data(), d_C, bytes) );
    CUDA_CHECK( (CUresult)cuMemcpyDtoH(&h_Clock, d_Clock, sizeof(unsigned long long)) );

    // --- Report Results ---
    // Get the iteration count from your PTX to calculate clocks/op
    // I'll assume the 100,000 from the example
    const long long max_iters = 100000;
    const long long fmas_per_iter = N * N * N; // 8*8*8 = 512
    const long long total_fmas = fmas_per_iter * max_iters;
    
    double clocks_per_fma = static_cast<double>(h_Clock) / total_fmas;

    std::cout << "--------------------------------" << std::endl;
    std::cout << "--- Benchmark Results ---" << std::endl;
    std::cout << "Total Clocks:    " << h_Clock << std::endl;
    std::cout << "Total FMAs:      " << total_fmas << std::endl;
    std::cout << "Clocks / FMA:    " << clocks_per_fma << std::endl;
    std::cout << "--------------------------------" << std::endl;

    // Optional: Check a result to be sure
    // For A=1.0, B=2.0, C[0][0] = k*A[0][k]*B[k][0] = 8 * (1.0 * 2.0) = 16.0
    // (This check depends on your B matrix layout)
    // std::cout << "Test C[0] value: " << h_C[0] << std::endl;


    // === 7. Cleanup ===
    std::cout << "Cleaning up..." << std::endl;
    CUDA_CHECK( (CUresult)cuMemFree(d_A) );
    CUDA_CHECK( (CUresult)cuMemFree(d_B) );
    CUDA_CHECK( (CUresult)cuMemFree(d_C) );
    CUDA_CHECK( (CUresult)cuMemFree(d_Clock) );
    CUDA_CHECK( (CUresult)cuModuleUnload(module) );
    CUDA_CHECK( (CUresult)cuCtxDestroy(context) );

    return 0;
}