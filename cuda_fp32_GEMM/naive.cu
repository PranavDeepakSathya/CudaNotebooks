

#include<cuda_runtime.h>
#include<cuda.h> 
#include<stdio.h> 
#include<stdlib.h> 
#include<time.h> // Added for random init and CPU timing
#include<math.h> // Added for fabs in comparison

// Error checking macro
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d in %s: %s\n",
                file, line, func, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


constexpr int n_blocks_per_cluster = 1; //#keep this fixed 
constexpr int n_threads_per_block = 1024; //#keep this fixed

constexpr int M = 4096; 
constexpr int N = 4096; 
constexpr int K = 4096; 
constexpr int n_blocks_per_grid = (M*N)/(n_threads_per_block); 

// Host-side CPU Matrix Multiplication for verification (C = A * B)
void matmul_cpu(const float *A, const float *B, float *C_cpu, int m, int n, int k)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < k; l++)
            {
                // A is M x K (A[i*k + l]), B is K x N (B[l*n + j])
                sum += A[i * k + l] * B[l * n + j];
            }
            C_cpu[i * n + j] = sum; // C is M x N
        }
    }
}

// Host-side Result Comparison
void check_result(const float *C_h, const float *C_cpu, size_t size)
{
    int errors = 0;
    const float tolerance = 1e-3f;
    for (size_t i = 0; i < size; i++)
    {
        if (fabs(C_h[i] - C_cpu[i]) > tolerance)
        {
            if (errors < 5) // Print only the first few errors
            {
                fprintf(stderr, "Mismatch at index %zu: GPU=%f, CPU=%f\n", i, C_h[i], C_cpu[i]);
            }
            errors++;
        }
    }
    
    if (errors > 0)
    {
        fprintf(stderr, "\nVerification FAILED! Total errors: %d\n", errors);
    }
    else
    {
        printf("\nVerification PASSED!\n");
    }
}


__global__ void kernel(float *A, float*B, float*C)
{
    uint idx = (threadIdx.x*blockDim.x) + (blockIdx.x*gridDim.x);
    uint I = idx / N; 
    uint J = idx % N; 

    for (int k = 0; k < K; k++)
    { 
      if (I < M && J < N)
        C[I*N + J] += A[I*K + k] + B[k*N + J];
    }

}


int main ()
{

  size_t size_A = M*K*sizeof(float); 
  size_t size_B = K*N*sizeof(float); 
  size_t size_C = M*N*sizeof(float); 

  float *A_h = (float*) malloc(size_A);
  float *B_h = (float*) malloc(size_B); 
  float *C_h = (float*) malloc(size_C); 
  float *C_cpu = (float*) malloc(size_C); 
  
  if (!A_h || !B_h || !C_h || !C_cpu)
  {
      fprintf(stderr, "Host memory allocation failed!\n");
      exit(EXIT_FAILURE);
  }

  // Random Initialization
  srand(time(NULL)); 
  for (int i = 0; i < M*K; i++) // Init A (M x K)
  {
    A_h[i] = (float)rand() / ((float)RAND_MAX*10.0f); 
  }
  for (int i = 0; i < K*N; i++) // Init B (K x N)
  {
    B_h[i] = (float)rand() / ((float)RAND_MAX*10.0f); 
  }
  for (int i = 0; i < M*N; i++) // Init C and C_cpu (M x N)
  {
    C_h[i] = 0.0f; 
    C_cpu[i] = 0.0f;
  }
  
  float* A_d, *B_d, *C_d; 
  CHECK_CUDA_ERROR(cudaMalloc(&A_d, size_A));
  CHECK_CUDA_ERROR(cudaMalloc(&B_d, size_B)); 
  CHECK_CUDA_ERROR(cudaMalloc(&C_d, size_C)); 

  CHECK_CUDA_ERROR(cudaMemcpy(A_d, A_h, size_A, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(B_d, B_h, size_B, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(C_d, C_h, size_C, cudaMemcpyHostToDevice));
  
  dim3 TPB(n_threads_per_block,1,1);
  dim3 BPG(n_blocks_per_grid,1,1);
  
  cudaLaunchConfig_t config = {0};
  config.gridDim = BPG;
  config.blockDim = TPB; 
  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = n_blocks_per_cluster; //# Cluster size in X-dimension
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;
  config.attrs = attribute;
  config.numAttrs = 1;

  // CUDA Event Timer Setup
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));

 
  CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

  cudaLaunchKernelEx(&config, kernel, A_d, B_d, C_d);


  CHECK_CUDA_ERROR(cudaGetLastError()); 

  CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("GPU Kernel time: %f ms\n", milliseconds);

  CHECK_CUDA_ERROR(cudaMemcpy(C_h, C_d, size_C, cudaMemcpyDeviceToHost));

  printf("Running CPU verification...\n");
  clock_t cpu_start = clock();
  matmul_cpu(A_h, B_h, C_cpu, M, N, K);
  clock_t cpu_stop = clock();
  double cpu_time = (double)(cpu_stop - cpu_start) * 1000.0 / CLOCKS_PER_SEC;
  printf("CPU Matmul time: %f ms\n", cpu_time);

  check_result(C_h, C_cpu, M*N);


  free(A_h);
  free(B_h);
  free(C_h);
  free(C_cpu);
  
  CHECK_CUDA_ERROR(cudaFree(A_d));
  CHECK_CUDA_ERROR(cudaFree(B_d));
  CHECK_CUDA_ERROR(cudaFree(C_d));
  
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));

  return 0;
}