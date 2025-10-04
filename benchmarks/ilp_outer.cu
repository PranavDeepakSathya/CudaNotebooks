//#okay lets write some simple shit
#include <stdio.h> 
#include<stdlib.h> 
#include<cuda.h> 
#include<cuda_runtime.h>

constexpr int N = 4; 
constexpr int reps = 1;


__global__ void inner_k(float *A, float *B, float *C) {

  float A_reg[N*N];
  float B_reg[N*N];
  float C_reg[N*N] = {0.0f};
  uint t = threadIdx.x; 

  for (int i = 0; i < N*N; i++) {
      A_reg[i] = A[t + i];
      B_reg[i] = B[t + i];
  }
  __syncthreads();
  for (int repeat = 0; repeat < reps; repeat++) {
      for (int k = 0; k < N; k++) {
          for (int i = 0; i < N; i++) {
              for (int j = 0; j < N; j++) {
                  C_reg[i*N + j] += A_reg[i*N + k] * B_reg[k*N + j];
              }
          }
      }
  }
  __syncthreads();
  
  // Write result back
  for (int i = 0; i < N*N; i++) {
      C[i] = C_reg[t + i];
  }
}


int main() {
    //# Host arrays
    float A[N*N], B[N*N], C[N*N];

    //# Init A and B
    for (int i = 0; i < N*N; i++) {
        A[i] = 0.01f;
        B[i] = 0.02f;
    }

    // #Device arrays
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    //# Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    inner_k<<<1,1>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel time: %f ms\n", ms);

    // # Copy back result
    cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    // #Print output matrix
    printf("C =\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", C[i*N + j]);
        }
        printf("\n");
    }

    // #Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}