#include <stdio.h>
#include <cuda_runtime.h>

constexpr int M = 8192; 
constexpr int N = 8192; 
constexpr int K = 8192; 
constexpr int BS = 32; 
constexpr int BM = 128; 
constexpr int BN = BM; 
constexpr int TM = BM/BS; 
constexpr int TN = BN/BS; 
constexpr int tpb_M = BM/TM; 
constexpr int tpb_N = BN/TN; 
constexpr int BK = (4*BS*BS)/BM;

__global__ void matmul (float*A, float*B, float*C)
{

   __shared__ float Asub[BM*BK]; 
   __shared__ float Bsub[BK*BN];
   uint cb_row = blockIdx.x; 
   uint cb_col = blockIdx.y; 
   uint t = threadIdx.x; 
   C += (cb_row*BM)*N + (cb_col*BN);
   A += (cb_row*BM)*K;
   B += (cb_col*BN);
   uint c_inner_row = t / tpb_N; 
   uint c_inner_col = t % tpb_N; 
   //(interpret BS*BS as (BM)*(BK/4)) for loading A int Asub
   //(interpret BS*BS as (BK)*(BN/4)) for B to tmpvec 
   //(interpret BS*BS as (BN)*(BK/4)) for tempvec to Bsub 
   uint al_row = t/(BK/4); 
   uint al_col = t % (BK/4);
   uint bl_row = t/(BN/4); 
   uint bl_col = t % (BK/4); 
   uint bs_row = t/(BK/4); 
   uint bs_col = t % (BK/4); 
   float regM[TM] = {0.0};
   float regN[TN] = {0.0};
   float t_results[TM*TN] = {0.0}; 

   for (uint bk = 0; bk < K; bk += BK)
   {
      if (al_row < BM && (4*al_col) + 3 < BK)
      {
         reinterpret_cast<float4 *>(&Asub[al_row*BK + (4*al_col)])[0]
         = reinterpret_cast<float4 *>(&A[al_row*K + (4*al_col)])[0]; 
      }
      if (bl_row < BK && (4*bl_col)+3 < BN && bs_row < BN && (4*bs_col) + 3 < BK)
      {
         float4 v = reinterpret_cast<float4 *>(&B[bl_row*N + (4*bl_col)])[0];
         Bsub[(bs_row)*BK + (4*bs_col) + 0] = v.x; 
         Bsub[(bs_row)*BK + (4*bs_col) + 1] = v.y; 
         Bsub[(bs_row)*BK + (4*bs_col) + 2] = v.z; 
         Bsub[(bs_row)*BK + (4*bs_col) + 3] = v.w; 
      }
      __syncthreads();
      A += BK;
      B += N*BK;
      for (uint dotIdx = 0; dotIdx < BK; dotIdx++)
      {
         for (uint i = 0; i < TM; i++)
         {
            regM[i] = Asub[(al_row)*BK + dotIdx];
         }
         for (uint j = 0; j < TN; j++)
         {
            regN[j] = Bsub[(bs_row)*BK + dotIdx];
         }
         for (uint i = 0; i < TM; i++)
         {
            for (uint j = 0; j < TN; j++)
            {
               t_results[i*TN + j] += regM[i]*regN[j]; 
            }

         }
         
      }
      __syncthreads();
      
   }

   for (uint i = 0; i < TM; i++)
   {
      for (uint j = 0; j < TN; j++)
      {
         uint c_row = (c_inner_row*TM + i);
         uint c_col = (c_inner_col*TN + j); 
         C[c_row*N + c_col] = t_results[i*TN + j];
      }

   }

}

int main()
{
   float *h_A = (float *)malloc(M * K * sizeof(float));
   float *h_B = (float *)malloc(K * N * sizeof(float));
   float *h_C = (float *)malloc(M * N * sizeof(float));

   for (int i = 0; i < M * K; ++i)
      h_A[i] = 0.22f;
   for (int i = 0; i < K * N; ++i)
      h_B[i] = 0.11f;
   memset(h_C, 0, M * N * sizeof(float));

   float *d_A, *d_B, *d_C;
   cudaMalloc(&d_A, M * K * sizeof(float));
   cudaMalloc(&d_B, K * N * sizeof(float));
   cudaMalloc(&d_C, M * N * sizeof(float));

   cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);

   dim3 threads(BS * BS);
   dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);

   // === CUDA Event Timer ===
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start);

   matmul<<<blocks, threads>>>(d_A, d_B, d_C);

   cudaEventRecord(stop);
   cudaEventSynchronize(stop);

   float ms = 0;
   cudaEventElapsedTime(&ms, start, stop);

   double gflops = (2.0 * M * N * K) / (ms * 1.0e6); // GFLOP/s
   double tflops = gflops / 1000.0;

   printf("Time: %.3f ms\n", ms);
   printf("Throughput: %.3f TFLOP/s\n", tflops);
   // Copy back C matrix
   cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

   int zero_count_C = 0;
   for (int i = 0; i < M * N; ++i)
      if (h_C[i] == 0.0f)
            zero_count_C++;

   printf("Zeros in C: %d / %d\n", zero_count_C, M * N);

   printf("First few C values: ");
   for (int i = (M)*(N)-10; i < M*N; ++i)
      printf("%.1f ", h_C[i]);
   printf("\n");

   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   free(h_A);
   free(h_B);
   free(h_C);
   printf("=== GEMM Kernel Config ===\n");
   printf("M       = %d\n", M);
   printf("N       = %d\n", N);
   printf("K       = %d\n", K);
   printf("BS      = %d\n", BS);
   printf("BM      = %d\n", BM);
   printf("BN      = %d\n", BN);
   printf("TM      = %d\n", TM);
   printf("TN      = %d\n", TN);
   printf("tpb_M   = %d\n", tpb_M);
   printf("tpb_N   = %d\n", tpb_N);
   printf("BK      = %d\n", BK);
   printf("==========================\n\n");


   return 0;
}