#include <iostream>
#include <fstream>

#define NX 256
#define BATCH 1

void main() {






}

void fft1d() {

  cufftHandle plan;
  cufftComplex *data;
  cudaMalloc((void**)&data, sizeof(cufftComplex)*(NX/2+1)*BATCH);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return;
  }

  if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return;
  }

  ...

  /* Use the CUFFT plan to transform the signal in place. */
  if (cufftExecR2C(plan, (cufftReal*)data, data) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return;
  }

  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return;
  }

  cufftDestroy(plan);
  cudaFree(data);

}
