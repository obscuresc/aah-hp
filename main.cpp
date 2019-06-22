/*******************************************************************************
Automatic Alignment of Heliostats project

This program extracts heliostat reflection images (HRIs) from a live video feed
of a solar power tower (SPT) central receiver (CR) by using examining the
magnitude of frequency vibration intensity of component pixel vibrations
obtained using a fourier transform of the change of pixel intensity in time.

This project is maintained on github.

https://github.com/obscuresc/aah-hp

Jack Arney 22-06-19
*******************************************************************************/


#include <iostream>
#include <fstream>
// #include <cufft.h>

#define NX 256
#define BATCH 1

int main() {


  return 0;
}

/*
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

  // Use the CUFFT plan to transform the signal in place.
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

*/
