/*******************************************************************************
Automatic Alignment of Heliostats project

This program extracts heliostat reflection images (HRIs) from a live video feed
of a solar power tower (SPT) central receiver (CR) by using examining the
magnitude of frequency vibration intensity of component pixel vibrations
obtained using a fourier transform of the change of pixel intensity in time.

Project aims at using nVidia 1070Ti (Pascal architecture, GP104) graphics card
to optimise FFT processing. Assumes data lives in a single block of memory but
can be changed in by cuda macros.

This project is maintained on github.

https://github.com/obscuresc/aah-hp

Jack Arney 22-06-19
*******************************************************************************/


#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <cufft.h>

/* cuda macros */
#define NX            256       /* number of points */
#define BATCH         1         /* number of ffts to perform */
#define RANK          1         /*  */
#define IDIST         1         /* distance between 1st elements of batches */
#define ISTRIDE       1         /* do every ISTRIDEth index */
#define ODIST         1         /* distance between 1st elements of output */
#define OSTRIDE       1         /* distance between output elements */


/******************************************************************************/

/*
steps
  create plan
  assemble data (R, C etc)
  assign plan
    decide if in place or out of place
    streamed
    batches
    strides
  set call backs
  execute
  clean up
*/

/* perform one dimensional fft */
/* takes data location, number of elements in dimesions of in and out data */
void fft1d(int inembed, int oembed) {

  /* create plan for performing fft */
  cufftHandle plan;
  if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
    printf("Failed to create 1D plan\n");
    return;
  }

  /* assemble data */
  cufftReal *idata;
  cufftComplex *odata;
  cudaMalloc((void**) &idata, sizeof(cufftComplex)*NX*BATCH);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to load time data to memory\n");
    return;
  }

  /* perform fft in place*/
  if (cufftExecR2C(plan, idata, odata) != CUFFT_SUCCESS) {
    printf("Failed to perform perform forward fft.\n");
  }

  /* blocks until fft complete */
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("Failed to synchronise.\n");
  }

  /* clean up */
  cufftDestroy(plan);
  cudaFree(idata);

}

const char* getfield(char* line, int num) {

  const char* tok;
  for(tok = strtok(line, ";"); tok&& *tok; tok = strtok(NULL, ";\n")) {

    if(!--num) {
      return tok;
    }
  }

  return NULL;
}

int main() {

  FILE* stream = fopen("time_data.txt", "r");

  char line[1024];
  while(fgets(line, 1024, stream)) {

    char * tmp = strdup(line);
    printf("Field 3 would be %s\n", getfield(tmp, 3));
    free(tmp);
  }

  fft1d(3, 4);
  return 0;
}
