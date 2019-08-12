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
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <cufft.h>

// cuda macros
#define NX            256         // number of points
#define BATCH         1           // number of ffts to perform
#define RANK          1           //
#define IDIST         1           // distance between 1st elements of batches
#define ISTRIDE       1           // do every ISTRIDEth index
#define ODIST         1           // distance between 1st elements of output
#define OSTRIDE       1           // distance between output elements

// socket macros
#define DOMAIN        AF_INET     // ipv4 (AF_INET) or ipv6 (AF_INET6)
#define PROTOCOL      0           // default
#define TYPE          SOCK_DGRAM  // udp (SOCK_DGRAM) or tcp (SOCK_STREAM)

// server settings
#define BACKQUEUE     5           // max queue of connections
#define PORTNUM       8080

// client settings



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

// initiate transport layer service
bool comms_init() {

  // create and check status of socket
  int udp_socket = socket(DOMAIN, TYPE, PROTOCOL);
  if(udp_socket < 0) {
    printf("Communications error: Failed to create socket.\n");
    return -1;
  }

  // setup address
  struct sockaddr_in server_addr;
  struct sockaddr_in client_addr;
  server_addr.sin_family = DOMAIN;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port = htons(PORTNUM);          // convert to req. endianness

  // bind
  if(bind(udp_socket, (struct sockaddr *) &server_addr,
    sizeof(server_addr) < 0)) {

    printf("Communications error: Failed to bind socket.\n");
    return -1;
  }

  // listen
  listen(udp_socket, BACKQUEUE);

  // accept connections with allocated socket
  socklen_t client_length = sizeof(client_addr);
  int new_udp_socket = accept(udp_socket,
    (struct sockaddr *) &client_addr, &client_length);
  if(new_udp_socket < 0) {
    printf("Communications error: Failed to accept connection.\n");
    return -1;
  }

  // send
  send(new_udp_socket, "Hello, world!\n", 13, 0);

  return 1;
}


// perform one dimensional fft
// takes data location, number of elements in dimesions of in and out data
void fft1d(int inembed, int oembed) {

  // create plan for performing fft
  cufftHandle plan;
  if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
    printf("Failed to create 1D plan\n");
    return;
  }

  // assemble data
  cufftReal *idata;
  cufftComplex *odata;
  cudaMalloc((void**) &idata, sizeof(cufftComplex)*NX*BATCH);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to load time data to memory\n");
    return;
  }

  // perform fft in place*/
  if (cufftExecR2C(plan, idata, odata) != CUFFT_SUCCESS) {
    printf("Failed to perform perform forward fft.\n");
  }

  // blocks until fft complete
  if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("Failed to synchronise.\n");
  }

  // clean up
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

  comms_init();

  // FILE* stream = fopen("time_data.txt", "r");
  //
  // char line[1024];
  // while(fgets(line, 1024, stream)) {
  //
  //   char * tmp = strdup(line);
  //   printf("Field 3 would be %s\n", getfield(tmp, 3));
  //   free(tmp);
  // }
  //
  // fft1d(3, 4);
  return 0;
}
