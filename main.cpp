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
#include <unistd.h>

#include <cuda_runtime_api.h>
#include <cufft.h>

// cuda macros
#define NX            100         // number of points
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
#define BACKQUEUE     5           // max connections in queue
#define PORTNUM       8080

/******************************************************************************/


// initiate transport layer service server
// bool comms_init() {
//
//   // create and check status of socket
//   int udp_socket = socket(DOMAIN, TYPE, PROTOCOL);
//   if(udp_socket < 0) {
//     printf("Communications error: Failed to create socket.\n");
//     return -1;
//   }
//
//   // setup address
//   struct sockaddr_in server_addr;
//   struct sockaddr_in client_addr;
//   server_addr.sin_family = DOMAIN;
//   server_addr.sin_addr.s_addr = INADDR_ANY;
//   server_addr.sin_port = htons(PORTNUM);          // convert to req. endianness
//
//   // bind
//   if(bind(udp_socket, (struct sockaddr *) &server_addr,
//     sizeof(server_addr) < 0)) {
//
//     printf("Communications error: Failed to bind socket.\n");
//     return -1;
//   }
//
//   // listen
//   listen(udp_socket, BACKQUEUE);
//
//   // accept connections with allocated socket
//   socklen_t client_length = sizeof(client_addr);
//   int new_udp_socket = accept(udp_socket,
//     (struct sockaddr *) &client_addr, &client_length);
//   if(new_udp_socket < 0) {
//     printf("Communications error: Failed to accept connection.\n");
//     return -1;
//   }
//
//   // send
//   send(new_udp_socket, "Hello, world!\n", 13, 0);
//
//   return 1;
// }


// sends calculation from PC to remote raspberry pi
bool send_dev_rpi(char * message) {

  // create client socket
  int client_socket;
  struct sockaddr_in server_addr;
  struct hostent *server;
  client_socket = socket(DOMAIN, TYPE, PROTOCOL);
  if(client_socket < 0) {
    printf("Communications error: Failed to create socket.\n");
    return -1;
  }

  server_addr.sin_family = DOMAIN;
  server_addr.sin_port = htons(PORTNUM);
  server_addr.sin_addr.s_addr = INADDR_ANY;

  // create packet and send
  sendto(client_socket, message, strlen(message), MSG_CONFIRM,
        (const struct sockaddr *) &server_addr, sizeof(server_addr));

  // clean up
  close(client_socket);
  return 1;
}

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

// perform one dimensional fft
// takes data location, number of elements in dimesions of in and out data
void fft1d() {

  // create plan for performing fft
  cufftHandle plan;
  if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
    printf("Failed to create 1D plan\n");
    return;
  }

  // assemble data
  double temp_data[] = {2.598076211353316, 3.2402830701637395, 3.8494572900049224, 4.419388724529261, 4.944267282795252, 5.41874215947433, 5.837976382931011, 6.197696125093141, 6.494234270429254, 6.724567799874842, 6.886348608602047, 6.97792744346504, 6.998370716093996, 6.9474700202387565, 6.8257442563389, 6.6344343416615565, 6.37549055993378, 6.051552679431957, 5.665923042211819, 5.222532898817316, 4.725902331664744, 4.181094175657916, 3.5936624057845576, 2.9695955178498603, 2.315255479544737, 1.6373128742041732, 0.9426788984240022, 0.23843490677753865, -0.46823977812093664, -1.1701410542749289, -1.8601134815746807, -2.531123226988873, -3.176329770049035, -3.7891556376344524, -4.363353457155562, -4.893069644570959, -5.3729040779788875, -5.797965148448726, -6.163919626883915, -6.467036838555256, -6.704226694973039, -6.873071195387157, -6.971849076777267, -6.999553361041935, -6.955901620504255, -6.84133885708361, -6.657032965782207, -6.404862828733319, -6.0873991611848375, -5.707878304681281, -5.270169234606201, -4.778734118422206, -4.23858282669252, -3.6552218606153755, -3.0345982167228436, -2.383038761007964, -1.707185730522749, -1.0139290199674, -0.31033594356630245, 0.39642081173600463, 1.0991363072871054, 1.7906468025248339, 2.463902784786862, 3.1120408346390414, 3.728453594100783, 4.306857124485735, 4.841354967187034, 5.326498254347925, 5.757341256627454, 6.129491801786784, 6.439156050110601, 6.683177170206378, 6.859067520906216, 6.965034011197066, 6.999996379650895, 6.963598207007518, 6.85621054964381, 6.678928156888352, 6.433558310743566, 6.122602401787424, 5.749230429076629, 5.317248684008804, 4.831060947586139, 4.295623596650021, 3.7163950767501706, 3.0992802567403803, 2.4505702323708074, 1.7768781925409076, 1.0850720020162676, 0.3822041878858906, -0.3245599564963766, -1.0280154171511335, -1.7209909100394047, -2.3964219877733033, -3.0474230571943477, -3.667357573646071, -4.249905696354359, -4.78912871521179, -5.279529592175676, -5.716109000098287};
  cufftReal *idata;
  cudaMalloc((void**) &idata, sizeof(cufftComplex)*NX);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to allocate memory space for input data.\n");
    return;
  }

  cudaMemcpy(idata, temp_data, sizeof(temp_data)/sizeof(double), cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to load time data to memory.\n");
    return;
  }

  // prepare memory for return data
  cufftComplex *odata;
  cudaMalloc((void**) &odata, sizeof(cufftComplex)*(NX/2 + 1));
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to allocate memory for output data.\n");
  }

  // perform fft
  if (cufftExecR2C(plan, idata, odata) != CUFFT_SUCCESS) {
    printf("Failed to perform fft.\n");
    return;
  }

  // grab data from graphics and print (memcpy waits until complete) cuda memcopy doesn't complete
  // can return errors from previous cuda calls if they haven't been caught
  cufftComplex *out_temp_data;
  size_t num_bytes = (NX/2 + 1)*sizeof(cufftComplex);
  out_temp_data = new cufftComplex[NX/2 + 1];
  cudaMemcpy(out_temp_data, odata, num_bytes, cudaMemcpyDeviceToHost);
  int error_value = cudaGetLastError();
  printf("cudaMemcpy from device state: %i\n", error_value);
  if(error_value != cudaSuccess) {
    printf("Failed to pull data from device.\n");
    return;
  }

  for (size_t i = 0; i < (NX/2 + 1); i++) {
    printf("%lu %f %f\n", i, out_temp_data[i].x, out_temp_data[i].y);
  }

  // clean up
  delete(out_temp_data);
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

  char * message = (char *)"C++ default message";
  send_dev_rpi(message);

  // comms_init();

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
  fft1d();
  return 0;
}
