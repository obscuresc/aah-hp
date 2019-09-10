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

Jack Arney 08-09-19
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

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include "video_param.h"

#include <cuda_runtime_api.h>
#include <cufft.h>
#include "cuda.cu"

/******************************************************************************/

// cuda macros
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

// heliostat deviation parameters @pixelval or physical
struct heliostat_dev_t {

  float x;
  float y;
};

/******************************************************************************/

// sends calculation from PC to remote raspberry pi
bool send_dev_rpi(heliostat_dev_t * heliostat_dev) {

  // create client socket
  int client_socket;
  struct sockaddr_in server_addr;
  client_socket = socket(DOMAIN, TYPE, PROTOCOL);
  if(client_socket < 0) {
    printf("Communications error: Failed to create socket.\n");
    return -1;
  }

  server_addr.sin_family = DOMAIN;
  server_addr.sin_port = htons(PORTNUM);
  server_addr.sin_addr.s_addr = INADDR_ANY;

  // create packet and send
  const char * message = "@message";
  sendto(client_socket, message, strlen(message), MSG_CONFIRM,
        (const struct sockaddr *) &server_addr, sizeof(server_addr));

  // clean up
  close(client_socket);
  return 0;
}

/******************************************************************************/

bool load_video(std::string video_source, cv::cuda::GpuMat * d_mat,
  video_param_t * video_param) {

  // grab video
  cv::VideoCapture video_sample(video_source);
  if (!video_sample.isOpened()) {

    printf("Could not grab %s\n", video_source.c_str());
    return -1;
  }

  // allocate gpu memory
  cudaMalloc((void**) &d_mat, sizeof(cv::Mat) * video_param->n_frames);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to allocate memory space for video data.\n");
    return -1;
  }

  // load to gpu
  size_t d_mat_size = sizeof(cufftReal) * video_param->n_frames;
  cudaMemcpy(d_mat, (const void *) video_sample, d_mat_size, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to load video data to memory.\n");
    return -1;
  }

  // get parameters
  video_param.height = (int) video_sample.get(CAP_PROP_FRAME_HEIGHT);
  video_param.height = video_sample.get(CAP_PROP_FRAME_WIDTH);
  video_param.n_frames = video_sample.get(CAP_PROP_FRAME_COUNT);

  return 0;
}

/******************************************************************************/

int main(int argc, char** argv) {

  // setup processing parameters
  // int heliostat_freq[] = {1, 2, 3};
  if (argc == 0) {
    printf("Using default video.\n");
  }

  std::string video_source = "vib.avi";
  video_param_t video_param;

  unsigned char * d_mat;

  cufftReal * d_raw;
  // size_t n_threads = 2432;
  int n_pixel_vals;
  cufftComplex * d_ftd;
  // cufftReal * d_mag;

  // cv::Mat images[sizeof(heliostat_freq)/sizeof(int)];

  while(true) {

    // @will need to update depending on rolling buffer or large batch type
    // rolling will not be in-place

    if (load_video(video_source, d_mat, &video_param) != 0) {

      printf("Could not load video data.\n");
      return -1;
    }

    n_pixel_vals = video_param.height *
                      video_param.width *
                      video_param.n_frames;

    // prepare for fft @depend on config of gpu
    // cufftReal_convert<<<n_blocks, n_threads>>>(d_mat, d_raw);

    // perform fft on individual pixel streams and adjust to real values
    cudaMalloc((void**) &d_raw, sizeof(cufftComplex) * n_pixel_vals);
    if (cudaGetLastError() != cudaSuccess) {
      printf("Failed to allocate memory space for video data.\n");
      return -1;
    }

    if (fft_batched(d_raw, video_param, d_ftd)) {

      printf("Could not perform fft.\n");
      return -1;
    }
    // mag_adjust(d_ftd, d_mag);

    // grab bin values for specific frequencies across entire stream
    // @put into cuda func
    // image_collate(heliostat_freq, images);

    // perform centroid calculations (in place)
    heliostat_dev_t heliostat_dev[3];
    // centroid_calc(images, heliostat_dev);

    send_dev_rpi(heliostat_dev);

  }

  // float * fft_magnitude = fft1d();

  // clean up
  // delete(fft_magnitude);

  return 0;
}
