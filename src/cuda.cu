/*******************************************************************************
                        cuda functions for gpu processing
*******************************************************************************/

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_runtime_api.h>
#include <cufft.h>

#include "video_param.h"

/******************************************************************************/

// __device__ void cufftReal_convert(cv::Mat * d_mat, cufftReal * d_raw) {
//
//   d_raw[blockIdx.x] = (cufftReal) d_mat->at<double>(blockIdx.x);
// }
  

bool fft_batched(cufftReal * d_raw, video_param_t video_param, cufftComplex * d_ftd) {

  // create plan for performing fft
  cufftHandle plan;
  size_t batch = video_param.height * video_param.width;
  size_t n_points = video_param.n_frames;
  if (cufftPlan1d(&plan, n_points, CUFFT_R2C, batch) != CUFFT_SUCCESS) {
    printf("Failed to create 1D plan\n");
    return -1;
  }

  // allocate return data
  cudaMalloc((void**) &d_ftd, sizeof(cufftComplex)*n_points * batch);
  if (cudaGetLastError() != cudaSuccess) {
    printf("Failed to allocate memory space for transformed data.\n");
    return -1;
  }

  // perform fft
  if (cufftExecR2C(plan, d_raw, d_ftd) != CUFFT_SUCCESS) {
    printf("Failed to perform fft.\n");
    return -1;
  }

  cufftDestroy(plan);
  cudaFree(d_raw);

  return 0;
}

/******************************************************************************/
