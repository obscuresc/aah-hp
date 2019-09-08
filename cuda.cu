/*******************************************************************************
cuda functions for backend of fourier transforms
*******************************************************************************/

__device__ void cufftReal_convert(cv::Mat * d_mat, cufftReal * d_raw) {

  d_raw[blockIdx.x] = (cufftReal) d_mat->at(blockIdx.x);
}
