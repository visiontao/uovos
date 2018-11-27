#include "mex.h"
#include "project.h"
#include "Image.h"
#include "OpticalFlow.h"
#include <iostream>

using namespace std;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
  // check for proper number of input and output arguments
  if (nrhs < 2 || nrhs > 4) {
    mexErrMsgTxt("Only two to four input arguments are allowed!");
  }
  if (nlhs < 2 || nlhs > 3) {
    mexErrMsgTxt("Only two or three output arguments are allowed!");
  }
  DImage im1, im2, mask;
  im1.LoadMatlabImage(prhs[0]);
  im2.LoadMatlabImage(prhs[1]);
  if (nrhs > 3) mask.LoadMatlabImage(prhs[2]);  // Setup mask.

  if (im1.matchDimension(im2) == false) {
    mexErrMsgTxt("The two images don't match!");
  }
  // get the parameters
  double alpha = 0.01;
  double ratio = 0.75;
  int min_width = 40;
  int num_outer_fp_iterations = 3;
  int num_inner_fp_iterations = 1;
  int num_sor_iterations = 20;
  const int opt_idx = nrhs > 3 ? 3 : nrhs > 2 ? 2 : -1;
  if (opt_idx >= 0) {
    int nDims = mxGetNumberOfDimensions(prhs[opt_idx]);
    const int* dims = mxGetDimensions(prhs[opt_idx]);
    double* para = (double*) mxGetData(prhs[opt_idx]);
    int npara = dims[0] * dims[1];
    if (npara > 0) alpha = para[0];
    if (npara > 1) ratio = para[1];
    if (npara > 2) min_width = para[2];
    if (npara > 3) num_outer_fp_iterations = para[3];
    if (npara > 4) num_inner_fp_iterations = para[4];
    if (npara > 5) num_sor_iterations = para[5];
  }
//mexPrintf("alpha: %f, ratio: %f, min_width: %d, num_outer_fp_iterations: %d, "
//          "num_inner_fp_iterations: %d, num_sor_iterations: %d (mask(%d)\n",
//          alpha, ratio, min_width, num_outer_fp_iterations,
//          num_inner_fp_iterations,num_sor_iterations, mask.npixels());

  DImage vx, vy, warp_im2;
  OpticalFlow::Coarse2FineFlow(vx, vy, warp_im2, im1, im2, mask, alpha, ratio,
                               min_width, num_outer_fp_iterations,
                               num_inner_fp_iterations, num_sor_iterations);
  // output the parameters
  vx.OutputToMatlab(plhs[0]);
  vy.OutputToMatlab(plhs[1]);
  if (nlhs > 2) warp_im2.OutputToMatlab(plhs[2]);
}
