#pragma once

#include "Image.h"
#include "NoiseModel.h"
#include "Vector.h"
#include <vector>

typedef double FlowPrecisionType;

class OpticalFlow {
 public:
  static bool display_msg_;

  OpticalFlow();
  ~OpticalFlow();

  // function of coarse to fine optical flow
  static void Coarse2FineFlow(
      DImage& vx, DImage& vy, DImage &img2_warp,
      const DImage& img1, const DImage& img2, const DImage& mask,
      double alpha = 0.01, double ratio = 0.75, int min_width = 20,
      int num_outer_fp_iterations = 7, int num_inner_fp_iterations = 1,
      int num_cg_iterations = 30);
// ComputeOpticalFlow member function : 0.01, 0.75, 30, 15, 1, 40

  static void getDxs(DImage& imdx, DImage& imdy, DImage& imdt,
                     const DImage& im1, const DImage& im2);
  static void warpFL(DImage& warpIm2, const DImage& Im1, const DImage& Im2,
                     const DImage& vx, const DImage& vy);

  static void SmoothFlowSOR(int lvl, const DImage& img1, const DImage& img2,
                            const DImage& mask,
                            DImage& warpimg2, DImage& u, DImage& v,
                            double alpha, int num_outer_fp_iterations,
                            int num_inner_fp_iterations,
                            int num_sor_iterations);
  static void SmoothFlowPDE(int lvl, const DImage& img1, const DImage& img2,
                            const DImage& mask,
                            DImage& warpimg2, DImage& u, DImage& v,
                            double alpha, int num_outer_fp_iterations,
                            int num_inner_fp_iterations,
                            int num_cg_iterations);

  static void Laplacian(DImage& output, const DImage& input,
                        const DImage& weight);
  static void Laplacian2(DImage &output, const DImage &input,
                         const DImage& weight);

  static void genInImageMask(DImage &mask, const DImage &vx, const DImage &vy);
  static void genInImageMask(DImage &mask,
                             const DImage &vx, const DImage &vy,
                             const DImage& mask2);

  // function to convert image to features
  static void im2feature(DImage& imfeature, const DImage& im);
};

