#include "OpticalFlow.h"
#include "ImageProcessing.h"
#include "GaussianPyramid.h"
#include <cstdlib> 
#include <iostream>
#include <fstream>
#include <stdarg.h>

using namespace std;

#ifndef _MATLAB
bool OpticalFlow::display_msg_ = true;
#else
bool OpticalFlow::display_msg_ = false;
#endif

namespace {

template <typename T> inline T Square(const T& v) { return v * v; }

void SaveImage(const DImage& img, const char* filename_fmt, ...) {
  char filename[1024];
  va_list args;
  va_start(args, filename_fmt);
  vsprintf(filename, filename_fmt, args);
  va_end (args);

  ofstream f(filename, ios::out);
  if (f.is_open()) {
    const double* ptr = img.data();
    const int w = img.width(), h = img.height(), d = img.nchannels();
    for (int k = 0; k < d; ++k) {
      for (int y = 0; y < h; ++y) {
        for (int x = 0, idx = y * w * d + k; x < w; ++x, idx += d) {
          f << " " << ptr[idx];
        }
        f << endl;
      }
    }
    f.close();
  }
}

}  // namespace

OpticalFlow::OpticalFlow() {}
OpticalFlow::~OpticalFlow() {}

//-----------------------------------------------------------------------------
// function to convert image to feature image
//-----------------------------------------------------------------------------

void OpticalFlow::im2feature(DImage& img_feature, const DImage& img) {
  const int width = img.width();
  const int height = img.height();
  const int num_channels = img.nchannels();
  DImage img_dx, img_dy, img_gray;
  if (num_channels == 1) {
    img.dx(img_dx, true);
    img.dy(img_dy, true);
    img_feature.allocate(width, height, 3);
    FlowPrecisionType* feat = img_feature.data();
    const FlowPrecisionType* gray = img.data();
    const FlowPrecisionType* dx = img_dx.data();
    const FlowPrecisionType* dy = img_dy.data();
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++) {
        const int offset = y * width + x;
        feat[offset * 3] = gray[offset];
        feat[offset * 3 + 1] = dx[offset];
        feat[offset * 3 + 2] = dy[offset];
      }
  } else if (num_channels == 3) {
    img.desaturate(img_gray);
    img_gray.dx(img_dx, true);
    img_gray.dy(img_dy, true);
    img_feature.allocate(width, height, 5);
    FlowPrecisionType* feat = img_feature.data();
    const FlowPrecisionType* rgb = img.data();
    const FlowPrecisionType* gray = img_gray.data();
    const FlowPrecisionType* dx = img_dx.data();
    const FlowPrecisionType* dy = img_dy.data();
    for (int y = 0; y < height; y++)
      for (int x = 0; x < width; x++) {
        const int offset = y * width + x;
        feat[offset * 5] = gray[offset];
        feat[offset * 5 + 1] = dx[offset];
        feat[offset * 5 + 2] = dy[offset];
        feat[offset * 5 + 3] = rgb[offset * 3 + 1] - rgb[offset * 3];
        feat[offset * 5 + 4] = rgb[offset * 3 + 1] - rgb[offset * 3 + 2];
      }
  } else {
    img_feature.copyData(img);
  }
}

//-----------------------------------------------------------------------------
// function to perfomr coarse to fine optical flow estimation
//-----------------------------------------------------------------------------

void OpticalFlow::Coarse2FineFlow(DImage &vx, DImage &vy, DImage &img2_warp,
                                  const DImage &img1, const DImage &img2,
                                  const DImage& mask,
                                  double alpha, double ratio, int min_width,
                                  int num_outer_fp_iterations,
                                  int num_inner_fp_iterations,
                                  int num_cg_iterations) {
//display_msg_ = true;
  // first build the pyramid of the two images
  GaussianPyramid gaussian_pyramid1;
  GaussianPyramid gaussian_pyramid2;
  GaussianPyramid gaussian_pyramid_mask;
  if (display_msg_) cout << "Constructing pyramid... mask: " << mask.npixels() << endl;
  gaussian_pyramid1.ConstructPyramid(img1, ratio, min_width);
  gaussian_pyramid2.ConstructPyramid(img2, ratio, min_width);
  if (!mask.IsEmpty()) {
    gaussian_pyramid_mask.ConstructPyramid(mask, ratio, min_width);
  }
  if (display_msg_) cout << "done!" << endl;
  
  // now iterate from the top level to the bottom
  DImage img1_feature, img2_feature, img2_feature_warp, level_mask;
  for (int k = gaussian_pyramid1.nlevels() - 1; k >= 0; k--) {
    const int width = gaussian_pyramid1.Image(k).width();
    const int height = gaussian_pyramid1.Image(k).height();
    if (display_msg_) cout << "Pyramid level " << k << ": " << width << "x" << height << endl;
    im2feature(img1_feature, gaussian_pyramid1.Image(k));
    im2feature(img2_feature, gaussian_pyramid2.Image(k));

    if (k == gaussian_pyramid1.nlevels() - 1) { // if at the top level
      vx.allocate(width, height);
      vy.allocate(width, height);
      img2_feature_warp.copyData(img2_feature);
    } else {
      vx.imresize(width, height);
      vx.Multiplywith(1 / ratio);
      vy.imresize(width, height);
      vy.Multiplywith(1 / ratio);
      warpFL(img2_feature_warp, img1_feature, img2_feature, vx, vy);
    }
    if (!mask.IsEmpty()) level_mask.copy(gaussian_pyramid_mask.Image(k));
    SmoothFlowSOR(k, img1_feature, img2_feature, level_mask,
                  img2_feature_warp, vx, vy,
                  alpha, num_outer_fp_iterations + k, num_inner_fp_iterations,
                  num_cg_iterations + k * 3);
//    SmoothFlowPDE(k, img1_feature, img2_feature, level_mask,
//                  img2_feature_warp, vx, vy,
//                  alpha, num_outer_fp_iterations + k, num_inner_fp_iterations,
//                  num_cg_iterations + k * 3);
  }
  img2.warpImageBicubicRef(img1, img2_warp, vx, vy);
  img2_warp.threshold();
}

//-----------------------------------------------------------------------------
// function to compute optical flow field using two fixed point iterations
// Input arguments:
//   img1, img2: frame 1 and frame 2
//   warpimg2: the warped frame 2 according to the current flow field u and v
//   u,v: the current flow field, NOTICE that they are also output arguments
//-----------------------------------------------------------------------------
void OpticalFlow::SmoothFlowSOR(int lvl, const DImage& img1, const DImage& img2,
                                const DImage& mask,
                                DImage &warpimg2, DImage &u, DImage &v,
                                double alpha, int num_outer_fp_iterations,
                                int num_inner_fp_iterations,
                                int num_sor_iterations) {
//  DImage mask;
  DImage imdx, imdy, imdt;
  const int width = img1.width();
  const int height = img1.height();
  const int num_channels = img1.nchannels();
  const int num_pixels = width * height;

//SaveImage(img1, "_img1_%02d.txt", lvl);
//SaveImage(img2, "_img2_%02d.txt", lvl);
//SaveImage(u, "_u_%02d.txt", lvl);
//SaveImage(v, "_v_%02d.txt", lvl);

  DImage du(width, height), dv(width, height);
  DImage uu(width, height), vv(width, height);
  DImage ux(width, height), uy(width, height);
  DImage vx(width, height), vy(width, height);
  DImage phi_1st(width, height);
  DImage psi_1st(width, height, num_channels);

  DImage imdxy, imdx2, imdy2, imdtdx, imdtdy;
  DImage ImDxy, ImDx2, ImDy2, ImDtDx, ImDtDy;
  DImage foo1, foo2;

  double prob1, prob2, prob11, prob22;
  double var_epsilon_phi = pow(0.001, 2);
  double var_epsilon_psi = pow(0.001, 2);

  //--------------------------------------------------------------------------
  // the outer fixed point iteration
  //--------------------------------------------------------------------------
  for (int count = 0; count < num_outer_fp_iterations; count++) {
    // compute the gradient
    getDxs(imdx, imdy, imdt, img1, warpimg2);

    // set the derivative of the flow field to be zero
    du.reset();
    dv.reset();

    //-------------------------------------------------------------------------
    // the inner fixed point iteration
    //-------------------------------------------------------------------------
    for (int hh = 0; hh < num_inner_fp_iterations; hh++) {
      // compute the derivatives of the current flow field
      if (hh == 0) {
        uu.copyData(u);
        vv.copyData(v);
      } else {
        uu.Add(u, du);
        vv.Add(v, dv);
      }
      uu.dx(ux);
      uu.dy(uy);
      vv.dx(vx);
      vv.dy(vy);

      // compute the weight of phi
      phi_1st.reset();
      FlowPrecisionType* phiData = phi_1st.data();
      const FlowPrecisionType* uxData = ux.data();
      const FlowPrecisionType* uyData = uy.data();
      const FlowPrecisionType* vxData = vx.data();
      const FlowPrecisionType* vyData = vy.data();
      for (int i = 0; i < num_pixels; i++) {
        double temp = Square(uxData[i]) + Square(uyData[i]) +
            Square(vxData[i]) + Square(vyData[i]);
        phiData[i] = 0.5 / sqrt(temp + var_epsilon_phi);
        //phiData[i] = 0.5 * pow(temp + var_epsilon_phi, power_alpha - 1);
        //phiData[i] = 1 / (0.5 + temp);
      }
      // compute the nonlinear term of psi
      psi_1st.reset();
      FlowPrecisionType* psiData = psi_1st.data();
      const FlowPrecisionType* imdxData = imdx.data();
      const FlowPrecisionType* imdyData = imdy.data();
      const FlowPrecisionType* imdtData = imdt.data();
      const FlowPrecisionType* duData = du.data();
      const FlowPrecisionType* dvData = dv.data();
    
      if (num_channels == 1) {
        for (int i = 0; i < num_pixels; i++) {
          double temp = Square(imdtData[i] + imdxData[i] * duData[i] +
                               imdyData[i] * dvData[i]);
          psiData[i] = 1 / (2 * sqrt(temp + var_epsilon_psi));
        }
      } else {
        for (int i = 0; i < num_pixels; i++)
          for (int k = 0; k < num_channels; k++) {
            int offset = i * num_channels + k;
            double temp = Square(imdtData[offset] +
                                 imdxData[offset] * duData[i] +
                                 imdyData[offset] * dvData[i]);
            psiData[offset] = 1 / (2 * sqrt(temp + var_epsilon_psi));
          }
      }
//SaveImage(psi_1st, "_psi_%02d_%02d_%d.txt", lvl, count, hh);
//SaveImage(phi_1st, "_phi_%02d_%02d_%d.txt", lvl, count, hh);
      // prepare the components of the large linear system
      ImDxy.Multiply(psi_1st, imdx, imdy);
      ImDx2.Multiply(psi_1st, imdx, imdx);
      ImDy2.Multiply(psi_1st, imdy, imdy);
      ImDtDx.Multiply(psi_1st, imdx, imdt);
      ImDtDy.Multiply(psi_1st, imdy, imdt);

      if (num_channels > 1) {
        ImDxy.collapse(imdxy);
        ImDx2.collapse(imdx2);
        ImDy2.collapse(imdy2);
        ImDtDx.collapse(imdtdx);
        ImDtDy.collapse(imdtdy);
      } else {
        imdxy.copyData(ImDxy);
        imdx2.copyData(ImDx2);
        imdy2.copyData(ImDy2);
        imdtdx.copyData(ImDtDx);
        imdtdy.copyData(ImDtDy);
      }
      // Multiply with mask.
      if (mask.IsEmpty() == false) {
        imdxy.Multiplywith(mask);
        imdx2.Multiplywith(mask);
        imdy2.Multiplywith(mask);
        imdtdx.Multiplywith(mask);
        imdtdy.Multiplywith(mask);
      }
      // Laplacian filtering of the current flow field
      Laplacian(foo1, u, phi_1st);
      Laplacian(foo2, v, phi_1st);

      for (int i = 0; i < num_pixels; i++) {
        imdtdx.data()[i] = -imdtdx.data()[i] - alpha * foo1.data()[i];
        imdtdy.data()[i] = -imdtdy.data()[i] - alpha * foo2.data()[i];
      }
//SaveImage(imdx2, "_imdx2_%02d_%02d_%d.txt", lvl, count, hh);
//SaveImage(imdy2, "_imdy2_%02d_%02d_%d.txt", lvl, count, hh);
//SaveImage(imdxy, "_imdxy_%02d_%02d_%d.txt", lvl, count, hh);
//SaveImage(imdtdx, "_imdtdx_%02d_%02d_%d.txt", lvl, count, hh);
//SaveImage(imdtdy, "_imdtdy_%02d_%02d_%d.txt", lvl, count, hh);
      // set omega
      const double omega = 1.8;
      DImage coeff_x(width, height), coeff_y(width, height);
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          const int offset = i * width + j;
          double c = 0.05;
          if (j > 0) c += phiData[offset - 1];
          if (j < width - 1) c += phiData[offset];
          if (i > 0) c += phiData[offset - width];
          if (i < height - 1) c += phiData[offset];
          coeff_x.data()[offset] = omega / (imdx2.data()[offset] + alpha * c);
          coeff_y.data()[offset] = omega / (imdy2.data()[offset] + alpha * c);
        }
      }
      // here we start SOR
      du.reset();
      dv.reset();
      for (int k = 0; k < num_sor_iterations; k++) {
        for (int i = 0; i < height; i++)
          for (int j = 0; j < width; j++) {
            const int offset = i * width+j;
            double sigma1 = 0, sigma2 = 0, coeff = 0;
            if (j > 0) {
              const double w = phiData[offset - 1];
              sigma1 += w * du.data()[offset - 1];
              sigma2 += w * dv.data()[offset - 1];
//              coeff += w;
            }
            if (j < width - 1) {
              const double w = phiData[offset];
              sigma1 += w * du.data()[offset + 1];
              sigma2 += w * dv.data()[offset + 1];
//              coeff += w;
            }
            if (i > 0) {
              const double w = phiData[offset - width];
              sigma1 += w * du.data()[offset - width];
              sigma2 += w * dv.data()[offset - width];
//              coeff += w;
            }
            if (i < height - 1) {
              const double w = phiData[offset];
              sigma1 += w * du.data()[offset + width];
              sigma2 += w * dv.data()[offset + width];
//              coeff += w;
            }
            // compute du
            du.data()[offset] = (1 - omega) * du.data()[offset] +
                coeff_x.data()[offset] *
//                omega / (imdx2.data()[offset] + alpha * (0.05 + coeff)) *
                (alpha * sigma1 + imdtdx.data()[offset]
                 - imdxy.data()[offset] * dv.data()[offset]);
            // compute dv
            dv.data()[offset] = (1 - omega) * dv.data()[offset] +
                coeff_y.data()[offset] *
//                omega / (imdy2.data()[offset] + alpha * (0.05 + coeff)) *
                (alpha * sigma2 + imdtdy.data()[offset]
                 - imdxy.data()[offset] * du.data()[offset]);
          }
//        char filename[256];
//        sprintf(filename, "_du_%03d_%02d_%d_%02d.pgm", width, count, hh, k);
//        du.saveImagePGM(filename, 100, 128);
//        sprintf(filename, "_dv_%03d_%02d_%d_%02d.pgm", width, count, hh, k);
//        dv.saveImagePGM(filename, 100, 128);

//SaveImage(du, "_du_%02d_%02d_%d_%02d.txt", lvl, count, hh, k);
//SaveImage(dv, "_dv_%02d_%02d_%d_%02d.txt", lvl, count, hh, k);
      }
    }
    u.Add(du);
    v.Add(dv);
    warpFL(warpimg2, img1, img2, u, v);
//SaveImage(u, "_u_%02d_%02d.txt", lvl, count);
//SaveImage(v, "_v_%02d_%02d.txt", lvl, count);
//SaveImage(warpimg2, "_warpimg2_%02d_%02d.txt", lvl, count);
    //img2.warpImageBicubicRef(img1,warpimg2,BicubicCoeff,u,v);
  }
}

void OpticalFlow::SmoothFlowPDE(int lvl, const DImage& img1, const DImage& img2,
                                const DImage& mask,
                                DImage &warpimg2, DImage &u, DImage &v,
                                double alpha, int num_outer_fp_iterations,
                                int num_inner_fp_iterations,
                                int num_cg_iterations) {
  DImage imdx, imdy, imdt;
  const int width = img1.width();
  const int height = img1.height();
  const int num_channels = img1.nchannels();
  const int num_pixels = width * height;

  DImage du(width, height), dv(width, height);
  DImage uu(width, height), vv(width, height);
  DImage ux(width, height), uy(width, height);
  DImage vx(width, height), vy(width, height);
  DImage phi_1st(width, height);
  DImage psi_1st(width, height, num_channels);

  DImage imdxy, imdx2, imdy2, imdtdx, imdtdy;
  DImage ImDxy, ImDx2, ImDy2, ImDtDx, ImDtDy;
	DImage A11,A12,A22,b1,b2;
  DImage foo1, foo2;
	DImage r1, r2, p1, p2, q1, q2;

  double prob1, prob2, prob11, prob22;
  double var_epsilon_phi = pow(0.001, 2);
  double var_epsilon_psi = pow(0.001, 2);

  //--------------------------------------------------------------------------
  // the outer fixed point iteration
  //--------------------------------------------------------------------------
  for (int count = 0; count < num_outer_fp_iterations; count++) {
    // compute the gradient
    getDxs(imdx, imdy, imdt, img1, warpimg2);

    // set the derivative of the flow field to be zero
    du.reset();
    dv.reset();

    //-------------------------------------------------------------------------
    // the inner fixed point iteration
    //-------------------------------------------------------------------------
    for (int hh = 0; hh < num_inner_fp_iterations; hh++) {
      // compute the derivatives of the current flow field
      if (hh == 0) {
        uu.copyData(u);
        vv.copyData(v);
      } else {
        uu.Add(u, du);
        vv.Add(v, dv);
      }
      uu.dx(ux);
      uu.dy(uy);
      vv.dx(vx);
      vv.dy(vy);
      // Multiply with the mask. ??

      // compute the weight of phi
      phi_1st.reset();
      FlowPrecisionType* phiData = phi_1st.data();
      const FlowPrecisionType* uxData = ux.data();
      const FlowPrecisionType* uyData = uy.data();
      const FlowPrecisionType* vxData = vx.data();
      const FlowPrecisionType* vyData = vy.data();
      for (int i = 0; i < num_pixels; i++) {
        double temp = Square(uxData[i]) + Square(uyData[i]) +
            Square(vxData[i]) + Square(vyData[i]);
        phiData[i] = 0.5 / sqrt(temp + var_epsilon_phi);
        //phiData[i] = 0.5 * pow(temp + var_epsilon_phi, power_alpha - 1);
        //phiData[i] = 1 / (0.5 + temp);
      }
      // compute the nonlinear term of psi
      psi_1st.reset();
      FlowPrecisionType* psiData = psi_1st.data();
      const FlowPrecisionType* imdxData = imdx.data();
      const FlowPrecisionType* imdyData = imdy.data();
      const FlowPrecisionType* imdtData = imdt.data();
      const FlowPrecisionType* duData = du.data();
      const FlowPrecisionType* dvData = dv.data();
    
      if (num_channels == 1) {
        for (int i = 0; i < num_pixels; i++) {
          double temp = Square(imdtData[i] + imdxData[i] * duData[i] +
                               imdyData[i] * dvData[i]);
          psiData[i] = 1 / (2 * sqrt(temp + var_epsilon_psi));
        }
      } else {
        for (int i = 0; i < num_pixels; i++)
          for (int k = 0; k < num_channels; k++) {
            int offset = i * num_channels + k;
            double temp = Square(imdtData[offset] +
                                 imdxData[offset] * duData[i] +
                                 imdyData[offset] * dvData[i]);
            psiData[offset] = 1 / (2 * sqrt(temp + var_epsilon_psi));
          }
      }
//SaveImage(psi_1st, "_psi_%02d_%02d_%d.txt", lvl, count, hh);
//SaveImage(phi_1st, "_phi_%02d_%02d_%d.txt", lvl, count, hh);
      // prepare the components of the large linear system
      ImDxy.Multiply(psi_1st, imdx, imdy);
      ImDx2.Multiply(psi_1st, imdx, imdx);
      ImDy2.Multiply(psi_1st, imdy, imdy);
      ImDtDx.Multiply(psi_1st, imdx, imdt);
      ImDtDy.Multiply(psi_1st, imdy, imdt);

      if (num_channels > 1) {
        ImDxy.collapse(imdxy);
        ImDx2.collapse(imdx2);
        ImDy2.collapse(imdy2);
        ImDtDx.collapse(imdtdx);
        ImDtDy.collapse(imdtdy);
      } else {
        imdxy.copyData(ImDxy);
        imdx2.copyData(ImDx2);
        imdy2.copyData(ImDy2);
        imdtdx.copyData(ImDtDx);
        imdtdy.copyData(ImDtDy);
      }
      // Multiply with the mask.
      if (mask.IsEmpty() == false) {
        imdxy.Multiplywith(mask);
        imdx2.Multiplywith(mask);
        imdy2.Multiplywith(mask);
        imdtdx.Multiplywith(mask);
        imdtdy.Multiplywith(mask);
      }
      // filtering
      imdx2.smoothing(A11, 3);
      imdxy.smoothing(A12, 3);
      imdy2.smoothing(A22, 3);

      // add epsilon to A11 and A22
      A11.Add(alpha * 0.1);
      A22.Add(alpha * 0.1);

      // form b
      imdtdx.smoothing(b1, 3);
      imdtdy.smoothing(b2, 3);
      // laplacian filtering of the current flow field
      Laplacian(foo1, u, phi_1st);
      Laplacian(foo2, v, phi_1st);

      double* b1Data = b1.data();
      double* b2Data = b2.data();
      const double* foo1Data = foo1.data();
      const double* foo2Data = foo2.data();
      for (int i = 0; i < num_pixels; i++) {
        b1Data[i] = -b1Data[i] - alpha * foo1Data[i];
        b2Data[i] = -b2Data[i] - alpha * foo2Data[i];
      }

      //-----------------------------------------------------------------------
      // conjugate gradient algorithm
      //-----------------------------------------------------------------------
      r1.copyData(b1);
      r2.copyData(b2);
      du.reset();
      dv.reset();

      double prev_rou;
      for (int k = 0; k < num_cg_iterations; k++) {
        double rou = r1.norm2() + r2.norm2();
        if (rou < 1E-10) break;
        if (k == 0) {
          p1.copyData(r1);
          p2.copyData(r2);
        } else {
          double ratio = rou / prev_rou;
          p1.Add(r1, p1, ratio);
          p2.Add(r2, p2, ratio);
        }
        // go through the large linear system
        foo1.Multiply(A11, p1);
        foo2.Multiply(A12, p2);
        q1.Add(foo1, foo2);
        Laplacian(foo1, p1, phi_1st);
        q1.Add(foo1, alpha);

        foo1.Multiply(A12, p1);
        foo2.Multiply(A22, p2);
        q2.Add(foo1, foo2);
        Laplacian(foo2, p2, phi_1st);
        q2.Add(foo2, alpha);

        double beta = rou / (p1.innerproduct(q1) + p2.innerproduct(q2));
        du.Add(p1, beta);
        dv.Add(p2, beta);
        r1.Add(q1, -beta);
        r2.Add(q2, -beta);

        prev_rou = rou;
      }
    }
    // update the flow field
    u.Add(du, 1);
    v.Add(dv, 1);
    warpFL(warpimg2, img1, img2, u, v);
  }
}

//-----------------------------------------------------------------------------
// function to warp image based on the flow field
//-----------------------------------------------------------------------------
void OpticalFlow::warpFL(DImage &warpimg2,
                         const DImage &img1, const DImage &img2,
                         const DImage &vx, const DImage &vy) {
  if (warpimg2.matchDimension(img2) == false) {
    warpimg2.allocate(img2.width(), img2.height(), img2.nchannels());
  }
  ImageProcessing::warpImage(warpimg2.data(), img1.data(), img2.data(),
                             vx.data(), vy.data(), img2.width(), img2.height(),
                             img2.nchannels());
}

void OpticalFlow::Laplacian(DImage &output, const DImage &input,
                            const DImage& weight) {
  if (output.matchDimension(input) == false) output.allocate(input);
  output.reset();

  if (input.matchDimension(weight) == false) {
    cout<<"Error in image dimension matching OpticalFlow::Laplacian()!"<<endl;
    return;
  }

  const FlowPrecisionType* inputData = input.data();
  const FlowPrecisionType* weightData = weight.data();
  int width = input.width(), height = input.height();
  DImage foo(width, height);
  FlowPrecisionType* fooData = foo.data();
  FlowPrecisionType* outputData = output.data();

  // horizontal filtering
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width - 1; j++) {
      const int offset = i * width + j;
      fooData[offset] = (inputData[offset + 1] - inputData[offset]) *
          weightData[offset];
    }
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      const int offset = i * width + j;
      if (j < width - 1) outputData[offset] -= fooData[offset];
      if (j > 0) outputData[offset] += fooData[offset - 1];
    }
  foo.reset();
  // vertical filtering
  for (int i = 0; i < height - 1; i++)
    for (int j = 0; j < width; j++) {
      const int offset = i * width + j;
      fooData[offset] = (inputData[offset + width] - inputData[offset]) *
          weightData[offset];
    }
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      const int offset = i * width + j;
      if (i < height - 1) outputData[offset] -= fooData[offset];
      if (i > 0) outputData[offset] += fooData[offset - width];
    }
}

void OpticalFlow::Laplacian2(DImage &output, const DImage &input,
                             const DImage& weight)
{
  if (output.matchDimension(input) == false) output.allocate(input);
  output.reset();

  if (input.matchDimension(weight) == false) {
    cout<<"Error in image dimension matching OpticalFlow::Laplacian()!"<<endl;
    return;
  }
  DImage foo1, foo2;
  input.dxx(foo1);
  foo1.Multiplywith(weight);
  foo1.dxx(output);

  input.dyy(foo1);
  foo1.Multiplywith(weight);
  foo1.dyy(foo2);

  output.Add(foo2);
}

//-----------------------------------------------------------------------------
//  function to compute dx, dy and dt for motion estimation
//-----------------------------------------------------------------------------

void OpticalFlow::getDxs(DImage &imdx, DImage &imdy, DImage &imdt,
                         const DImage &im1, const DImage &im2) {
  double gfilter[5] = { 0.02, 0.11, 0.74, 0.11, 0.02 };
  DImage img1, img2, Im;

  im1.imfilter_hv(img1, gfilter, 2, gfilter, 2);
  im2.imfilter_hv(img2, gfilter, 2, gfilter, 2);
  Im.copyData(img1);
  Im.Multiplywith(0.4);
  Im.Add(img2, 0.6);

  Im.dx(imdx, true);
  Im.dy(imdy, true);
  imdt.Subtract(img2, img1);

  imdx.setDerivative();
  imdy.setDerivative();
  imdt.setDerivative();
}

void OpticalFlow::genInImageMask(DImage &mask,
                                 const DImage &vx, const DImage &vy) {
  const int width = vx.width();
  const int height = vx.height();
  if (mask.matchDimension(vx) == false) mask.allocate(width, height);
  const double* pvx = vx.data();
  const double* pvy = vy.data();
  mask.reset();
  double* pmask=mask.data();
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      const int offset = i*width+j;
      const double y = i + pvx[offset];
      const double x = j + pvy[offset];
      if (x < 0 || x > width - 1 || y < 0 || y > height - 1) continue;
      pmask[offset]=1;
    }
}


void OpticalFlow::genInImageMask(DImage &mask,
                                 const DImage &vx, const DImage &vy,
                                 const DImage& mask2) {
  const int width = vx.width();
  const int height = vx.height();
  if (mask.matchDimension(vx) == false) mask.allocate(width,height);
  if (mask.matchDimension(mask2) == false) {
    cout << "Error in image dimensions! " << mask.width() << "x" <<
        mask.height() << "," << mask.nchannels()
        << ", " << mask2.width() << "x" << mask2.height() << "," <<
        mask2.nchannels() << endl;
  }
  const double* pvx = vx.data();
  const double* pvy = vy.data();
  const double* pmask2 = mask2.data();
  mask.reset();
  double* pmask = mask.data();
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      const int offset = i * width + j;
      const int y = i + pvx[offset];
      const int x = j + pvy[offset];
      if (x < 0 || x > width - 1 || y < 0 || y > height - 1) continue;
      if (pmask2[y * width + x]>0) pmask[offset] = pmask2[y * width + x];
    }
}
