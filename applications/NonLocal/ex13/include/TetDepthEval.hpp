#ifndef __femus_TetDepthEval_hpp__
#define __femus_TetDepthEval_hpp__

#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cmath>
#include "tet_depth_weights.hpp"

// Depth predictor (7 -> 16 -> 16 -> 1, Tanh). Features:
// [0..3] sorted (|x_i - xg| - R)/h ; [4] h/R ; [5] n_in ; [6] min|(|x_i|-R)/h|
inline double EvalDepthNet(const std::array<float, 7> &f) {
  using Eigen::RowMajor;
  Eigen::Map<const Eigen::Matrix<float, 16, 7, RowMajor>> W0(DepthNetWeights::W0);
  Eigen::Map<const Eigen::Matrix<float, 16, 1>> b0(DepthNetWeights::b0);
  Eigen::Map<const Eigen::Matrix<float, 16, 16, RowMajor>> W1(DepthNetWeights::W1);
  Eigen::Map<const Eigen::Matrix<float, 16, 1>> b1(DepthNetWeights::b1);
  Eigen::Map<const Eigen::Matrix<float, 1, 16, RowMajor>> W2(DepthNetWeights::W2);
  Eigen::Map<const Eigen::Matrix<float, 1, 1>> b2(DepthNetWeights::b2);
  Eigen::Map<const Eigen::Matrix<float, 7, 1>> x(f.data());
  Eigen::Matrix<float, 16, 1> h0 = (W0 * x + b0).array().tanh();
  Eigen::Matrix<float, 16, 1> h1 = (W1 * h0 + b1).array().tanh();
  return (double)((W2 * h1 + b2)(0, 0));
}

// Predicted depth cap for one (tet element, ball center) pair.
// xv: [dim][node] coordinates; xg: ball center; R: horizon.
inline int PredictTetDepth(const std::vector<std::vector<double>> &xv,
                           const std::vector<double> &xg, const double R) {
  double dist[4];
  for (int i = 0; i < 4; i++) {
    double s = 0.;
    for (int k = 0; k < 3; k++) s += (xv[k][i]-xg[k])*(xv[k][i]-xg[k]);
    dist[i] = std::sqrt(s);
  }
  double h = 0.; int ne = 0;
  for (int i = 0; i < 4; i++)
    for (int j = i+1; j < 4; j++) {
      double s = 0.;
      for (int k = 0; k < 3; k++) s += (xv[k][i]-xv[k][j])*(xv[k][i]-xv[k][j]);
      h += std::sqrt(s); ne++;
    }
  h /= ne;
  std::array<float, 7> f;
  double d[4]; int n_in = 0;
  for (int i = 0; i < 4; i++) { d[i] = (dist[i]-R)/h; if (dist[i] < R) n_in++; }
  std::sort(d, d+4);
  double mind = std::min(std::min(std::fabs(d[0]), std::fabs(d[1])),
                         std::min(std::fabs(d[2]), std::fabs(d[3])));
  for (int i = 0; i < 4; i++) f[i] = (float)d[i];
  f[4] = (float)(h/R); f[5] = (float)n_in; f[6] = (float)mind;
  int lp = (int)std::ceil(EvalDepthNet(f) - 0.5);
  return std::max(lp, 0);
}

#endif
