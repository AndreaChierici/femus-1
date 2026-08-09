#ifndef __femus_QuadDepthEval_hpp__
#define __femus_QuadDepthEval_hpp__

#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>
#include "quad_depth_weights.hpp"

// 2D depth predictor (7 -> 16 -> 16 -> 5, Tanh).
// Outputs log10 of the quadtree area error at depths l = 0..4.
// Features: [0..3] sorted (|x_i - xg| - R)/h ; [4] h/R ; [5] n_in ;
// [6] min|(|x_i| - R)/h| ; h = mean side length of the quad.
inline void EvalQuadDepthNet(const std::array<float, 7> &f, double out[5]) {
  using Eigen::RowMajor;
  Eigen::Map<const Eigen::Matrix<float, 16, 7, RowMajor>> W0(QuadDepthNetWeights::W0);
  Eigen::Map<const Eigen::Matrix<float, 16, 1>> b0(QuadDepthNetWeights::b0);
  Eigen::Map<const Eigen::Matrix<float, 16, 16, RowMajor>> W1(QuadDepthNetWeights::W1);
  Eigen::Map<const Eigen::Matrix<float, 16, 1>> b1(QuadDepthNetWeights::b1);
  Eigen::Map<const Eigen::Matrix<float, 5, 16, RowMajor>> W2(QuadDepthNetWeights::W2);
  Eigen::Map<const Eigen::Matrix<float, 5, 1>> b2(QuadDepthNetWeights::b2);
  Eigen::Map<const Eigen::Matrix<float, 7, 1>> x(f.data());
  Eigen::Matrix<float, 16, 1> h0 = (W0 * x + b0).array().tanh();
  Eigen::Matrix<float, 16, 1> h1 = (W1 * h0 + b1).array().tanh();
  Eigen::Matrix<float, 5, 1> y = W2 * h1 + b2;
  for (int l = 0; l < 5; l++) out[l] = (double)y(l, 0);
}

// Predicted remaining depth for one (quad sub-element, ball center) pair.
// xv: [dim][node] coordinates (FEMuS layout, first 4 nodes used);
// xg: ball center; R: horizon; eps: area tolerance relative to the quad;
// margin: log10 safety margin. Returns the first depth l whose predicted
// error 10^(pred_l + margin) < eps; returns 4 if none qualifies.
inline int PredictQuadDepth(const std::vector<std::vector<double>> &xv,
                            const std::vector<double> &xg, const double R,
                            const double eps, const double margin = 0.3) {
  double dist[4];
  for (int i = 0; i < 4; i++) {
    double s = 0.;
    for (int k = 0; k < 2; k++) s += (xv[k][i] - xg[k]) * (xv[k][i] - xg[k]);
    dist[i] = std::sqrt(s);
  }
  double h = 0.;
  for (int i = 0; i < 4; i++) {
    int j = (i + 1) % 4;
    double s = 0.;
    for (int k = 0; k < 2; k++) s += (xv[k][i] - xv[k][j]) * (xv[k][i] - xv[k][j]);
    h += std::sqrt(s);
  }
  h *= 0.25;
  double d[4]; int n_in = 0;
  for (int i = 0; i < 4; i++) { d[i] = (dist[i] - R) / h; if (dist[i] < R) n_in++; }
  std::sort(d, d + 4);
  double mind = std::min(std::min(std::fabs(d[0]), std::fabs(d[1])),
                         std::min(std::fabs(d[2]), std::fabs(d[3])));
  std::array<float, 7> f;
  for (int i = 0; i < 4; i++) f[i] = (float)d[i];
  f[4] = (float)(h / R); f[5] = (float)n_in; f[6] = (float)mind;
  double p[5];
  EvalQuadDepthNet(f, p);
  const double leps = std::log10(eps);
  for (int l = 0; l < 5; l++)
    if (p[l] + margin < leps) return l;
  return 4;
}

#endif
