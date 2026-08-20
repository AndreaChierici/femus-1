#ifndef __femus_FracDepthEval_hpp__
#define __femus_FracDepthEval_hpp__

#include <Eigen/Dense>
#include <array>
#include <algorithm>
#include <cmath>
#include <vector>
#include "frac_depth_weights.hpp"

// Depth predictor for the FRACTIONAL kernel (10 -> 16 -> 16 -> 6, Tanh).
// Outputs log10 of the relative quadrature error at depths l = 0..5.
//
// Differences from the constant-kernel predictor in QuadDepthEval.hpp:
//   * the kernel is not constant, so the singularity at y = xg drives the
//     depth wherever the sub-element is near xg; two features carry that
//   * the element size scale h is the GEOMETRIC mean sqrt(hx*hy), not the
//     mean side length, and the aspect ratio is a feature, because graded
//     meshes contain elongated elements
//   * features are standardised before the first layer (featMean/featStd)
//
// Feature layout, matching gen_frac_labels.py exactly:
//   [0..3] sorted (|x_i - xg| - R)/h        signed corner distances
//   [4]    h/R
//   [5]    n_in/4                           fraction of corners inside the ball
//   [6]    min |(|x_i| - R)/h|
//   [7]    log10(h/d0)                      element size in kernel widths
//   [8]    log10((dist + d0)/h)             distance from xg to the element
//   [9]    log10(hy/hx)                     aspect ratio

// Build the feature vector for one (quad sub-element, quadrature point) pair.
// Kept free of Eigen so it can be unit-tested against the Python generator.
// xv: [dim][node], FEMuS layout, first 4 nodes used (cartesian quads).
inline void FracDepthFeatures(const std::vector<std::vector<double>> &xv,
                              const std::vector<double> &xg, const double R,
                              const double d0, std::array<double, 10> &f) {
  // side lengths: nodes 0-1 and 3-2 span x, nodes 1-2 and 0-3 span y
  auto edge = [&](int i, int j) {
    double s = 0.;
    for (int k = 0; k < 2; k++) s += (xv[k][i] - xv[k][j]) * (xv[k][i] - xv[k][j]);
    return std::sqrt(s);
  };
  const double hx = 0.5 * (edge(0, 1) + edge(3, 2));
  const double hy = 0.5 * (edge(1, 2) + edge(0, 3));
  const double h = std::sqrt(hx * hy);

  double d[4];
  int n_in = 0;
  for (int i = 0; i < 4; i++) {
    double s = 0.;
    for (int k = 0; k < 2; k++) s += (xv[k][i] - xg[k]) * (xv[k][i] - xg[k]);
    const double dist_i = std::sqrt(s);
    if (dist_i < R) n_in++;
    d[i] = (dist_i - R) / h;
  }
  std::sort(d, d + 4);
  double mind = std::fabs(d[0]);
  for (int i = 1; i < 4; i++) mind = std::min(mind, std::fabs(d[i]));

  // distance from xg to the element, zero when xg is inside it; exact for
  // axis-aligned quads, which is what the cartesian meshes provide
  double lo[2], hi[2];
  for (int k = 0; k < 2; k++) {
    lo[k] = hi[k] = xv[k][0];
    for (int i = 1; i < 4; i++) {
      lo[k] = std::min(lo[k], xv[k][i]);
      hi[k] = std::max(hi[k], xv[k][i]);
    }
  }
  double dd = 0.;
  for (int k = 0; k < 2; k++) {
    const double e = std::max(std::max(lo[k] - xg[k], 0.), xg[k] - hi[k]);
    dd += e * e;
  }
  const double dist = std::sqrt(dd);

  for (int i = 0; i < 4; i++) f[i] = d[i];
  f[4] = h / R;
  f[5] = 0.25 * (double)n_in;
  f[6] = mind;
  f[7] = std::log10(h / d0);
  f[8] = std::log10((dist + d0) / h);
  f[9] = std::log10(hy / hx);
}

struct FracDepthElem {
  double h, f4, f7, f9;
  double lo[2], hi[2];          // axis-aligned bounds, for the distance feature
};

inline void FracDepthElemInit(const std::vector<std::vector<double>> &xv,
                              const double R, const double d0,
                              FracDepthElem &c) {
  auto edge = [&](int i, int j) {
    double s = 0.;
    for (int k = 0; k < 2; k++) s += (xv[k][i] - xv[k][j]) * (xv[k][i] - xv[k][j]);
    return std::sqrt(s);
  };
  const double hx = 0.5 * (edge(0, 1) + edge(3, 2));
  const double hy = 0.5 * (edge(1, 2) + edge(0, 3));
  c.h  = std::sqrt(hx * hy);
  c.f4 = c.h / R;
  c.f7 = std::log10(c.h / d0);
  c.f9 = std::log10(hy / hx);
  for (int k = 0; k < 2; k++) {
    c.lo[k] = c.hi[k] = xv[k][0];
    for (int i = 1; i < 4; i++) {
      c.lo[k] = std::min(c.lo[k], xv[k][i]);
      c.hi[k] = std::max(c.hi[k], xv[k][i]);
    }
  }
                              }

inline void EvalFracDepthNet(const std::array<double, 10> &f, double out[6]) {
  using Eigen::RowMajor;
  Eigen::Map<const Eigen::Matrix<double, 16, 10, RowMajor>> W0(FracDepthNetWeights::W0);
  Eigen::Map<const Eigen::Matrix<double, 16, 1>> b0(FracDepthNetWeights::b0);
  Eigen::Map<const Eigen::Matrix<double, 16, 16, RowMajor>> W1(FracDepthNetWeights::W1);
  Eigen::Map<const Eigen::Matrix<double, 16, 1>> b1(FracDepthNetWeights::b1);
  Eigen::Map<const Eigen::Matrix<double, 6, 16, RowMajor>> W2(FracDepthNetWeights::W2);
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> b2(FracDepthNetWeights::b2);
  Eigen::Map<const Eigen::Matrix<double, 10, 1>> mu(FracDepthNetWeights::featMean);
  Eigen::Map<const Eigen::Matrix<double, 10, 1>> sd(FracDepthNetWeights::featStd);
  Eigen::Map<const Eigen::Matrix<double, 10, 1>> xraw(f.data());
  Eigen::Matrix<double, 10, 1> x = (xraw - mu).array() / sd.array();
  Eigen::Matrix<double, 16, 1> h0 = (W0 * x + b0).array().tanh();
  Eigen::Matrix<double, 16, 1> h1 = (W1 * h0 + b1).array().tanh();
  Eigen::Matrix<double, 6, 1> y = W2 * h1 + b2;
  for (int l = 0; l < 6; l++) out[l] = y(l, 0);
}

// Predicted depth for one (quad sub-element, quadrature point) pair.
// eps: relative tolerance on the inner integral; margin: log10 safety margin.
// Returns the first depth l whose predicted error satisfies
// 10^(pred_l + margin) < eps, or 5 if none does.
inline int PredictFracDepth(const std::vector<std::vector<double>> &xv,
                            const std::vector<double> &xg, const double R,
                            const double d0, const double eps,
                            const double margin = 0.7) {
  std::array<double, 10> f;
  FracDepthFeatures(xv, xg, R, d0, f);
  double p[6];
  EvalFracDepthNet(f, p);
  const double leps = std::log10(eps);
  for (int l = 0; l < 6; l++)
    if (p[l] + margin < leps) return l;
  return 5;
}

// Same as PredictFracDepth, with the per-sub-element part precomputed.
inline int PredictFracDepthCached(const FracDepthElem &c,
                                  const std::vector<std::vector<double>> &xv,
                                  const std::vector<double> &xg, const double R,
                                  const double d0, const double eps,
                                  const double margin = 0.7) {
    if(getenv("GATE_NONET")) return 5;
  std::array<double, 10> f;
  double d[4];
  int n_in = 0;
  for (int i = 0; i < 4; i++) {
    double s = 0.;
    for (int k = 0; k < 2; k++) s += (xv[k][i] - xg[k]) * (xv[k][i] - xg[k]);
    const double dist_i = std::sqrt(s);
    if (dist_i < R) n_in++;
    d[i] = (dist_i - R) / c.h;
  }
  std::sort(d, d + 4);
  double mind = std::fabs(d[0]);
  for (int i = 1; i < 4; i++) mind = std::min(mind, std::fabs(d[i]));

  double dd = 0.;
  for (int k = 0; k < 2; k++) {
    const double e = std::max(std::max(c.lo[k] - xg[k], 0.), xg[k] - c.hi[k]);
    dd += e * e;
  }

  for (int i = 0; i < 4; i++) f[i] = d[i];
  f[4] = c.f4;
  f[5] = 0.25 * (double)n_in;
  f[6] = mind;
  f[7] = c.f7;
  f[8] = std::log10((std::sqrt(dd) + d0) / c.h);
  f[9] = c.f9;

  double p[6];
  EvalFracDepthNet(f, p);
  const double leps = std::log10(eps);
  for (int l = 0; l < 6; l++)
    if (p[l] + margin < leps) return l;
    return 5;
}

#endif
