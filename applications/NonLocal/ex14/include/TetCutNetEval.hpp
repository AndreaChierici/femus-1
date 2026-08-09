#ifndef __femus_TetCutNetEval_hpp__
#define __femus_TetCutNetEval_hpp__

#include <Eigen/Dense>
#include <array>
#include "tet_no_crossings_weights.hpp"

// Forward pass for TetCutNet (6 -> 128 x4 Tanh -> 1), using the weights
// exported by export_weights_header.py.
// Feature order (no_crossings set): [a2_0, a2_1, a2_2, t1, cnt, d2_geo]
inline double EvalTetCutNet(const std::array<float, 6> &features) {
  using namespace TetCutNetWeights;
  using Eigen::RowMajor;

  Eigen::Map<const Eigen::Matrix<float, 128, 6, RowMajor>> W0(TetCutNetWeights::W0);
  Eigen::Map<const Eigen::Matrix<float, 128, 1>> b0(TetCutNetWeights::b0);
  Eigen::Map<const Eigen::Matrix<float, 128, 128, RowMajor>> W1(TetCutNetWeights::W1);
  Eigen::Map<const Eigen::Matrix<float, 128, 1>> b1(TetCutNetWeights::b1);
  Eigen::Map<const Eigen::Matrix<float, 128, 128, RowMajor>> W2(TetCutNetWeights::W2);
  Eigen::Map<const Eigen::Matrix<float, 128, 1>> b2(TetCutNetWeights::b2);
  Eigen::Map<const Eigen::Matrix<float, 128, 128, RowMajor>> W3(TetCutNetWeights::W3);
  Eigen::Map<const Eigen::Matrix<float, 128, 1>> b3(TetCutNetWeights::b3);
  Eigen::Map<const Eigen::Matrix<float, 1, 128, RowMajor>> W4(TetCutNetWeights::W4);
  Eigen::Map<const Eigen::Matrix<float, 1, 1>> b4(TetCutNetWeights::b4);

  Eigen::Map<const Eigen::Matrix<float, 6, 1>> x(features.data());

  Eigen::Matrix<float, 128, 1> h0 = (W0 * x + b0).array().tanh();
  Eigen::Matrix<float, 128, 1> h1 = (W1 * h0 + b1).array().tanh();
  Eigen::Matrix<float, 128, 1> h2 = (W2 * h1 + b2).array().tanh();
  Eigen::Matrix<float, 128, 1> h3 = (W3 * h2 + b3).array().tanh();
  Eigen::Matrix<float, 1, 1> out = W4 * h3 + b4;

  return (double) out(0, 0);
}

#endif
