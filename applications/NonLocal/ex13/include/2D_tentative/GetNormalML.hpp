#ifndef __femus_GetNormalML_hpp__
#define __femus_GetNormalML_hpp__

#include "GetNormal.hpp"
#include <torch/script.h>

#include <string>
#include <vector>
#include <cmath>
#include <iostream>

// ============================================================================
// CutCase: stores everything needed to apply the ML offset after batch inference
// ============================================================================
struct CutCase {
  unsigned jel;
  unsigned jg;
  unsigned nve;                    // 3 (TRI) or 4 (QUAD)
  unsigned elType;
  unsigned crossEdge[2];
  double   crossT[2];
  double   sigma;
  std::vector<float> feat;        // feature vector for the network
  // Physical geometry needed for ApplyOffset
  std::vector<std::vector<double>> xv;  // element node coords
  std::vector<double> xg;              // ball center
  double R;
};

// ============================================================================
// BallApproximationML
// ============================================================================
class BallApproximationML : public BallApproximation {

public:

  BallApproximationML(const std::string &modelPathQuad,
                      const std::string &modelPathTri) {
    _useML = true;

    try {
      _modelQuad = torch::jit::load(modelPathQuad);
      _modelQuad.eval();
      _hasModelQuad = true;
      std::cout << "[GetNormalML] Loaded QUAD4 model: " << modelPathQuad << "\n";
    } catch(const c10::Error &e) {
      std::cerr << "[GetNormalML] WARNING: QUAD4 model not loaded: " << e.what() << "\n";
      _hasModelQuad = false;
    }

    try {
      _modelTri = torch::jit::load(modelPathTri);
      _modelTri.eval();
      _hasModelTri = true;
      std::cout << "[GetNormalML] Loaded TRI3 model: " << modelPathTri << "\n";
    } catch(const c10::Error &e) {
      std::cerr << "[GetNormalML] WARNING: TRI3 model not loaded: " << e.what() << "\n";
      _hasModelTri = false;
    }
  }

  void SetUseML(bool useML) { _useML = useML; }

  // --------------------------------------------------------------------------
  // Original single-call interface (kept for compatibility, not used in ex12)
  // --------------------------------------------------------------------------
  void GetNormal(const unsigned &elType,
                 const std::vector<std::vector<double>> &xv,
                 const std::vector<double> &xg,
                 const double &R,
                 std::vector<double> &b,
                 double &db,
                 unsigned &cut) {
    if(!_useML) {
      BallApproximation::GetNormal(elType, xv, xg, R, b, db, cut);
      return;
    }
    // Detect
    CutCase cc;
    DetectCut(elType, xv, xg, R, cut, cc);
    if(cut != 1) return;
    // Single inference
    auto &model = (elType == 4) ? _modelTri : _modelQuad;
    double s_norm = RunInferenceSingle(model, cc.feat);
    // double s_norm = 0.;
    // Apply
    ApplyOffset(cc, s_norm, b, db);

    // --- Temporary debug: compare ML vs geometric ---
    {
      std::vector<double> b_geo;
      double db_geo;
      unsigned cut_geo;
      BallApproximation::GetNormal(elType, xv, xg, R, b_geo, db_geo, cut_geo);
      static int cnt = 0;
      if(cnt++ < 5) {
        std::cerr << "[CMP] s_norm=" << s_norm << " sigma=" << cc.sigma << "\n"
        << "  b_ml =(" << b[0]     << "," << b[1]     << ") db_ml =" << db     << "\n"
        << "  b_geo=(" << b_geo[0] << "," << b_geo[1] << ") db_geo=" << db_geo << "\n\n";
      }
    }
    // --- End debug ---
  }

  void GetNormalQuad(const std::vector<std::vector<double>> &xv,
                     const std::vector<double> &xg,
                     const double &R,
                     std::vector<double> &b,
                     double &db,
                     unsigned &cut) {
    GetNormal(3, xv, xg, R, b, db, cut);
  }

  void GetNormalTri(const std::vector<std::vector<double>> &xv,
                    const std::vector<double> &xg,
                    const double &R,
                    std::vector<double> &b,
                    double &db,
                    unsigned &cut) {
    GetNormal(4, xv, xg, R, b, db, cut);
  }

  // --------------------------------------------------------------------------
  // BATCHED interface -- call these from NonLocal.hpp
  // --------------------------------------------------------------------------

  // Phase A: detect cut for one (element, ball center) pair.
  // Fills cc and returns cut (0/1/2).
  // If cut==1, cc contains everything needed for batch inference and ApplyOffset.
  unsigned DetectCut(const unsigned elType,
                     const std::vector<std::vector<double>> &xv,
                     const std::vector<double> &xg,
                     const double R,
                     unsigned &cut,
                     CutCase &cc) {
    unsigned nve = (elType == 4) ? 3u : 4u;
    unsigned dim = xv.size();

    // Compute element size
    double h = 0.;
    if(nve == 4) {
      double hx = 0.5*(std::fabs(xv[0][2]-xv[0][0])+std::fabs(xv[0][3]-xv[0][1]));
      double hy = 0.5*(std::fabs(xv[1][2]-xv[1][0])+std::fabs(xv[1][3]-xv[1][1]));
      h = std::sqrt(hx*hx + hy*hy);
    } else {
      double hx=(std::fabs(xv[0][1]-xv[0][0])+std::fabs(xv[0][2]-xv[0][1])+std::fabs(xv[0][2]-xv[0][0]))/3.;
      double hy=(std::fabs(xv[1][1]-xv[1][0])+std::fabs(xv[1][2]-xv[1][1])+std::fabs(xv[1][2]-xv[1][0]))/3.;
      h = std::sqrt(hx*hx + hy*hy);
    }
    double eps = 1.0e-10 * h;

    // Node distances
    _dist.assign(nve, 0.);
    _dist0.resize(nve);
    unsigned cnt0 = 0;
    for(unsigned i = 0; i < nve; i++) {
      for(unsigned k = 0; k < dim; k++)
        _dist[i] += (xv[k][i]-xg[k])*(xv[k][i]-xg[k]);
      _dist[i] = std::sqrt(_dist[i]) - R;
      if(std::fabs(_dist[i]) < eps) {
        _dist0[i] = (_dist[i] < 0) ? -eps : eps;
        _dist[i] = 0.; cnt0++;
      } else {
        _dist0[i] = _dist[i];
      }
    }

    if(cnt0 > 0) {
      unsigned cntp = 0;
      for(unsigned i = 0; i < nve; i++) {
        if(_dist[i] > 0) cntp++;
        _dist[i] = _dist0[i];
      }
      if(cntp == 0)          { cut = 0; return cut; }
      if(cntp == nve - cnt0) { cut = 2; return cut; }
    }

    // Edge crossings
    unsigned cnt = 0;
    double theta[2];
    unsigned crossEdge[2]; double crossT[2];
    for(unsigned e = 0; e < nve; e++) {
      unsigned ep1 = (e+1) % nve;
      if(_dist[e] * _dist[ep1] < 0.) {
        double t = 0.5*(1. + (_dist[e]+_dist[ep1])/(_dist[e]-_dist[ep1]));
        if(cnt < 2) {
          crossEdge[cnt] = e; crossT[cnt] = t;
          theta[cnt] = std::atan2((1-t)*xv[1][e]+t*xv[1][ep1]-xg[1],
                                  (1-t)*xv[0][e]+t*xv[0][ep1]-xg[0]);
        }
        cnt++;
      }
    }

    if(cnt == 0) { cut = (_dist[0] < 0) ? 0 : 2; return cut; }
    if(cnt != 2) {
      // Degenerate: fall back to geometric
      BallApproximation::GetNormal(elType, xv, xg, R, _bFallback, _dbFallback, cut);
      return cut;
    }

    cut = 1;

    // Orient arc
    if(theta[0] > theta[1]) {
      std::swap(theta[0], theta[1]);
      std::swap(crossEdge[0], crossEdge[1]);
      std::swap(crossT[0], crossT[1]);
    }
    double DT = theta[1] - theta[0];
    if(DT > M_PI) {
      std::swap(theta[0], theta[1]); theta[1] += 2.*M_PI;
      std::swap(crossEdge[0], crossEdge[1]); std::swap(crossT[0], crossT[1]);
      DT = theta[1] - theta[0];
    }

    // Chord points
    std::vector<double> Pi(dim), Pj(dim);
    for(unsigned k = 0; k < dim; k++) {
      Pi[k] = (1.-crossT[0])*xv[k][crossEdge[0]] + crossT[0]*xv[k][(crossEdge[0]+1)%nve];
      Pj[k] = (1.-crossT[1])*xv[k][crossEdge[1]] + crossT[1]*xv[k][(crossEdge[1]+1)%nve];
    }
    double dx = Pj[0]-Pi[0], dy = Pj[1]-Pi[1];
    double ell = std::sqrt(dx*dx + dy*dy);
    double disc = R*R - ell*ell/4.;
    double sigma = (disc >= 0.) ? R - std::sqrt(disc) : 0.;

    // Inside flags
    std::vector<int> inside(nve);
    for(unsigned i = 0; i < nve; i++) inside[i] = (_dist[i] < 0) ? 1 : 0;

    // Fill CutCase
    cc.nve = nve; cc.elType = elType;
    cc.crossEdge[0] = crossEdge[0]; cc.crossEdge[1] = crossEdge[1];
    cc.crossT[0]    = crossT[0];    cc.crossT[1]    = crossT[1];
    cc.sigma = sigma;
    cc.xv = xv; cc.xg = xg; cc.R = R;

    // Build feature vector
    if(nve == 3) {
      cc.feat.resize(11);
      for(unsigned k = 0; k < 3; k++) {
        double ddx=xv[0][k]-xg[0], ddy=xv[1][k]-xg[1];
        cc.feat[k] = static_cast<float>(std::sqrt(ddx*ddx+ddy*ddy)/R);
      }
      for(unsigned k = 0; k < 3; k++) {
        unsigned kp1=(k+1)%3;
        double ex=xv[0][kp1]-xv[0][k], ey=xv[1][kp1]-xv[1][k];
        cc.feat[3+k] = static_cast<float>(std::sqrt(ex*ex+ey*ey)/R);
      }
      cc.feat[6] = static_cast<float>(ell/R);
      cc.feat[7] = static_cast<float>(sigma/R);
      for(unsigned k = 0; k < 3; k++)
        cc.feat[8+k] = static_cast<float>(inside[k] ? 1.f : -1.f);
    } else {
      cc.feat.resize(14);
      for(unsigned k = 0; k < 4; k++) {
        cc.feat[2*k]   = static_cast<float>((xv[0][k]-xg[0])/R);
        cc.feat[2*k+1] = static_cast<float>((xv[1][k]-xg[1])/R);
      }
      cc.feat[8]  = static_cast<float>(ell/R);
      cc.feat[9]  = static_cast<float>(sigma/R);
      for(unsigned k = 0; k < 4; k++)
        cc.feat[10+k] = static_cast<float>(inside[k] ? 1.f : -1.f);
    }

    return cut;
  }

  // Phase B: batch inference for all cut cases.
  // Returns s_norm values in the same order as cutCases.
  std::vector<double> BatchInfer(const unsigned elType,
                                 const std::vector<CutCase> &cutCases) {
    std::vector<double> results(cutCases.size(), 0.5);
    if(cutCases.empty()) return results;

    bool isTri = (elType == 4);
    if(isTri && !_hasModelTri)   return results;
    if(!isTri && !_hasModelQuad) return results;

    int n_feat = isTri ? 11 : 14;
    int N = static_cast<int>(cutCases.size());

    // Build batch tensor (N, n_feat) as float64
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor batch = torch::zeros({N, n_feat}, opts);
    for(int i = 0; i < N; i++)
      for(int j = 0; j < n_feat; j++)
        batch[i][j] = static_cast<double>(cutCases[i].feat[j]);

        std::cerr << "[BatchInfer] N=" << N << "\n";
    // One forward pass
    torch::NoGradGuard no_grad;
    auto &model = isTri ? _modelTri : _modelQuad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch);
    torch::Tensor output = model.forward(inputs).toTensor();

    for(int i = 0; i < N; i++)
      results[i] = output[i][0].item<double>();

    return results;
  }

  // Phase C: apply offset for one cut case given s_norm, fill (b, db).
  void ApplyOffset(const CutCase &cc,
                   const double s_norm,
                   std::vector<double> &b,
                   double &db) {
    unsigned nve = cc.nve;
    unsigned dim = cc.xv.size();
    double s = s_norm * cc.sigma;

    // Chord crossing points
    std::vector<double> Pi(dim), Pj(dim);
    for(unsigned k = 0; k < dim; k++) {
      Pi[k] = (1.-cc.crossT[0])*cc.xv[k][cc.crossEdge[0]]
             +    cc.crossT[0] *cc.xv[k][(cc.crossEdge[0]+1)%nve];
      Pj[k] = (1.-cc.crossT[1])*cc.xv[k][cc.crossEdge[1]]
             +    cc.crossT[1] *cc.xv[k][(cc.crossEdge[1]+1)%nve];
    }

    // Chord normal, oriented away from ball center
    double chord[2] = {Pj[0]-Pi[0], Pj[1]-Pi[1]};
    double clen = std::sqrt(chord[0]*chord[0]+chord[1]*chord[1]);
    if(clen < 1e-14) {
      BallApproximation::GetNormal(cc.elType, cc.xv, cc.xg, cc.R, b, db, _cutFallback);
      return;
    }
    double d_hat[2] = {chord[0]/clen, chord[1]/clen};
    double n_hat[2] = {-d_hat[1], d_hat[0]};
    double Mx = 0.5*(Pi[0]+Pj[0]), My = 0.5*(Pi[1]+Pj[1]);
    if((cc.xg[0]-Mx)*n_hat[0]+(cc.xg[1]-My)*n_hat[1] > 0.) {
      n_hat[0]=-n_hat[0]; n_hat[1]=-n_hat[1];
    }
    double c_lev = Pi[0]*n_hat[0]+Pi[1]*n_hat[1];

    // Shifted crossings
    auto edge_dot = [&](unsigned e, double nx, double ny, bool isA) -> double {
      unsigned ep1 = (e+1)%nve;
      double Ax = cc.xv[0][e],   Ay = cc.xv[1][e];
      double Bx = cc.xv[0][ep1]-Ax, By = cc.xv[1][ep1]-Ay;
      double B_dot_n = Bx*nx+By*ny;
      double A_dot_n = Ax*nx+Ay*ny;
      if(std::fabs(B_dot_n) < 1e-14) return isA ? cc.crossT[0] : cc.crossT[1];
      return (c_lev+s-A_dot_n)/B_dot_n;
    };
    double ti = edge_dot(cc.crossEdge[0], n_hat[0], n_hat[1], true);
    double tj = edge_dot(cc.crossEdge[1], n_hat[0], n_hat[1], false);

    // Shifted physical midpoint
    std::vector<double> Pi_new(dim), Pj_new(dim), xm(dim);
    for(unsigned k = 0; k < dim; k++) {
      Pi_new[k] = (1.-ti)*cc.xv[k][cc.crossEdge[0]] + ti*cc.xv[k][(cc.crossEdge[0]+1)%nve];
      Pj_new[k] = (1.-tj)*cc.xv[k][cc.crossEdge[1]] + tj*cc.xv[k][(cc.crossEdge[1]+1)%nve];
      xm[k] = 0.5*(Pi_new[k]+Pj_new[k]);
    }

    // Physical normal at new midpoint (toward ball center, matching original convention)
    double dcx=Pj_new[0]-Pi_new[0], dcy=Pj_new[1]-Pi_new[1];
    double dc_len=std::sqrt(dcx*dcx+dcy*dcy);
    if(dc_len < 1e-14) {
      BallApproximation::GetNormal(cc.elType, cc.xv, cc.xg, cc.R, b, db, _cutFallback);
      return;
    }
    double nx=-dcy/dc_len, ny=dcx/dc_len;
    if((cc.xg[0]-xm[0])*nx+(cc.xg[1]-xm[1])*ny < 0.) { nx=-nx; ny=-ny; }
    double a_phys[2] = {nx, ny};

    // Map to reference coordinates
    if(nve == 3) {
      // TRI3: constant Jacobian
      const double &x1=cc.xv[0][0],&x2=cc.xv[0][1],&x3=cc.xv[0][2];
      const double &y1=cc.xv[1][0],&y2=cc.xv[1][1],&y3=cc.xv[1][2];
      double J[2][2] = {{-x1+x2,-x1+x3},{-y1+y2,-y1+y3}};
      double den = x3*y1-x1*y3+x2*J[1][1]-y2*J[0][1];
      double xi0 = (x3*y1-x1*y3+xm[0]*J[1][1]-xm[1]*J[0][1])/den;
      double xi1 = (x1*y2-x2*y1-xm[0]*J[1][0]+xm[1]*J[0][0])/den;
      b.assign(2,0.);
      for(unsigned k=0;k<2;k++) for(unsigned j=0;j<2;j++) b[k]+=J[j][k]*a_phys[j];
      double bN=std::sqrt(b[0]*b[0]+b[1]*b[1]); b[0]/=bN; b[1]/=bN;
      db = -b[0]*xi0-b[1]*xi1;
    } else {
      // QUAD4: Jacobian at element center
      const double &x1=cc.xv[0][0],&x2=cc.xv[0][1],&x3=cc.xv[0][2],&x4=cc.xv[0][3];
      const double &y1=cc.xv[1][0],&y2=cc.xv[1][1],&y3=cc.xv[1][2],&y4=cc.xv[1][3];
      double J00=0.25*(-(x1-x2)+(x4-x3)), J01=0.25*(-(x1-x4)+(x2-x3));
      double J10=0.25*(-(y1-y2)+(y4-y3)), J11=0.25*(-(y1-y4)+(y2-y3));
      b.assign(2,0.);
      b[0]=J00*a_phys[0]+J10*a_phys[1];
      b[1]=J01*a_phys[0]+J11*a_phys[1];
      double bN=std::sqrt(b[0]*b[0]+b[1]*b[1]); b[0]/=bN; b[1]/=bN;
      double detJ=J00*J11-J01*J10;
      double xs=xm[0]-0.25*(x1+x2+x3+x4), ys=xm[1]-0.25*(y1+y2+y3+y4);
      double xi0=( J11*xs-J01*ys)/detJ;
      double xi1=(-J10*xs+J00*ys)/detJ;
      db = -b[0]*xi0-b[1]*xi1;
    }
  }

  // Stored fallback output (used when geometric fallback is needed inside DetectCut)
  std::vector<double> _bFallback;
  double _dbFallback = 0.;
  unsigned _cutFallback = 0;

private:
  torch::jit::script::Module _modelQuad;
  torch::jit::script::Module _modelTri;
  bool _hasModelQuad = false;
  bool _hasModelTri  = false;
  bool _useML        = true;

  std::vector<double> _dist, _dist0;

  double RunInferenceSingle(torch::jit::script::Module &model,
                            const std::vector<float> &feat) {
    int n = static_cast<int>(feat.size());
    auto opts = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor input = torch::zeros({1, n}, opts);
    for(int i = 0; i < n; i++) input[0][i] = static_cast<double>(feat[i]);
    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    return model.forward(inputs).toTensor()[0][0].item<double>();
  }
};

#endif // __femus_GetNormalML_hpp__
