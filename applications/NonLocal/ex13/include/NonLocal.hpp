#ifndef __femus_NonLocal_hpp__
#define __femus_NonLocal_hpp__

#include "GetNormal.hpp"
#include "TetDepthEval.hpp"
#include "QuadDepthEval.hpp"

std::ofstream fout;

class NonLocal {
  public:
    NonLocal() {
      _ballAprx = new BallApproximation();
    };
    ~NonLocal() {
      delete _ballAprx;
    };
    double GetDistance(const std::vector < double>  &x1, const std::vector < double>  &x2) const {
      double distance  = 0.;
      for(unsigned k = 0; k < x1.size(); k++) {
        distance += (x2[k] - x1[k]) * (x2[k] - x1[k]);
      }
      return sqrt(distance);

    };
    virtual double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &size) const = 0;
    virtual void SetKernel(const double  &kappa, const double &delta, const double &eps) = 0;
    const double & GetKernel() const {
      return _kernel;
    };
    virtual double GetArea(const double &delta, const double &eps) const = 0;
    virtual double GetGamma(const double &d) const = 0;
    virtual double GetGamma(const std::vector < double>  &x1, const std::vector < double>  &x2) const = 0;

    virtual bool KernelIsConstant() const { return true; }


    void ZeroLocalQuantities(const unsigned &nDof1, const Region &region2, const unsigned &levelMax1);

    void Assembly1(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                   const OctTreeElement &octTreeElement1, RefineElement &element1,
                   const Region &region2, const std::vector <unsigned> &jelIndexF, const vector < double >  &solu1,
                   const double &kappa, const double &delta, const bool &printMesh);

    double Assembly2(const RefineElement &element1, const Region &region2, const std::vector<unsigned> & jelIndex,
                     const unsigned &nDof1, const vector < double > &xg1,
                     const double &twoWeigh1Kernel, const vector < double > &phi1, const vector < double >  &solu1,
                     const double &delta, const bool &printMesh);

    void AssemblyCutFem1(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                         const OctTreeElement &octTreeElement1, const OctTreeElement &octTreeElement1CF,
                         RefineElement &element1, Region &region2, const std::vector <unsigned> &jelIndexF, const vector < double >  &solu1,
                         const double &kappa, const double &delta, const bool &printMesh);

    void AssemblyCutFemI2(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                          const OctTreeElement &octTreeElement1, const OctTreeElement &octTreeElement1CF,
                          RefineElement &element1, Region &region2, const std::vector <unsigned> &jelIndexF, const vector < double >  &solu1,
                          const double &kappa, const double &delta, const bool &printMesh);


    void AssemblyCutFem2(const std::vector <double> &phi1W1,
                         const double &solu1W1W2,
                         const double &W1W2,
                         const unsigned &jel,
                         const unsigned &nDof2,
                         const double *phi2,
                         const double &solu2gW1W2,
                         const double &W2);

    double GetSmoothTestFunction(const double &dg1, const double &eps);

    std::vector < double > & GetRes2(const unsigned &jel) {
      return _res2[jel];
    };

    std::vector < double > & GetJac21(const unsigned &jel) {
      return _jac21[jel];
    };
    std::vector < double > & GetJac22(const unsigned &jel) {
      return _jac22[jel];
    };

  private:
    std::vector < std::vector < double > > _res2;
    std::vector < std::vector < double > > _jac21;
    std::vector < std::vector < double > > _jac22;

    std::vector <unsigned> _jelIndexI;
    std::vector < std::vector <unsigned> >_jelIndexR;

    // per-pair gate: surviving Gauss-point lists, parallel to the jel lists
    std::vector < std::vector <unsigned> > _jgIndexRoot;
    std::vector < std::vector < std::vector <unsigned> > > _jgIndexR;

    void AssemblyCutFem1Gated(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                              const OctTreeElement &octTreeElement1, const OctTreeElement &octTreeElement1CF,
                              RefineElement &element1, Region &region2,
                              const std::vector <unsigned> &jelIndexF,
                              const std::vector < std::vector <unsigned> > &jgIndexF,
                              const vector < double >  &solu1,
                              const double &kappa, const double &delta, const bool &printMesh);

    BallApproximation *_ballAprx;
    std::vector<double> _a;

    std::vector< std::vector < double> > _xg1;
    std::vector< std::vector < double> > _xg1CF;

    std::vector < double> _weight1;
    std::vector < double> _weight1CF;
    std::vector<double> _eqPolyWeight;

    std::vector <double> _phi1W1;
    std::vector <double> _phi1W1CF;

    std::vector<double> _phi1W1g;

    std::vector < double > _phi1W1W2;
    std::vector < double > _phi2W1W2;

    std::vector<double>::iterator _jac21It;
    std::vector<double>::iterator _jac21End;
    std::vector<double>::iterator _jac22It;
    std::vector<double>::iterator _res2It;
    std::vector<double>::const_iterator _phi1W1It;
    std::vector<double>::iterator _phi1W1W2It;
    std::vector<double>::iterator _phi1W1W2Begin;
    std::vector<double>::iterator _phi1W1W2End;
    std::vector<double>::iterator _phi2W1W2It;
    std::vector<double>::iterator _phi2W1W2Begin;
    std::vector<double>::iterator _phi2W1W2End;
    const double *_phi2pt;


    double _d;
    unsigned _cut;

    void PrintElement(const std::vector < std::vector < double> > &xv, const RefineElement &refineElement);

  protected:
    double _kernel;

};

void NonLocal::ZeroLocalQuantities(const unsigned &nDof1, const Region &region2, const unsigned &levelMax1) {
  _res2.resize(region2.size());
  _jac21.resize(region2.size());
  _jac22.resize(region2.size());

  for(unsigned jel = 0; jel < region2.size(); jel++) {
    unsigned nDof2 = region2.GetDofNumber(jel);
    _jac21[jel].assign(nDof2 * nDof1, 0.);
    _jac22[jel].assign(nDof2 * nDof2, 0.);
    _res2[jel].assign(nDof2, 0.);
  }

  _jelIndexI.reserve(region2.size());
  _jelIndexR.resize(levelMax1);
  _jgIndexR.resize(levelMax1);
  for(unsigned level = 0; level < levelMax1; level++) {
    _jelIndexR[level].reserve(region2.size());
  }

}

void NonLocal::Assembly1(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                         const OctTreeElement &octTreeElement1, RefineElement &element1,
                         const Region &region2, const std::vector <unsigned> &jelIndexF, const vector < double >  &solu1,
                         const double &kappa, const double &delta, const bool &printMesh) {


  if(level < levelMin1) {
    element1.BuildElement1Prolongation(level, iFather);
    for(unsigned i = 0; i < element1.GetNumberOfChildren(); i++) {
      Assembly1(level + 1, levelMin1, levelMax1, i,
                *octTreeElement1.GetElement(std::vector<unsigned> {i}), element1, region2, jelIndexF,
                solu1, kappa, delta, printMesh);
    }
  }
  else if(level == levelMax1 - 1) {
    const unsigned &dim = element1.GetDimension();
    const std::vector < std::vector <double> >  &xv1 = element1.GetElement1NodeCoordinates(level, iFather);
    double eps = element1.GetEps();
    const unsigned &nDof1 = element1.GetNumberOfNodes();
    //double kernel = this->GetKernel(kappa, delta, eps);

    const elem_type *fem1 = element1.GetFem1();

    std::vector < double> xg1(dim);
    double weight1;
    const double *phi1;

    const std::vector < std::vector < double> > & phi1F = octTreeElement1.GetGaussShapeFunctions();

    for(unsigned ig = 0; ig < fem1->GetGaussPointNumber(); ig++) {
      fem1->GetGaussQuantities(xv1, ig, weight1, phi1);
      xg1.assign(dim, 0.);
      for(unsigned k = 0; k < dim; k++) {
        for(unsigned i = 0; i < nDof1; i++) {
          xg1[k] += xv1[k][i] * phi1[i];
        }
      }

      Assembly2(element1, region2, jelIndexF, nDof1, xg1, 2. * weight1 * _kernel,
                phi1F[ig], solu1, delta, printMesh);
    }
  }
  else {
    const unsigned &dim = element1.GetDimension();
    const std::vector < std::vector <double> >  &xv1 = element1.GetElement1NodeCoordinates(level, iFather);
    double eps = element1.GetEps();

    _jelIndexR[level].resize(0);
    _jelIndexI.resize(0);

    std::vector < std::pair<std::vector<double>::const_iterator, std::vector<double>::const_iterator> > x1MinMax(dim);
    for(unsigned k = 0; k < dim; k++) {
      x1MinMax[k] = std::minmax_element(xv1[k].begin(), xv1[k].end());
    }

    std::vector < double > dmM2(dim);
    std::vector < double > dMm2(dim);
    std::vector < double > dist(pow(2, dim));

    for(unsigned j = 0; j < jelIndexF.size(); j++) {
      unsigned jel = jelIndexF[j];
      const std::vector<std::vector<double>>& x2MinMax = region2.GetMinMax(jel);

      for(unsigned k = 0; k < dim; k++) {
        dmM2[k] = (*(x1MinMax[k].first) - x2MinMax[k][1]) * (*(x1MinMax[k].first) - x2MinMax[k][1]);
        dMm2[k] = (*(x1MinMax[k].second) - x2MinMax[k][0]) * (*(x1MinMax[k].second) - x2MinMax[k][0]);
      }

      if(dim == 2) {
        dist[0] = sqrt(dmM2[0] + dmM2[1]);
        dist[1] = sqrt(dMm2[0] + dmM2[1]);
        dist[2] = sqrt(dmM2[0] + dMm2[1]);
        dist[3] = sqrt(dMm2[0] + dMm2[1]);
      }
      else if(dim == 3) {
        dist[0] = sqrt(dmM2[0] + dmM2[1] + dmM2[2]);
        dist[1] = sqrt(dMm2[0] + dmM2[1] + dmM2[2]);
        dist[2] = sqrt(dmM2[0] + dMm2[1] + dmM2[2]);
        dist[3] = sqrt(dMm2[0] + dMm2[1] + dmM2[2]);

        dist[4] = sqrt(dmM2[0] + dmM2[1] + dMm2[2]);
        dist[5] = sqrt(dMm2[0] + dmM2[1] + dMm2[2]);
        dist[6] = sqrt(dmM2[0] + dMm2[1] + dMm2[2]);
        dist[7] = sqrt(dMm2[0] + dMm2[1] + dMm2[2]);
      }

      if(*std::max_element(dist.begin(), dist.end()) < delta - eps) {
        _jelIndexI.resize(_jelIndexI.size() + 1, jel);
      }
      else {
        _jelIndexR[level].resize(_jelIndexR[level].size() + 1, jel);
      }
    }
    if(_jelIndexI.size() > 0) {
      const unsigned &nDof1 = element1.GetNumberOfNodes();
      const elem_type *fem1 = element1.GetFem1();

      std::vector < double> xg1(dim);
      double weight1;
      const double *phi1;

      const std::vector < std::vector < double> > & phi1F = octTreeElement1.GetGaussShapeFunctions();

      for(unsigned ig = 0; ig < fem1->GetGaussPointNumber(); ig++) {
        fem1->GetGaussQuantities(xv1, ig, weight1, phi1);
        xg1.assign(dim, 0.);
        for(unsigned k = 0; k < dim; k++) {
          for(unsigned i = 0; i < nDof1; i++) {
            xg1[k] += xv1[k][i] * phi1[i];
          }
        }
        Assembly2(element1, region2, _jelIndexI, nDof1, xg1, 2. * weight1 * _kernel,
                  phi1F[ig], solu1, delta, printMesh);
      }
    }
    if(_jelIndexR[level].size() > 0) {
      element1.BuildElement1Prolongation(level, iFather);
      for(unsigned i = 0; i < element1.GetNumberOfChildren(); i++) {
        Assembly1(level + 1, levelMin1, levelMax1, i,
                  *octTreeElement1.GetElement(std::vector<unsigned> {i}), element1, region2, _jelIndexR[level],
                  solu1, kappa, delta, printMesh);
      }
    }
  }
}






double NonLocal::Assembly2(const RefineElement & element1, const Region & region2, const std::vector<unsigned> &jelIndex,
                           const unsigned & nDof1, const vector < double > &xg1,
                           const double & twoWeigh1Kernel, const vector < double > &phi1, const vector < double >  &solu1,
                           const double & delta, const bool & printMesh) {

  double area = 0.;

  double solu1g = 0.;
  for(unsigned i = 0; i < nDof1; i++) {
    solu1g += solu1[i] * phi1[i];
  }

  const double *phi2;
  const double *phi2pt;
  double U;

  std::vector< double > mCphi2iSum;

  const double& eps = element1.GetEps();

  for(unsigned jj = 0; jj < jelIndex.size(); jj++) {

    unsigned jel = jelIndex[jj];

    const unsigned &dim = region2.GetDimension(jel);
    const std::vector<std::vector<double>>& x2MinMax = region2.GetMinMax(jel);

    bool coarseIntersectionTest = true;
    for(unsigned k = 0; k < dim; k++) {
      if((xg1[k]  - x2MinMax[k][1]) > delta + eps  || (x2MinMax[k][0] - xg1[k]) > delta + eps) {
        coarseIntersectionTest = false;
        break;
      }
    }

    if(coarseIntersectionTest) {

      const unsigned &nDof2 = region2.GetDofNumber(jel);
      const elem_type *fem = region2.GetFem(jel);
      const std::vector <double >  &solu2g = region2.GetGaussSolution(jel);
      const std::vector <double >  &weight2 = region2.GetGaussWeight(jel);
      const std::vector < std::vector <double> >  &xg2 = region2.GetGaussCoordinates(jel);

      mCphi2iSum.assign(nDof2, 0.);

      for(unsigned jg = 0; jg < fem->GetGaussPointNumber(); jg++) {
        phi2 = fem->GetPhi(jg);
        U = element1.GetSmoothStepFunction(this->GetInterfaceDistance(xg1, xg2[jg], delta));
        if(U > 0.) {
          double C =  U * GetGamma(xg1, xg2[jg]) *  weight2[jg] * twoWeigh1Kernel;
          double *jac22pt = &_jac22[jel][0];
          for(unsigned i = 0; i < nDof2; i++) {
            double cPhi2i = C * phi2[i];
            mCphi2iSum[i] -= cPhi2i;
            unsigned j = 0;
            for(phi2pt = phi2; j < nDof2; j++, phi2pt++, jac22pt++) {
              *jac22pt -= cPhi2i * (*phi2pt);
            }
            _res2[jel][i] += cPhi2i * solu2g[jg];
          }
        }//end if U > 0.
      }//end jg loop

      unsigned ijIndex = 0;
      for(unsigned i = 0; i < nDof2; i++) {
        for(unsigned j = 0; j < nDof1; j++, ijIndex++) {
          _jac21[jel][ijIndex] -= mCphi2iSum[i] * phi1[j];
        }
        _res2[jel][i] += mCphi2iSum[i] * solu1g;
      }
    }
  }
  return area;
}





void NonLocal::AssemblyCutFemI2(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                               const OctTreeElement &octTreeElement1, const OctTreeElement &octTreeElement1CF,
                               RefineElement &element1, Region &region2,
                               const std::vector <unsigned> &jelIndexF, const vector < double >  &solu1,
                               const double &kappa, const double &delta, const bool &printMesh) {

  unsigned levelMax1Loc = levelMax1;
  if(level == 0 && element1.GetElementType() == 1) {   // TET only
    const unsigned &dim = element1.GetDimension();
    const std::vector<std::vector<double>> &xv0 = element1.GetElement1NodeCoordinates(0, 0);
    std::vector<std::pair<std::vector<double>::const_iterator,
    std::vector<double>::const_iterator>> mm(dim);
    for(unsigned k = 0; k < dim; k++) mm[k] = std::minmax_element(xv0[k].begin(), xv0[k].end());
    double lpSum = 0.; unsigned lpCnt = 0;
    // for(unsigned jj = 0; jj < jelIndexF.size() && lpMax + 1 < (int)levelMax1; jj++) {
    for(unsigned jj = 0; jj < jelIndexF.size(); jj++) {
      const std::vector<std::vector<double>> &xg2 = region2.GetGaussCoordinates(jelIndexF[jj]);
      // for(unsigned jg = 0; jg < xg2.size() && lpMax + 1 < (int)levelMax1; jg++) {
      for(unsigned jg = 0; jg < xg2.size(); jg++) {
        bool near = true;
        for(unsigned k = 0; k < dim; k++)
          if((xg2[jg][k] - *(mm[k].second)) > delta || (*(mm[k].first) - xg2[jg][k]) > delta) { near = false; break; }
          // if(near) { lpSum += PredictTetDepth(xv0, xg2[jg], delta); lpCnt++; }
          if(near) { if(PredictTetDepth(xv0, xg2[jg], delta) >= 1) lpSum += 1.; lpCnt++; }
      }
    }
    // levelMax1Loc = std::max(levelMin1 + 1, std::min((unsigned)(lpMax + 1), levelMax1));
    // int lpAgg = (lpCnt > 0) ? (int)std::ceil(lpSum/lpCnt + 0.3) : 0;
    // levelMax1Loc = std::max(levelMin1 + 1, std::min((unsigned)(lpAgg + 1), levelMax1));
    double hardFrac = (lpCnt > 0) ? lpSum/lpCnt : 0.;
    unsigned cap = (hardFrac > 0.05) ? 3u : 2u;      // knob: 0.05
    levelMax1Loc = std::min(cap, levelMax1);

  }

  if(level < levelMin1) {
    element1.BuildElement1Prolongation(level, iFather);
    for(unsigned i = 0; i < element1.GetNumberOfChildren(); i++) {
      AssemblyCutFemI2(level + 1, levelMin1, levelMax1Loc, i,
                      *octTreeElement1.GetElement(std::vector<unsigned> {i}),
                      *octTreeElement1CF.GetElement(std::vector<unsigned> {i}),
                      element1, region2, jelIndexF,
                      solu1, kappa, delta, printMesh);
    }
  }
  else if(level == levelMax1Loc - 1) {
    const unsigned &dim = element1.GetDimension();
    const std::vector < std::vector <double> >  &xv1 = element1.GetElement1NodeCoordinates(level, iFather);

    const unsigned &nDof1 = element1.GetNumberOfNodes();
    const elem_type *fem1 = element1.GetFem1();
    const elem_type *fem1CF = element1.GetFem1CF();

    const unsigned &ng1 = fem1->GetGaussPointNumber();
    const unsigned &ng1CF = fem1CF->GetGaussPointNumber();
    _xg1.assign(ng1, std::vector<double>(dim, 0));
    _weight1.resize(ng1);
    for(unsigned ig = 0; ig < ng1; ig++) {
      const double *phi;
      fem1->GetGaussQuantities(xv1, ig, _weight1[ig], phi);
      for(unsigned i = 0; i < nDof1; i++) {
        for(unsigned k = 0; k < dim; k++) {
          _xg1[ig][k] += phi[i] * xv1[k][i];
        }
      }
    }

    _weight1CF.resize(ng1CF);
    _xg1CF.assign(ng1CF, std::vector<double>(dim, 0));
    for(unsigned ig = 0; ig < ng1CF; ig++) {
      const double *phi;
      fem1CF->GetGaussQuantities(xv1, ig, _weight1CF[ig], phi);
      for(unsigned i = 0; i < nDof1; i++) {
        for(unsigned k = 0; k < dim; k++) {
          _xg1CF[ig][k] += phi[i] * xv1[k][i];
        }
      }
    }

    //BEGIN NEW STUFF

    std::vector < std::pair<std::vector<double>::const_iterator, std::vector<double>::const_iterator> > x1MinMax(dim);
    for(unsigned k = 0; k < dim; k++) {
      x1MinMax[k] = std::minmax_element(xv1[k].begin(), xv1[k].end());
    }

    for(unsigned jj = 0; jj < jelIndexF.size(); jj++) {
      unsigned jel = jelIndexF[jj];
      const std::vector<std::vector<double>>& x2MinMax = region2.GetMinMax(jel);

      const elem_type *fem2 = region2.GetFem(jel);
      const std::vector < std::vector <double> >  &xg2 = region2.GetGaussCoordinates(jel);

      for(unsigned jg = 0; jg < fem2->GetGaussPointNumber(); jg++) {

        bool coarseIntersectionTest = true;
        for(unsigned k = 0; k < dim; k++) {
          if((xg2[jg][k]  - * (x1MinMax[k].second)) > delta || (*(x1MinMax[k].first) - xg2[jg][k]) > delta) {  // this can be improved with the l2 norm
            coarseIntersectionTest = false;
            break;
          }
        }

        if(coarseIntersectionTest) {
          _ballAprx->GetNormal(element1.GetElementType(), xv1, xg2[jg], delta, _a, _d, _cut);

          if(_cut == 0) { //interior element
            double d2W1 = 0.;
            for(unsigned ig = 0; ig < ng1; ig++) {
              double d2 = 0.;
              for(unsigned k = 0; k < dim; k++) {
                d2 += (xg2[jg][k] - _xg1[ig][k]) * (xg2[jg][k] - _xg1[ig][k]);
              }
              // d2W1 += d2 * _weight1[ig];
              d2W1 += d2 * GetGamma(_xg1[ig], xg2[jg]) * _weight1[ig];
              //d2W1 += _weight1[ig];
            }
            region2.AddI2(jel, jg, d2W1);
          }
          else if(_cut == 1) { //cut element
            element1.GetCutFem()->clear();
//             element1.GetCutFem()->GetWeightWithMap(0, _a, _d, _eqPolyWeight);
//             (*element1.GetCutFem())(0, _a, _d, _eqPolyWeight);
            element1.GetCDweight()->GetWeight(_a, _d, _eqPolyWeight);

            double d2W1CF = 0.;
            for(unsigned ig = 0; ig < ng1CF; ig++) {
              double d2 = 0.;
              for(unsigned k = 0; k < dim; k++) {
                d2 += (xg2[jg][k] - _xg1CF[ig][k]) * (xg2[jg][k] - _xg1CF[ig][k]);
              }
              // d2W1CF += d2 * _weight1CF[ig] * _eqPolyWeight[ig];
              d2W1CF += d2 * GetGamma(_xg1CF[ig], xg2[jg]) * _weight1CF[ig] * _eqPolyWeight[ig];
              //d2W1CF += _weight1CF[ig] * _eqPolyWeight[ig];
            }
            region2.AddI2(jel, jg, d2W1CF);
          }
        }
      }
    }
  }
  else {
    const unsigned &dim = element1.GetDimension();
    const std::vector < std::vector <double> >  &xv1 = element1.GetElement1NodeCoordinates(level, iFather);

    _jelIndexR[level].resize(0);
    _jelIndexI.resize(0);

    std::vector < std::pair<std::vector<double>::const_iterator, std::vector<double>::const_iterator> > x1MinMax(dim);
    for(unsigned k = 0; k < dim; k++) {
      x1MinMax[k] = std::minmax_element(xv1[k].begin(), xv1[k].end());
    }

    std::vector < double > dmM2(dim);
    std::vector < double > dMm2(dim);
    std::vector < double > dist(pow(2, dim));

    for(unsigned j = 0; j < jelIndexF.size(); j++) {
      unsigned jel = jelIndexF[j];
      const std::vector<std::vector<double>>& x2MinMax = region2.GetMinMax(jel);

      for(unsigned k = 0; k < dim; k++) {
        dmM2[k] = (*(x1MinMax[k].first) - x2MinMax[k][1]) * (*(x1MinMax[k].first) - x2MinMax[k][1]);
        dMm2[k] = (*(x1MinMax[k].second) - x2MinMax[k][0]) * (*(x1MinMax[k].second) - x2MinMax[k][0]);
      }

      if(dim == 2) {
        dist[0] = sqrt(dmM2[0] + dmM2[1]);
        dist[1] = sqrt(dMm2[0] + dmM2[1]);
        dist[2] = sqrt(dmM2[0] + dMm2[1]);
        dist[3] = sqrt(dMm2[0] + dMm2[1]);
      }
      else if(dim == 3) {
        dist[0] = sqrt(dmM2[0] + dmM2[1] + dmM2[2]);
        dist[1] = sqrt(dMm2[0] + dmM2[1] + dmM2[2]);
        dist[2] = sqrt(dmM2[0] + dMm2[1] + dmM2[2]);
        dist[3] = sqrt(dMm2[0] + dMm2[1] + dmM2[2]);

        dist[4] = sqrt(dmM2[0] + dmM2[1] + dMm2[2]);
        dist[5] = sqrt(dMm2[0] + dmM2[1] + dMm2[2]);
        dist[6] = sqrt(dmM2[0] + dMm2[1] + dMm2[2]);
        dist[7] = sqrt(dMm2[0] + dMm2[1] + dMm2[2]);
      }

      // if(*std::max_element(dist.begin(), dist.end()) < delta) {
      if(*std::max_element(dist.begin(), dist.end()) < delta && KernelIsConstant()) {
        _jelIndexI.resize(_jelIndexI.size() + 1, jel);
      }
      else {
        _jelIndexR[level].resize(_jelIndexR[level].size() + 1, jel);
      }
    }
    if(_jelIndexI.size() > 0) {

      const unsigned &dim = element1.GetDimension();
      const std::vector < std::vector <double> >  &xv1 = element1.GetElement1NodeCoordinates(level, iFather);

      const unsigned &nDof1 = element1.GetNumberOfNodes();
      const elem_type *fem1 = element1.GetFem1();

      //these are the shape functions of iel evaluated in the gauss points of the refined elements
      const unsigned &ng1 = fem1->GetGaussPointNumber();

      _xg1.assign(ng1, std::vector<double>(dim, 0));
      _weight1.resize(ng1);
      for(unsigned ig = 0; ig < ng1; ig++) {
        const double *phi;
        fem1->GetGaussQuantities(xv1, ig, _weight1[ig], phi);
        for(unsigned i = 0; i < nDof1; i++) {
          for(unsigned k = 0; k < dim; k++) {
            _xg1[ig][k] += phi[i] * xv1[k][i];
          }
        }
      }

      for(unsigned jj = 0; jj < _jelIndexI.size(); jj++) {
        unsigned jel = _jelIndexI[jj];
        const elem_type *fem2 = region2.GetFem(jel);

        const std::vector < std::vector <double> >  &xg2 = region2.GetGaussCoordinates(jel);

        for(unsigned jg = 0; jg < fem2->GetGaussPointNumber(); jg++) {
          double d2W1 = 0.;
          for(unsigned ig = 0; ig < ng1; ig++) {
            double d2 = 0.;
            for(unsigned k = 0; k < dim; k++) {
              d2 += (xg2[jg][k] - _xg1[ig][k]) * (xg2[jg][k] - _xg1[ig][k]);
            }
            d2W1 += d2 * _weight1[ig];
            //d2W1 += _weight1[ig];
          }
          region2.AddI2(jel, jg, d2W1);
        }
      }
    }
    if(_jelIndexR[level].size() > 0) {
      element1.BuildElement1Prolongation(level, iFather);
      for(unsigned i = 0; i < element1.GetNumberOfChildren(); i++) {
        AssemblyCutFemI2(level + 1, levelMin1, levelMax1Loc, i,
                        *octTreeElement1.GetElement(std::vector<unsigned> {i}),
                        *octTreeElement1CF.GetElement(std::vector<unsigned> {i}),
                        element1, region2, _jelIndexR[level],
                        solu1, kappa, delta, printMesh);
      }
    }
  }
}
























void NonLocal::AssemblyCutFem1(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                               const OctTreeElement &octTreeElement1, const OctTreeElement &octTreeElement1CF,
                               RefineElement &element1, Region &region2,
                               const std::vector <unsigned> &jelIndexF, const vector < double >  &solu1,
                               const double &kappa, const double &delta, const bool &printMesh) {
  // entry wrapper: all Gauss points of every jel start active
  _jgIndexRoot.resize(jelIndexF.size());
  for(unsigned jj = 0; jj < jelIndexF.size(); jj++) {
    const unsigned ng2 = region2.GetFem(jelIndexF[jj])->GetGaussPointNumber();
    _jgIndexRoot[jj].resize(ng2);
    for(unsigned jg = 0; jg < ng2; jg++) _jgIndexRoot[jj][jg] = jg;
  }
  AssemblyCutFem1Gated(level, levelMin1, levelMax1, iFather, octTreeElement1, octTreeElement1CF,
                       element1, region2, jelIndexF, _jgIndexRoot, solu1, kappa, delta, printMesh);
}


void NonLocal::AssemblyCutFem1Gated(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                                    const OctTreeElement &octTreeElement1, const OctTreeElement &octTreeElement1CF,
                                    RefineElement &element1, Region &region2,
                                    const std::vector <unsigned> &jelIndexF,
                                    const std::vector < std::vector <unsigned> > &jgIndexF,
                                    const vector < double >  &solu1,
                                    const double &kappa, const double &delta, const bool &printMesh) {

  // per-pair gate configuration (env; GATE=0 reproduces the ungated code path)
  static const int    gateOn     = [](){ const char *s = getenv("GATE");        return s ? atoi(s) : 0;    }();
  static const double gateEps    = [](){ const char *s = getenv("GATE_EPS");    return s ? atof(s) : 1e-3; }();
  static const double gateMargin = [](){ const char *s = getenv("GATE_MARGIN"); return s ? atof(s) : 0.3;  }();

  if(gateOn != 0 && !KernelIsConstant()) {
    std::cerr << "[FRAC] ERROR: the ML gate is not supported with a non-constant kernel yet "
    "(intermediate-level resolutions bypass the gamma weighting). Run with GATE=0."
    << std::endl;
    abort();
  }

  static long _gateDrop[16] = {0}, _gateInt[16] = {0}, _gateCut[16] = {0}, _gateProp[16] = {0};
  static long _gateLeafPairs = 0;
  static struct _GateRep { ~_GateRep() {
    std::cout << "[GATE-HIST] level: dropAABB/resInterior/resCut/propagate" << std::endl;
    for(int l = 0; l < 16; l++)
      if(_gateDrop[l] || _gateInt[l] || _gateCut[l] || _gateProp[l])
        std::cout << "[GATE-HIST]   L" << l << ": " << _gateDrop[l] << "/" << _gateInt[l]
                  << "/" << _gateCut[l] << "/" << _gateProp[l] << std::endl;
    std::cout << "[GATE-HIST] leaf pairs processed: " << _gateLeafPairs << std::endl;
  } } _gateRep;

  // ---- FRAC probe: raw gamma-moments about a designated y (active if FRAC_PROBE_R>0)
  static const double probeR = [](){ const char *s = getenv("FRAC_PROBE_R"); return s ? atof(s) : 0.; }();
  static const double probeX = [](){ const char *s = getenv("FRAC_PROBE_X"); return s ? atof(s) : 0.; }();
  static const double probeY = [](){ const char *s = getenv("FRAC_PROBE_Y"); return s ? atof(s) : 0.; }();
  struct _FP { double S0 = 0., S2 = 0., S4 = 0., I2v = 0., x = 0., y = 0.; };
  static std::map<std::pair<unsigned, unsigned>, _FP> _probe;
  static struct _FPRep { ~_FPRep() {
    if(_probe.empty()) return;
    std::cout.precision(12);
    std::cout << "[FRAC-PROBE] jel jg | x y | S0 S2 S4 | I2" << std::endl;
    for(auto &p : _probe)
      std::cout << "[FRAC-PROBE] " << p.first.first << " " << p.first.second
                << " | " << p.second.x << " " << p.second.y
                << " | " << p.second.S0 << " " << p.second.S2 << " " << p.second.S4
                << " | " << p.second.I2v << std::endl;
  } } _fpRep;

  unsigned levelMax1Loc = levelMax1;
  if(level == 0 && element1.GetElementType() == 1) {   // TET only
    const unsigned &dim = element1.GetDimension();
    const std::vector<std::vector<double>> &xv0 = element1.GetElement1NodeCoordinates(0, 0);
    std::vector<std::pair<std::vector<double>::const_iterator,
    std::vector<double>::const_iterator>> mm(dim);
    for(unsigned k = 0; k < dim; k++) mm[k] = std::minmax_element(xv0[k].begin(), xv0[k].end());
    double lpSum = 0.; unsigned lpCnt = 0;
    for(unsigned jj = 0; jj < jelIndexF.size(); jj++) {
      const std::vector<std::vector<double>> &xg2 = region2.GetGaussCoordinates(jelIndexF[jj]);
      for(unsigned jg = 0; jg < xg2.size(); jg++) {
        bool near = true;
        for(unsigned k = 0; k < dim; k++)
          if((xg2[jg][k] - *(mm[k].second)) > delta || (*(mm[k].first) - xg2[jg][k]) > delta) { near = false; break; }
          if(near) { if(PredictTetDepth(xv0, xg2[jg], delta) >= 1) lpSum += 1.; lpCnt++; }
      }
    }
    double hardFrac = (lpCnt > 0) ? lpSum/lpCnt : 0.;
    unsigned cap = (hardFrac > 0.05) ? 3u : 2u;      // knob: 0.05
    levelMax1Loc = std::min(cap, levelMax1);

    static long _capHist[8] = {0};
    static struct _CapRep { ~_CapRep(){ std::cout << "[CAP-HIST] ";
      for(int i=0;i<8;i++) if(_capHist[i]) std::cout << i << ":" << _capHist[i] << "  ";
      std::cout << std::endl; } } _capRep;
      _capHist[levelMax1Loc]++;
  }


  if(level < levelMin1) {
    element1.BuildElement1Prolongation(level, iFather);
    for(unsigned i = 0; i < element1.GetNumberOfChildren(); i++) {
      AssemblyCutFem1Gated(level + 1, levelMin1, levelMax1Loc, i,
                           *octTreeElement1.GetElement(std::vector<unsigned> {i}),
                           *octTreeElement1CF.GetElement(std::vector<unsigned> {i}),
                           element1, region2, jelIndexF, jgIndexF,
                           solu1, kappa, delta, printMesh);
    }
  }
  else if(level == levelMax1Loc - 1) {
    const unsigned &dim = element1.GetDimension();
    const std::vector < std::vector <double> >  &xv1 = element1.GetElement1NodeCoordinates(level, iFather);

    const unsigned &nDof1 = element1.GetNumberOfNodes();
    const elem_type *fem1 = element1.GetFem1();
    const elem_type *fem1CF = element1.GetFem1CF();

    //these are the shape functions of iel evaluated in the gauss points of the refined elements
    const std::vector < std::vector < double> > & phi1 = octTreeElement1.GetGaussShapeFunctions();
    const std::vector < std::vector < double> > & phi1CF = octTreeElement1CF.GetGaussShapeFunctions();

    const unsigned &ng1 = fem1->GetGaussPointNumber();
    const unsigned &ng1CF = fem1CF->GetGaussPointNumber();

    double W1 = 0.;
    _phi1W1.assign(nDof1, 0.);
    _xg1.assign(ng1, std::vector<double>(dim, 0));
    _weight1.resize(ng1);
    for(unsigned ig = 0; ig < ng1; ig++) {
      const double *phi;
      fem1->GetGaussQuantities(xv1, ig, _weight1[ig], phi);
      W1 += _weight1[ig];
      for(unsigned i = 0; i < nDof1; i++) {
        _phi1W1[i] += phi1[ig][i] * _weight1[ig];
        for(unsigned k = 0; k < dim; k++) {
          _xg1[ig][k] += phi[i] * xv1[k][i];
        }
      }
    }
    double solu1W1 = 0.;
    for(unsigned i = 0; i < nDof1; i++) {
      solu1W1 += solu1[i] * _phi1W1[i];
    }

    _weight1CF.resize(ng1CF);
    _xg1CF.assign(ng1CF, std::vector<double>(dim, 0));
    for(unsigned ig = 0; ig < ng1CF; ig++) {
      const double *phi;
      fem1CF->GetGaussQuantities(xv1, ig, _weight1CF[ig], phi);
      for(unsigned i = 0; i < nDof1; i++) {
        for(unsigned k = 0; k < dim; k++) {
          _xg1CF[ig][k] += phi[i] * xv1[k][i];
        }
      }
    }

    //BEGIN NEW STUFF

    std::vector < std::pair<std::vector<double>::const_iterator, std::vector<double>::const_iterator> > x1MinMax(dim);
    for(unsigned k = 0; k < dim; k++) {
      x1MinMax[k] = std::minmax_element(xv1[k].begin(), xv1[k].end());
    }

    for(unsigned jj = 0; jj < jelIndexF.size(); jj++) {
      unsigned jel = jelIndexF[jj];
      const std::vector<std::vector<double>>& x2MinMax = region2.GetMinMax(jel);

      const elem_type *fem2 = region2.GetFem(jel);

      const std::vector <double >  &solu2g = region2.GetGaussSolution(jel);
      const std::vector <double >  &weight2 = region2.GetGaussWeight(jel);
      const std::vector < std::vector <double> >  &xg2 = region2.GetGaussCoordinates(jel);
      const std::vector<double>& I2 = region2.GetI2(jel);

      const std::vector <unsigned> &jgList = jgIndexF[jj];
      for(unsigned jgi = 0; jgi < jgList.size(); jgi++) {
        const unsigned jg = jgList[jgi];
        _gateLeafPairs++;

        bool coarseIntersectionTest = true;
        for(unsigned k = 0; k < dim; k++) {
          if((xg2[jg][k]  - * (x1MinMax[k].second)) > delta || (*(x1MinMax[k].first) - xg2[jg][k]) > delta) {  // this can be improved with the l2 norm
            coarseIntersectionTest = false;
            break;
          }
        }

        if(coarseIntersectionTest) {
          _ballAprx->GetNormal(element1.GetElementType(), xv1, xg2[jg], delta, _a, _d, _cut);

          // if(_cut == 0) { //interior element
          //   double W2 = 2. * weight2[jg] * _kernel * I2[jg];
          //   double W1W2 = W1 * W2;
          //   AssemblyCutFem2(_phi1W1, solu1W1 * W2,  W1W2, jel, region2.GetDofNumber(jel), fem2->GetPhi(jg), solu2g[jg] * W1W2, W2);
          // }
          if(_cut == 0) { //interior element
            if(KernelIsConstant()) {
              double W2 = 2. * weight2[jg] * _kernel * I2[jg];
              double W1W2 = W1 * W2;
              AssemblyCutFem2(_phi1W1, solu1W1 * W2,  W1W2, jel, region2.GetDofNumber(jel), fem2->GetPhi(jg), solu2g[jg] * W1W2, W2);
            }
            else { // gamma-weighted y-moments, per ball; I2 correction bypassed
              double W1g = 0.;
              _phi1W1g.assign(nDof1, 0.);
              for(unsigned ig = 0; ig < ng1; ig++) {
                const double wg = _weight1[ig] * GetGamma(_xg1[ig], xg2[jg]);
                W1g += wg;
                for(unsigned i = 0; i < nDof1; i++) _phi1W1g[i] += phi1[ig][i] * wg;
              }
              double solu1W1g = 0.;
              for(unsigned i = 0; i < nDof1; i++) solu1W1g += solu1[i] * _phi1W1g[i];

              if(probeR > 0. && dim == 2) {
                const double px = xg2[jg][0] - probeX, py = xg2[jg][1] - probeY;
                if(px * px + py * py < probeR * probeR) {
                  _FP &p = _probe[std::make_pair(jel, jg)];
                  p.x = xg2[jg][0]; p.y = xg2[jg][1]; p.I2v = I2[jg];
                  for(unsigned ig = 0; ig < ng1; ig++) {
                    const double d = GetDistance(_xg1[ig], xg2[jg]);
                    const double wg = _weight1[ig] * GetGamma(d);
                    p.S0 += wg; p.S2 += d * d * wg; p.S4 += d * d * d * d * wg;
                  }
                }
              }

              double W2 = 2. * weight2[jg] * _kernel * I2[jg];
              double W1gW2 = W1g * W2;
              AssemblyCutFem2(_phi1W1g, solu1W1g * W2, W1gW2, jel, region2.GetDofNumber(jel), fem2->GetPhi(jg), solu2g[jg] * W1gW2, W2);
            }
          }
          else if(_cut == 1) { //cut element


            if(element1.GetElementType() == 3) {
              static const long dumpStride = [](){ const char *s = getenv("DUMP_STRIDE");
                return s ? atol(s) : 0L; }();
                if(dumpStride > 0) {
                  static long dumpCnt = 0;
                  if((dumpCnt++ % dumpStride) == 0) {
                    std::cout << "[DUMP] " << dumpCnt;
                    for(unsigned i = 0; i < 4; i++)
                      std::cout << " " << xv1[0][i] << " " << xv1[1][i];
                    std::cout << " " << xg2[jg][0] << " " << xg2[jg][1] << " " << delta << "\n";
                  }
                }
            }


            element1.GetCutFem()->clear();
            element1.GetCDweight()->GetWeight(_a, _d, _eqPolyWeight);

            double W1CF = 0.;
            _phi1W1CF.assign(nDof1, 0.);
            for(unsigned ig = 0; ig != ng1CF; ++ig) {
              double weightigjg = _weight1CF[ig] * _eqPolyWeight[ig];
              if(!KernelIsConstant()) weightigjg *= GetGamma(_xg1CF[ig], xg2[jg]);
              W1CF += weightigjg;
              for(unsigned i = 0; i != nDof1; ++i) {
                _phi1W1CF[i] += phi1CF[ig][i] * weightigjg;
              }
            }
            double solu1W1CF = 0.;
            for(unsigned i = 0; i != nDof1; ++i) {
              solu1W1CF += solu1[i] * _phi1W1CF[i];
            }

            if(probeR > 0. && dim == 2 && !KernelIsConstant()) {
              const double px = xg2[jg][0] - probeX, py = xg2[jg][1] - probeY;
              if(px * px + py * py < probeR * probeR) {
                _FP &p = _probe[std::make_pair(jel, jg)];
                p.x = xg2[jg][0]; p.y = xg2[jg][1]; p.I2v = I2[jg];
                for(unsigned ig = 0; ig != ng1CF; ++ig) {
                  const double d = GetDistance(_xg1CF[ig], xg2[jg]);
                  const double wg = _weight1CF[ig] * _eqPolyWeight[ig] * GetGamma(d);
                  p.S0 += wg; p.S2 += d * d * wg; p.S4 += d * d * d * d * wg;
                }
              }
            }

            double W2 = 2. * weight2[jg] * _kernel * I2[jg];
            double W1CFW2 = W1CF * W2;
            AssemblyCutFem2(_phi1W1CF, solu1W1CF * W2,  W1CFW2, jel, region2.GetDofNumber(jel), fem2->GetPhi(jg), solu2g[jg] * W1CFW2, W2);
          }
        }
      }
    }
  }
  else {
    const unsigned &dim = element1.GetDimension();
    const std::vector < std::vector <double> >  &xv1 = element1.GetElement1NodeCoordinates(level, iFather);

    _jelIndexR[level].resize(0);
    _jgIndexR[level].resize(0);
    _jelIndexI.resize(0);

    std::vector < std::pair<std::vector<double>::const_iterator, std::vector<double>::const_iterator> > x1MinMax(dim);
    for(unsigned k = 0; k < dim; k++) {
      x1MinMax[k] = std::minmax_element(xv1[k].begin(), xv1[k].end());
    }

    const bool gateActive = (gateOn != 0) && (element1.GetElementType() == 3) && (dim == 2);

    // fem quantities at THIS (level, iFather), computed lazily: needed by the
    // whole-jel interior block (as before) and by any pair the gate resolves here
    const unsigned &nDof1 = element1.GetNumberOfNodes();
    const elem_type *fem1 = element1.GetFem1();
    const elem_type *fem1CF = element1.GetFem1CF();
    const std::vector < std::vector < double> > & phi1 = octTreeElement1.GetGaussShapeFunctions();
    const std::vector < std::vector < double> > & phi1CF = octTreeElement1CF.GetGaussShapeFunctions();
    const unsigned &ng1 = fem1->GetGaussPointNumber();
    const unsigned &ng1CF = fem1CF->GetGaussPointNumber();

    double W1 = 0., solu1W1 = 0.;
    bool fem1Ready = false;
    auto EnsureFem1 = [&]() {
      if(fem1Ready) return;
      fem1Ready = true;
      W1 = 0.;
      _phi1W1.assign(nDof1, 0.);
      _xg1.assign(ng1, std::vector<double>(dim, 0));
      _weight1.resize(ng1);
      for(unsigned ig = 0; ig < ng1; ig++) {
        const double *phi;
        fem1->GetGaussQuantities(xv1, ig, _weight1[ig], phi);
        W1 += _weight1[ig];
        for(unsigned i = 0; i < nDof1; i++) {
          _phi1W1[i] += phi1[ig][i] * _weight1[ig];
          for(unsigned k = 0; k < dim; k++) {
            _xg1[ig][k] += phi[i] * xv1[k][i];
          }
        }
      }
      solu1W1 = 0.;
      for(unsigned i = 0; i < nDof1; i++) {
        solu1W1 += solu1[i] * _phi1W1[i];
      }
    };

    bool fem1CFReady = false;
    auto EnsureFem1CF = [&]() {
      if(fem1CFReady) return;
      fem1CFReady = true;
      _weight1CF.resize(ng1CF);
      _xg1CF.assign(ng1CF, std::vector<double>(dim, 0));
      for(unsigned ig = 0; ig < ng1CF; ig++) {
        const double *phi;
        fem1CF->GetGaussQuantities(xv1, ig, _weight1CF[ig], phi);
        for(unsigned i = 0; i < nDof1; i++) {
          for(unsigned k = 0; k < dim; k++) {
            _xg1CF[ig][k] += phi[i] * xv1[k][i];
          }
        }
      }
    };

    std::vector < double > dmM2(dim);
    std::vector < double > dMm2(dim);
    std::vector < double > dist(pow(2, dim));
    std::vector <unsigned> jgKeep;

    for(unsigned j = 0; j < jelIndexF.size(); j++) {
      unsigned jel = jelIndexF[j];
      const std::vector<std::vector<double>>& x2MinMax = region2.GetMinMax(jel);

      for(unsigned k = 0; k < dim; k++) {
        dmM2[k] = (*(x1MinMax[k].first) - x2MinMax[k][1]) * (*(x1MinMax[k].first) - x2MinMax[k][1]);
        dMm2[k] = (*(x1MinMax[k].second) - x2MinMax[k][0]) * (*(x1MinMax[k].second) - x2MinMax[k][0]);
      }

      if(dim == 2) {
        dist[0] = sqrt(dmM2[0] + dmM2[1]);
        dist[1] = sqrt(dMm2[0] + dmM2[1]);
        dist[2] = sqrt(dmM2[0] + dMm2[1]);
        dist[3] = sqrt(dMm2[0] + dMm2[1]);
      }
      else if(dim == 3) {
        dist[0] = sqrt(dmM2[0] + dmM2[1] + dmM2[2]);
        dist[1] = sqrt(dMm2[0] + dmM2[1] + dmM2[2]);
        dist[2] = sqrt(dmM2[0] + dMm2[1] + dmM2[2]);
        dist[3] = sqrt(dMm2[0] + dMm2[1] + dmM2[2]);

        dist[4] = sqrt(dmM2[0] + dmM2[1] + dMm2[2]);
        dist[5] = sqrt(dMm2[0] + dmM2[1] + dMm2[2]);
        dist[6] = sqrt(dmM2[0] + dMm2[1] + dMm2[2]);
        dist[7] = sqrt(dMm2[0] + dMm2[1] + dMm2[2]);
      }

      // KernelIsConstant(): this whole-jel shortcut assembles with W1/_phi1W1,
      // which carry no gamma. It is exact for a constant kernel (the integrand
      // factorizes and the fifth-order rule integrates the biquadratic products
      // exactly at any sub-element size), but for a varying gamma it would drop
      // the kernel weight entirely, so non-constant kernels must refine to the
      // leaf, where the gamma-weighted branches live.
      if(*std::max_element(dist.begin(), dist.end()) < delta && KernelIsConstant()) {
        if(!gateActive) {
          _jelIndexI.resize(_jelIndexI.size() + 1, jel);
        }
        else {
          // whole-jel interior, gated: resolve the ACTIVE pairs only.
          // Looping all Gauss points here would double-count pairs already
          // resolved at a coarser level (their contribution over the whole
          // parent, which contains this sub-element, is already assembled).
          EnsureFem1();
          const elem_type *fem2 = region2.GetFem(jel);
          const std::vector <double >  &solu2g = region2.GetGaussSolution(jel);
          const std::vector <double >  &weight2 = region2.GetGaussWeight(jel);
          const std::vector<double>& I2 = region2.GetI2(jel);
          const unsigned nDof2 = region2.GetDofNumber(jel);
          const std::vector <unsigned> &jgList = jgIndexF[j];
          for(unsigned jgi = 0; jgi < jgList.size(); jgi++) {
            const unsigned jg = jgList[jgi];
            double W2 = 2. * weight2[jg] * _kernel * I2[jg];
            double W1W2 = W1 * W2;
            AssemblyCutFem2(_phi1W1, solu1W1 * W2,  W1W2, jel, nDof2, fem2->GetPhi(jg), solu2g[jg] * W1W2, W2);
            _gateInt[level]++;
          }
        }
      }
      else if(!gateActive) {
        _jelIndexR[level].resize(_jelIndexR[level].size() + 1, jel);
        _jgIndexR[level].push_back(jgIndexF[j]);
      }
      else {
        // per-PAIR gate: classify each surviving Gauss point of this jel
        const elem_type *fem2 = region2.GetFem(jel);
        const std::vector <double >  &solu2g = region2.GetGaussSolution(jel);
        const std::vector <double >  &weight2 = region2.GetGaussWeight(jel);
        const std::vector < std::vector <double> >  &xg2 = region2.GetGaussCoordinates(jel);
        const std::vector<double>& I2 = region2.GetI2(jel);
        const unsigned nDof2 = region2.GetDofNumber(jel);

        jgKeep.resize(0);
        const std::vector <unsigned> &jgList = jgIndexF[j];
        for(unsigned jgi = 0; jgi < jgList.size(); jgi++) {
          const unsigned jg = jgList[jgi];

          double dist2ToBox = 0.;
          for(unsigned k = 0; k < dim; k++) {
            const double lo = *(x1MinMax[k].first), hi = *(x1MinMax[k].second);
            const double c = (xg2[jg][k] < lo) ? lo : ((xg2[jg][k] > hi) ? hi : xg2[jg][k]);
            dist2ToBox += (xg2[jg][k] - c) * (xg2[jg][k] - c);
          }
          if(dist2ToBox >= delta * delta) {  // ball misses the sub-element AABB: zero contribution in the whole subtree
            _gateDrop[level]++;
            continue;
          }

          unsigned n_in = 0;    // corner vertices inside the ball
          for(unsigned i = 0; i < 4; i++) {
            double s = 0.;
            for(unsigned k = 0; k < dim; k++) s += (xv1[k][i] - xg2[jg][k]) * (xv1[k][i] - xg2[jg][k]);
            if(s < delta * delta) n_in++;
          }

          if(n_in == 4) {       // sub-element fully inside this ball: exact early resolution
            EnsureFem1();
            double W2 = 2. * weight2[jg] * _kernel * I2[jg];
            double W1W2 = W1 * W2;
            AssemblyCutFem2(_phi1W1, solu1W1 * W2,  W1W2, jel, nDof2, fem2->GetPhi(jg), solu2g[jg] * W1W2, W2);
            _gateInt[level]++;
          }
          else if(n_in > 0 && PredictQuadDepth(xv1, xg2[jg], delta, gateEps, gateMargin) == 0) {
            // cut pair predicted resolvable at the current sub-element: leaf machinery NOW
            _ballAprx->GetNormal(element1.GetElementType(), xv1, xg2[jg], delta, _a, _d, _cut);
            if(_cut == 0) {
              EnsureFem1();
              double W2 = 2. * weight2[jg] * _kernel * I2[jg];
              double W1W2 = W1 * W2;
              AssemblyCutFem2(_phi1W1, solu1W1 * W2,  W1W2, jel, nDof2, fem2->GetPhi(jg), solu2g[jg] * W1W2, W2);
            }
            else if(_cut == 1) {
              EnsureFem1CF();
              element1.GetCutFem()->clear();
              element1.GetCDweight()->GetWeight(_a, _d, _eqPolyWeight);

              double W1CF = 0.;
              _phi1W1CF.assign(nDof1, 0.);
              for(unsigned ig = 0; ig != ng1CF; ++ig) {
                double weightigjg = _weight1CF[ig] * _eqPolyWeight[ig];
                W1CF += weightigjg;
                for(unsigned i = 0; i != nDof1; ++i) {
                  _phi1W1CF[i] += phi1CF[ig][i] * weightigjg;
                }
              }
              double solu1W1CF = 0.;
              for(unsigned i = 0; i != nDof1; ++i) {
                solu1W1CF += solu1[i] * _phi1W1CF[i];
              }

              double W2 = 2. * weight2[jg] * _kernel * I2[jg];
              double W1CFW2 = W1CF * W2;
              AssemblyCutFem2(_phi1W1CF, solu1W1CF * W2,  W1CFW2, jel, nDof2, fem2->GetPhi(jg), solu2g[jg] * W1CFW2, W2);
            }
            // _cut == 2: outside, contributes nothing
            _gateCut[level]++;
          }
          else {
            jgKeep.push_back(jg);
            _gateProp[level]++;
          }
        }
        if(jgKeep.size() > 0) {
          _jelIndexR[level].resize(_jelIndexR[level].size() + 1, jel);
          _jgIndexR[level].push_back(jgKeep);
        }
      }
    }
    if(_jelIndexI.size() > 0) {

      EnsureFem1();

      for(unsigned jj = 0; jj < _jelIndexI.size(); jj++) {
        unsigned jel = _jelIndexI[jj];
        const elem_type *fem2 = region2.GetFem(jel);

        const std::vector <double >  &solu2g = region2.GetGaussSolution(jel);
        const std::vector <double >  &weight2 = region2.GetGaussWeight(jel);
        const std::vector < std::vector <double> >  &xg2 = region2.GetGaussCoordinates(jel);
        const std::vector<double>& I2 = region2.GetI2(jel);

        for(unsigned jg = 0; jg < fem2->GetGaussPointNumber(); jg++) {
          double W2 = 2. * weight2[jg] * _kernel * I2[jg];
          double W1W2 = W1 * W2;
          AssemblyCutFem2(_phi1W1, solu1W1 * W2,  W1W2, jel, region2.GetDofNumber(jel), fem2->GetPhi(jg), solu2g[jg] * W1W2, W2);
        }
      }
    }
    if(_jelIndexR[level].size() > 0) {
      element1.BuildElement1Prolongation(level, iFather);
      for(unsigned i = 0; i < element1.GetNumberOfChildren(); i++) {
        AssemblyCutFem1Gated(level + 1, levelMin1, levelMax1Loc, i,
                             *octTreeElement1.GetElement(std::vector<unsigned> {i}),
                             *octTreeElement1CF.GetElement(std::vector<unsigned> {i}),
                             element1, region2, _jelIndexR[level], _jgIndexR[level],
                             solu1, kappa, delta, printMesh);
      }
    }
  }
}


void NonLocal::AssemblyCutFem2(const std::vector <double> &phi1W1,
                               const double &solu1W1W2,
                               const double &W1W2,
                               const unsigned &jel,
                               const unsigned &nDof2,
                               const double *phi2,
                               const double &solu2W1W2,
                               const double &W2) {

  _phi1W1W2.resize(phi1W1.size());
  _phi1W1W2Begin = _phi1W1W2.begin();
  _phi1W1W2End = _phi1W1W2.end();

  for(_phi1W1W2It = _phi1W1W2Begin, _phi1W1It = phi1W1.begin(); _phi1W1W2It !=  _phi1W1W2End; ++_phi1W1W2It, ++_phi1W1It) {
    *_phi1W1W2It = *_phi1W1It * W2;
  }

  _phi2W1W2.resize(nDof2);
  _phi2W1W2Begin = _phi2W1W2.begin();
  _phi2W1W2End = _phi2W1W2.end();

  for( _phi2pt = phi2, _phi2W1W2It = _phi2W1W2Begin; _phi2W1W2It != _phi2W1W2End; ++_phi2pt, ++_phi2W1W2It) {
    *_phi2W1W2It = *_phi2pt * W1W2;
  }

  _jac21End = _jac21[jel].end();
  for(_jac21It = _jac21[jel].begin(), _jac22It = _jac22[jel].begin(), _res2It = _res2[jel].begin(), _phi2pt = phi2; _jac21It != _jac21End; ++_phi2pt, ++_res2It) {
    for(_phi1W1W2It = _phi1W1W2Begin; _phi1W1W2It != _phi1W1W2End; ++_phi1W1W2It, ++_jac21It) {
      *_jac21It += *_phi2pt * (*_phi1W1W2It);
    }
    for(_phi2W1W2It = _phi2W1W2Begin; _phi2W1W2It != _phi2W1W2End; ++_phi2W1W2It, ++_jac22It) {
      *_jac22It -= *_phi2pt * (*_phi2W1W2It);
    }
    *_res2It += *_phi2pt * (solu2W1W2 - solu1W1W2);
  }
}



void NonLocal::PrintElement(const std::vector < std::vector < double> > &xv, const RefineElement & refineElement) {
  fout.open("mesh.txt", std::ios::app);

  for(unsigned j = 0; j < refineElement.GetNumberOfLinearNodes(); j++) {
    fout << xv[0][j] << " " << xv[1][j] << " " << std::endl;
  }
  fout << xv[0][0] << " " << xv[1][0] << " " << std::endl;
  fout << std::endl;

  fout.close();
}


class NonLocalBall: public NonLocal {
  public:
    NonLocalBall(): NonLocal() {};
    ~NonLocalBall() {};

    double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &radius) const {
      double distance  = 0.;
      for(unsigned k = 0; k < xc.size(); k++) {
        distance += (xp[k] - xc[k]) * (xp[k] - xc[k]);
      }
      distance = radius - sqrt(distance);
      return distance;
    };

    void SetKernel(const double  &kappa, const double &delta, const double &eps) {
      _kernel = 4. * kappa / (M_PI  * delta * delta * delta * delta)
                / (1. + 6. / 11. * pow(eps / delta, 2) + 3. / 143. * pow(eps / delta, 4.));
    }

    double GetArea(const double &delta, const double &eps) const {
      return M_PI * (delta * delta + eps * eps / 11.);
    };

    double GetGamma(const double &d) const {
      return 1.;
    }

    double GetGamma(const std::vector < double>  &x1, const std::vector < double>  &x2) const {
      return 1.;
    }
};


class NonLocalBallFrac: public NonLocalBall {
public:
  NonLocalBallFrac(): NonLocalBall() {
    const char *e = getenv("FRAC_D0");
    _d0 = e ? atof(e) : 1.0e10;   // default: huge d0 = constant-kernel regression mode
  };
  ~NonLocalBallFrac() {};

  void SetKernel(const double &kappa, const double &delta, const double &eps) {
    // _kernel = kappa / (2. * M_PI);
    _kernel = kappa / (4. * M_PI);
  }
  bool KernelIsConstant() const { return false; }

  // // regularized fractional kernel: 1/max(d,d0)^3
  // double GetGamma(const double &d) const {
  //   const double dd = (d < _d0) ? _d0 : d;
  //   return 1. / (dd * dd * dd);
  // }
  double GetGamma(const double &d) const {
    const double dd = d * d + _d0 * _d0;
    return 1. / (dd * std::sqrt(dd));      // (d^2 + d0^2)^{-3/2}
  }
  double GetGamma(const std::vector<double> &x1, const std::vector<double> &x2) const {
    return GetGamma(GetDistance(x1, x2));
  }
private:
  double _d0;
};

class NonLocalBall3D: public NonLocal {
  public:
    NonLocalBall3D(): NonLocal() {};
    ~NonLocalBall3D() {};

    double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &radius) const {
      double distance  = 0.;
      for(unsigned k = 0; k < xc.size(); k++) {
        distance += (xp[k] - xc[k]) * (xp[k] - xc[k]);
      }
      distance = radius - sqrt(distance);
      return distance;
    };

    void SetKernel(const double  &kappa, const double &delta, const double &eps) {
      _kernel = 15. * kappa / (4. * M_PI  * delta * delta * delta * delta * delta)
                / (1. + 10. / 11. * pow(eps / delta, 2) + 15. / 143. * pow(eps / delta, 4.));
    }

    double GetArea(const double &delta, const double &eps) const {
      return 4. / 3. * M_PI * (delta * delta * delta) * (1. + 3. / 11. * pow(eps / delta, 2));
    };

    double GetGamma(const double &d) const {
      return 1.;
    }

    double GetGamma(const std::vector < double>  &x1, const std::vector < double>  &x2) const {
      return 1.;
    }
};



class NonLocalBall1: public NonLocal {
  public:
    NonLocalBall1(): NonLocal() {};
    ~NonLocalBall1() {};

    double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &radius) const {
      double distance  = 0.;
      for(unsigned k = 0; k < xc.size(); k++) {
        distance += (xp[k] - xc[k]) * (xp[k] - xc[k]);
      }
      distance = radius - sqrt(distance);
      return distance;
    }

    void SetKernel(const double  &kappa, const double &delta, const double &eps) {
      _kernel = 3. * kappa / (M_PI  * delta * delta * delta)
                / (1. + 3. / 11. * pow(eps / delta, 2.))  ;
    }

    double GetArea(const double &delta, const double &eps) const {
      return 2. * M_PI * delta;
    };

    double GetGamma(const double &d) const {
      return 1. / d;
    }

    double GetGamma(const std::vector < double>  &x1, const std::vector < double>  &x2) const {
      return 1. / GetDistance(x1, x2);
    }

};




class NonLocalBox: public NonLocal {
  public:
    NonLocalBox(): NonLocal() {};
    ~NonLocalBox() {};
    double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &halfSide) const;
    void SetKernel(const double  &kappa, const double &delta, const double &eps) {
      _kernel = 0.75 * kappa / (delta * delta * delta * delta);
    };
    double GetArea(const double &delta, const double &eps) const {
      return delta * delta;
    };

    double GetGamma(const double &d) const {
      return 1.;
    }
    double GetGamma(const std::vector < double>  &x1, const std::vector < double>  &x2) const {
      return 1.;
    }
};


double NonLocalBox::GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double & halfSide) const {

  double distance = 0.;
  unsigned dim = xc.size();
  std::vector < double > din(2 * dim); // used only if the point is inside
  std::vector < double > dout(dim, 0.); // used only if the point is outside

  bool inside = true;
  for(unsigned k = 0; k < dim; k++) {
    din[2 * k] = xp[k] - (xc[k] - halfSide); // point minus box left-side:  < 0 -> point is outside
    din[2 * k + 1] = (xc[k] + halfSide) - xp[k]; // box right-side minus point: < 0 -> point is outside
    if(din[2 * k] < 0.) {
      dout[k] = din[2 * k];
      inside = false;
    }
    else if(din[2 * k + 1] < 0.) {
      dout[k] = din[2 * k + 1];
      inside = false;
    }
  }

  if(inside) {
    distance = *std::min_element(din.begin(), din.end());
  }
  else {
    distance = 0.;
    for(unsigned k = 0; k < dim; k++) {
      distance += dout[k] * dout[k];
    }
    distance = -sqrt(distance);
  }
  return distance;
}


#endif






