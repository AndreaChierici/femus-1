#ifndef __femus_NonLocal_hpp__
#define __femus_NonLocal_hpp__

#include "GetNormal.hpp"
#include "RefineElement.hpp"

// #pragma omp requires unified_shared_memory

std::ofstream fout;

class NonLocalBall;

struct NonlocalMatrixView {
    std::vector<unsigned> offsetJac21;
    std::vector<unsigned> offsetJac22;
    std::vector<unsigned> offsetRes2;

    std::vector<double>   jac21Flat;
    std::vector<double>   jac22Flat;
    std::vector<double>   res2Flat;
};

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
    #pragma omp begin declare target
    virtual double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &size) const = 0;
    #pragma omp end declare target

    virtual void SetKernel(const double  &kappa, const double &delta, const double &eps) = 0;
    const double & GetKernel() const {
      return _kernel;
    };
    virtual double GetArea(const double &delta, const double &eps) const = 0;
    virtual double GetGamma(const double &d) const = 0;
    virtual double GetGamma(const std::vector < double>  &x1, const std::vector < double>  &x2) const = 0;


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

    double Assembly2_flat_CPU(const RefineElement& element1,
                          const RegionDeviceData& D,
                          const unsigned* jelIndex,
                          unsigned jelCount,
                          unsigned nDof1,
                          const double* xg1,
                          double twoWeigh1Kernel,
                          const double* phi1,
                          const double* solu1,
                          double delta,
                          bool printMesh,
                          const double* phi2Flat,
                          unsigned nGauss2_ref,
                          unsigned nDof2_ref);

    double Assembly2_flat_GPU(const RegionDeviceView& V, const unsigned* jelIndex, unsigned jelCount, unsigned nDof1,
      const double* xg1, double twoWeigh1Kernel, const double* phi1, const double* solu1, double delta, const double* phi2Flat,
      unsigned nGauss2_ref, unsigned nDof2_ref, const SmoothStepData& stepData, size_t dimCount, size_t nGauss2Count,
      size_t nDof2Count, size_t x2MinMaxOffsetCount, size_t x2MinMaxAllCount, size_t xg2OffsetCount, size_t xg2AllCount,
      size_t w2OffsetCount, size_t w2AllCount, size_t solu2OffsetCount, size_t solu2AllCount);

    double GetSmoothTestFunction(const double &dg1, const double &eps);

    void ProcessTasks_CPU(const RefineElement& element1, Region& region2, const std::vector<double>& solu1, const double& delta, const bool& printMesh);

    void ProcessTasks_GPU(const RefineElement& element1, Region& region2, const std::vector<double>& solu1, const double& delta, const bool& printMesh);

    void BuildMatrixView(const Region& region2, unsigned nDof1);

    void ScatterBackFromFlat(const Region& region2, unsigned nDof1);


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

    BallApproximation *_ballAprx;
    std::vector<double> _a;

    std::vector< std::vector < double> > _xg1;
    std::vector< std::vector < double> > _xg1CF;

    std::vector < double> _weight1;
    std::vector < double> _weight1CF;
    std::vector<double> _eqPolyWeight;

    std::vector <double> _phi1W1;
    std::vector <double> _phi1W1CF;

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

    struct NonlocalTask {
      double   xg1[3];          // 2D or 3D, we just use first 'dim'
      double   twoWeigh1Kernel;
      unsigned nDof1;

      unsigned phi1Offset;      // index into _phi1All where phi1 starts

      unsigned jelBegin;        // index into _jelIndexAll
      unsigned jelCount;        // how many entries for this task
    };

    std::vector<NonlocalTask> _tasks;
    std::vector<unsigned>     _jelIndexAll;
    std::vector<double>       _phi1All;

    NonlocalMatrixView        _matView;

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
  for(unsigned level = 0; level < levelMax1; level++) {
    _jelIndexR[level].reserve(region2.size());
  }

  _tasks.clear();
  _jelIndexAll.clear();
  _phi1All.clear();

}

void NonLocal::ProcessTasks_CPU(const RefineElement& element1,
                                Region& region2,
                                const std::vector<double>& solu1,
                                const double& delta,
                                const bool& printMesh) {

  for (unsigned t = 0; t < _tasks.size(); ++t) {
    const auto& task = _tasks[t];

    // rebuild jelIndex slice
    std::vector<unsigned> jel(task.jelCount);
    for (unsigned jj = 0; jj < task.jelCount; ++jj) {
      jel[jj] = _jelIndexAll[task.jelBegin + jj];
    }

    // rebuild phi1 for this task from flat storage
    std::vector<double> phi1(task.nDof1);
    for (unsigned i = 0; i < task.nDof1; ++i) {
      phi1[i] = _phi1All[task.phi1Offset + i];
    }

    // rebuild xg1 as std::vector<double>
    std::vector<double> xg1(element1.GetDimension());
    for (unsigned k = 0; k < element1.GetDimension(); ++k) {
      xg1[k] = task.xg1[k];
    }

    Assembly2(element1, region2, jel, task.nDof1, xg1, task.twoWeigh1Kernel, phi1,
              solu1, delta, printMesh);
  }
}

#pragma omp declare target
double interface_distance_ball_raw(const double* xc,
                                   const double* xp,
                                   unsigned dim,
                                   double radius);
#pragma omp end declare target

void NonLocal::ProcessTasks_GPU(const RefineElement& element1,
                                Region&              region2,
                                const std::vector<double>& solu1,
                                const double&        delta,
                                const bool&          /*printMesh*/)
{
  const unsigned nDof1    = element1.GetNumberOfNodes();
  const unsigned dimSpace = element1.GetDimension();
  const unsigned totalTasks = _tasks.size();

  // Estimate total GPU work
  unsigned totalJelWork = 0;
  for (unsigned t = 0; t < totalTasks; ++t) {
    totalJelWork += _tasks[t].jelCount;
  }

  const unsigned MIN_GPU_WORK = 10000;
  if (totalJelWork < MIN_GPU_WORK) {
    ProcessTasks_CPU(element1, region2, solu1, delta, /*printMesh*/ false);
    return;
  }

  // 1) Prepare flat matrix layout (for all jel in region2)
  BuildMatrixView(region2, nDof1);

  // 2) Build flattened Region data
  RegionDeviceData D;
  region2.BuildDeviceData(D);

  // 3) Build flat phi2 from a reference element (assume same type/order)
  const elem_type* fem2_ref    = region2.GetFem(0);
  const unsigned   nGauss2_ref = fem2_ref->GetGaussPointNumber();
  const unsigned   nDof2_ref   = region2.GetDofNumber(0); // assumed constant for region2

  std::vector<double> phi2Flat(nGauss2_ref * nDof2_ref);
  for (unsigned jg = 0; jg < nGauss2_ref; ++jg) {
    const double* phi2_jg = fem2_ref->GetPhi(jg);
    for (unsigned i = 0; i < nDof2_ref; ++i) {
      phi2Flat[jg * nDof2_ref + i] = phi2_jg[i];
    }
  }
  double* phi2FlatPtr = phi2Flat.data();

  // 4) Device view of RegionDeviceData
  RegionDeviceView V{
    D.dim.data(), D.nGauss2.data(), D.nDof2.data(),
    D.x2MinMaxOffset.data(), D.x2MinMaxAll.data(),
    D.xg2Offset.data(), D.xg2All.data(),
    D.w2Offset.data(), D.w2All.data(),
    D.solu2Offset.data(), D.solu2All.data()
  };

  const unsigned* dimPtr         = V.dim;
  const unsigned* nGauss2Ptr     = V.nGauss2;
  const unsigned* nDof2Ptr       = V.nDof2;
  const unsigned* x2MinMaxOffPtr = V.x2MinMaxOffset;
  const double*   x2MinMaxAllPtr = V.x2MinMaxAll;
  const unsigned* xg2OffPtr      = V.xg2Offset;
  const double*   xg2AllPtr      = V.xg2All;
  const unsigned* w2OffPtr       = V.w2Offset;
  const double*   w2AllPtr       = V.w2All;
  const unsigned* solu2OffPtr    = V.solu2Offset;
  const double*   solu2AllPtr    = V.solu2All;

  // Flat assembly buffers
  double*   jac21FlatPtr = _matView.jac21Flat.data();
  double*   jac22FlatPtr = _matView.jac22Flat.data();
  double*   res2FlatPtr  = _matView.res2Flat.data();
  unsigned* offJac21Ptr  = _matView.offsetJac21.data();
  unsigned* offJac22Ptr  = _matView.offsetJac22.data();
  unsigned* offRes2Ptr   = _matView.offsetRes2.data();

  SmoothStepData stepData = element1.GetSmoothStepData();
  const double   eps      = stepData.eps;

  // 5) Host loop over tasks; each task launches a kernel over its jel range
  for (const NonlocalTask& task : _tasks) {

    if (!task.jelCount) continue;

    const unsigned* jelPtr  = _jelIndexAll.data() + task.jelBegin;   // length = task.jelCount
    const double*   phi1Ptr = _phi1All.data()    + task.phi1Offset;  // length = task.nDof1

    // xg1 (local coords of current Gauss point of element1)
    double xg1_host[3] = {0.0, 0.0, 0.0};
    for (unsigned k = 0; k < dimSpace; ++k) {
      xg1_host[k] = task.xg1[k];
    }
    const double* xg1Ptr = xg1_host;

    // solu1(xg1) on host for this task
    double solu1g = 0.0;
    for (unsigned i = 0; i < task.nDof1; ++i) {
      solu1g += solu1[i] * phi1Ptr[i];
    }

    const unsigned threads_per_team = 128;
    const unsigned numTeams         = task.jelCount
                                      ? std::min<unsigned>(task.jelCount, 456u)
                                      : 1u;

    // 6) Parallel over jel for this task
    // Each thread owns one jel and uses local (per-thread) buffers:
    //   mCphi2_loc[nDof2_ref]
    //   res2_loc[nDof2_ref]
    //   jac22_loc[nDof2_ref * nDof2_ref]
    #pragma omp target teams distribute parallel for \
            num_teams(numTeams) thread_limit(threads_per_team)
    for (unsigned jj = 0; jj < task.jelCount; ++jj) {

      const unsigned jelIdx = jelPtr[jj];
      const unsigned nDof2  = nDof2Ptr[jelIdx];

      // If region2 is homogeneous, this should hold; keep guard anyway
      if (nDof2 == 0 || nDof2 > nDof2_ref) continue;

      // Per-thread local buffers (on device)
      double mCphi2_loc[nDof2_ref];
      double res2_loc [nDof2_ref];
      double jac22_loc[nDof2_ref * nDof2_ref];

      double* jac21_jel = jac21FlatPtr + offJac21Ptr[jelIdx];
      double* jac22_jel = jac22FlatPtr + offJac22Ptr[jelIdx];
      double* res2_jel  = res2FlatPtr  + offRes2Ptr [jelIdx];

      const unsigned dim        = dimPtr[jelIdx];
      const unsigned nGauss2    = nGauss2Ptr[jelIdx];
      const unsigned baseMinMax = x2MinMaxOffPtr[jelIdx];

      // Initialize locals from global (preserve current matrix/res state)
      for (unsigned i = 0; i < nDof2; ++i) {
        mCphi2_loc[i] = 0.0;
        res2_loc[i]   = res2_jel[i];
      }
      // Optional: zero the remaining part if nDof2 < nDof2_ref
      for (unsigned i = nDof2; i < nDof2_ref; ++i) {
        mCphi2_loc[i] = 0.0;
        res2_loc[i]   = 0.0;
      }

      for (unsigned i = 0; i < nDof2; ++i) {
        double* row_loc = jac22_loc + i * nDof2_ref;
        double* row_g   = jac22_jel + i * nDof2;
        for (unsigned j = 0; j < nDof2; ++j) {
          row_loc[j] = row_g[j];
        }
        // Optional: zero tail j from nDof2 to nDof2_ref
        for (unsigned j = nDof2; j < nDof2_ref; ++j) {
          row_loc[j] = 0.0;
        }
      }

      // Coarse intersection
      bool hit = true;
      for (unsigned k = 0; k < dim; ++k) {
        const unsigned base = baseMinMax + 2 * k;
        const double xmin  = x2MinMaxAllPtr[base    ];
        const double xmax  = x2MinMaxAllPtr[base + 1];
        if ((xg1Ptr[k] - xmax) > delta + eps ||
            (xmin - xg1Ptr[k]) > delta + eps) {
          hit = false;
          break;
        }
      }
      if (!hit) continue;

      const unsigned baseXg2   = xg2OffPtr   [jelIdx];
      const unsigned baseW2    = w2OffPtr    [jelIdx];
      const unsigned baseSolu2 = solu2OffPtr [jelIdx];

      for (unsigned jg = 0; jg < nGauss2; ++jg) {
        double xg2_jg[3] = {0.0, 0.0, 0.0};
        for (unsigned k = 0; k < dim; ++k) {
          xg2_jg[k] = xg2AllPtr[baseXg2 + jg * dim + k];
        }

        const double dg1    = interface_distance_ball_raw(xg1Ptr, xg2_jg, dim, delta);
        const double U_jjjg = SmoothStepEval(dg1, stepData);
        if (U_jjjg <= 0.0) continue;

        const double w2    = w2AllPtr   [baseW2    + jg];
        const double solu2 = solu2AllPtr[baseSolu2 + jg];

        const double C = U_jjjg * w2 * task.twoWeigh1Kernel;
        const double* phi2_jg = phi2FlatPtr + jg * nDof2_ref; // reference FE

        for (unsigned i = 0; i < nDof2; ++i) {
          const double cPhi2i = C * phi2_jg[i];
          mCphi2_loc[i] -= cPhi2i;

          double*       jac22_row = jac22_loc + i * nDof2_ref;
          const double* phi2pt    = phi2_jg;
          for (unsigned j = 0; j < nDof2; ++j, ++phi2pt) {
            jac22_row[j] -= cPhi2i * (*phi2pt);
          }

          res2_loc[i] += cPhi2i * solu2;
        }
      }

      // Use mCphi2_loc to update jac21 and res2_loc
      unsigned ijIndex = 0;
      for (unsigned i = 0; i < nDof2; ++i) {
        const double mSum = mCphi2_loc[i];
        for (unsigned j = 0; j < task.nDof1; ++j, ++ijIndex) {
          jac21_jel[ijIndex] -= mSum * phi1Ptr[j];
        }
        res2_loc[i] += mSum * solu1g;
      }

      // Final write back: one pass over res2 and jac22
      for (unsigned i = 0; i < nDof2; ++i) {
        res2_jel[i] = res2_loc[i];
      }
      for (unsigned i = 0; i < nDof2; ++i) {
        double* row_loc = jac22_loc + i * nDof2_ref;
        double* row_g   = jac22_jel + i * nDof2;
        for (unsigned j = 0; j < nDof2; ++j) {
          row_g[j] = row_loc[j];
        }
      }
    } // jj
  }   // tasks

  // 7) Scatter from flat to original structures
  ScatterBackFromFlat(region2, nDof1);
}



void NonLocal::BuildMatrixView(const Region& region2, unsigned nDof1) {
  const unsigned nElem = region2.size();

  // Resize offset arrays (prefix sums)
  _matView.offsetJac21.resize(nElem + 1);
  _matView.offsetJac22.resize(nElem + 1);
  _matView.offsetRes2 .resize(nElem + 1);

  _matView.offsetJac21[0] = 0;
  _matView.offsetJac22[0] = 0;
  _matView.offsetRes2 [0] = 0;

  for (unsigned jel = 0; jel < nElem; ++jel) {
    const unsigned nDof2 = region2.GetDofNumber(jel);

    _matView.offsetJac21[jel + 1] =
        _matView.offsetJac21[jel] + nDof2 * nDof1;   // nDof2 x nDof1 block

    _matView.offsetJac22[jel + 1] =
        _matView.offsetJac22[jel] + nDof2 * nDof2;   // nDof2 x nDof2 block

    _matView.offsetRes2 [jel + 1] =
        _matView.offsetRes2 [jel] + nDof2;           // length nDof2
  }

  const unsigned totalJac21 = _matView.offsetJac21[nElem];
  const unsigned totalJac22 = _matView.offsetJac22[nElem];
  const unsigned totalRes2  = _matView.offsetRes2 [nElem];

  // Allocate and zero flat storage
  _matView.jac21Flat.assign(totalJac21, 0.0);
  _matView.jac22Flat.assign(totalJac22, 0.0);
  _matView.res2Flat .assign(totalRes2,  0.0);
}


void NonLocal::ScatterBackFromFlat(const Region& region2, unsigned nDof1){
    const unsigned nElem = region2.size();

    for (unsigned jel = 0; jel < nElem; ++jel) {
        unsigned nDof2 = region2.GetDofNumber(jel);

        const unsigned offRes  = _matView.offsetRes2 [jel];
        const unsigned offJ21  = _matView.offsetJac21[jel];
        const unsigned offJ22  = _matView.offsetJac22[jel];

        // optional safety checks
        assert(_res2 [jel].size()  == nDof2);
        assert(_jac21[jel].size()  == nDof2 * nDof1);
        assert(_jac22[jel].size()  == nDof2 * nDof2);

        // res2: length nDof2
        std::copy(_matView.res2Flat.begin() + offRes,
                  _matView.res2Flat.begin() + offRes + nDof2,
                  _res2[jel].begin());

        // jac21: length nDof2 * nDof1
        std::copy(_matView.jac21Flat.begin() + offJ21,
                  _matView.jac21Flat.begin() + offJ21 + nDof2 * nDof1,
                  _jac21[jel].begin());

        // jac22: length nDof2 * nDof2
        std::copy(_matView.jac22Flat.begin() + offJ22,
                  _matView.jac22Flat.begin() + offJ22 + nDof2 * nDof2,
                  _jac22[jel].begin());
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

      // //OLD
      // Assembly2(element1, region2, jelIndexF, nDof1, xg1, 2. * weight1 * _kernel, phi1F[ig], solu1, delta, printMesh);
      NonlocalTask t;
      for (unsigned k = 0; k < dim; ++k) t.xg1[k] = xg1[k];
      t.twoWeigh1Kernel = 2. * weight1 * _kernel;
      t.nDof1           = nDof1;

      t.phi1Offset = _phi1All.size();
      _phi1All.insert(_phi1All.end(), phi1F[ig].begin(), phi1F[ig].end());

      t.jelBegin = _jelIndexAll.size();
      t.jelCount = jelIndexF.size();
      _jelIndexAll.insert(_jelIndexAll.end(),
                          jelIndexF.begin(), jelIndexF.end());

      _tasks.push_back(t);
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
        // OLD
        // Assembly2(element1, region2, _jelIndexI, nDof1, xg1, 2. * weight1 * _kernel, phi1F[ig], solu1, delta, printMesh);

        // NEW
        NonlocalTask t;
        for (unsigned k = 0; k < dim; ++k) t.xg1[k] = xg1[k];
        t.twoWeigh1Kernel = 2. * weight1 * _kernel;
        t.nDof1           = nDof1;

        // phi1 to flat storage
        t.phi1Offset = _phi1All.size();
        _phi1All.insert(_phi1All.end(), phi1F[ig].begin(), phi1F[ig].end());

        // jel slice
        t.jelBegin = _jelIndexAll.size();
        t.jelCount = _jelIndexI.size();
        _jelIndexAll.insert(_jelIndexAll.end(),
                            _jelIndexI.begin(), _jelIndexI.end());

        _tasks.push_back(t);

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






void NonLocal::AssemblyCutFemI2(const unsigned &level, const unsigned &levelMin1, const unsigned &levelMax1, const unsigned &iFather,
                               const OctTreeElement &octTreeElement1, const OctTreeElement &octTreeElement1CF,
                               RefineElement &element1, Region &region2,
                               const std::vector <unsigned> &jelIndexF, const vector < double >  &solu1,
                               const double &kappa, const double &delta, const bool &printMesh) {


  if(level < levelMin1) {
    element1.BuildElement1Prolongation(level, iFather);
    for(unsigned i = 0; i < element1.GetNumberOfChildren(); i++) {
      AssemblyCutFemI2(level + 1, levelMin1, levelMax1, i,
                      *octTreeElement1.GetElement(std::vector<unsigned> {i}),
                      *octTreeElement1CF.GetElement(std::vector<unsigned> {i}),
                      element1, region2, jelIndexF,
                      solu1, kappa, delta, printMesh);
    }
  }
  else if(level == levelMax1 - 1) {
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
              d2W1 += d2 * _weight1[ig];
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
              d2W1CF += d2 * _weight1CF[ig] * _eqPolyWeight[ig];
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

      if(*std::max_element(dist.begin(), dist.end()) < delta) {
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
        AssemblyCutFemI2(level + 1, levelMin1, levelMax1, i,
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


  if(level < levelMin1) {
    element1.BuildElement1Prolongation(level, iFather);
    for(unsigned i = 0; i < element1.GetNumberOfChildren(); i++) {
      AssemblyCutFem1(level + 1, levelMin1, levelMax1, i,
                      *octTreeElement1.GetElement(std::vector<unsigned> {i}),
                      *octTreeElement1CF.GetElement(std::vector<unsigned> {i}),
                      element1, region2, jelIndexF,
                      solu1, kappa, delta, printMesh);
    }
  }
  else if(level == levelMax1 - 1) {
    const unsigned &dim = element1.GetDimension();
    const std::vector < std::vector <double> >  &xv1 = element1.GetElement1NodeCoordinates(level, iFather);

    const unsigned &nDof1 = element1.GetNumberOfNodes();
    const elem_type *fem1 = element1.GetFem1();
    const elem_type *fem1CF = element1.GetFem1CF();

    //these are the shape functions of iel evaluated in the gauss points of the refined elements
    const std::vector < std::vector < double> > & phi1 = octTreeElement1.GetGaussShapeFunctions();
    const std::vector < std::vector < double> > & phi1CF = octTreeElement1CF.GetGaussShapeFunctions();

    //these are the shape functions of the refined element evaluated in the gauss points of the refined element

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
            double W2 = 2. * weight2[jg] * _kernel * I2[jg];
            double W1W2 = W1 * W2;
            AssemblyCutFem2(_phi1W1, solu1W1 * W2,  W1W2, jel, region2.GetDofNumber(jel), fem2->GetPhi(jg), solu2g[jg] * W1W2, W2);
          }
          else if(_cut == 1) { //cut element
            element1.GetCutFem()->clear();
            //       element1.GetCutFem()->GetWeightWithMap(0, _a, _d, _eqPolyWeight);
//             (*element1.GetCutFem())(0, _a, _d, _eqPolyWeight);
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

      if(*std::max_element(dist.begin(), dist.end()) < delta) {
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
      const std::vector < std::vector < double> > & phi1 = octTreeElement1.GetGaussShapeFunctions();
      const unsigned &ng1 = fem1->GetGaussPointNumber();


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
        AssemblyCutFem1(level + 1, levelMin1, levelMax1, i,
                        *octTreeElement1.GetElement(std::vector<unsigned> {i}),
                        *octTreeElement1CF.GetElement(std::vector<unsigned> {i}),
                        element1, region2, _jelIndexR[level],
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

#pragma omp begin declare target
    double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &radius) const {
      double distance  = 0.;
      for(unsigned k = 0; k < xc.size(); k++) {
        distance += (xp[k] - xc[k]) * (xp[k] - xc[k]);
      }
      distance = radius - sqrt(distance);
      return distance;
    };

    // No std::vector
    double GetInterfaceDistance_raw(const double* xc,
                                    const double* xp,
                                    unsigned dim,
                                    const double &radius) const {
      double distance = 0.;
      for (unsigned k = 0; k < dim; ++k) {
        double diff = xp[k] - xc[k];
        distance += diff * diff;
      }
      distance = radius - std::sqrt(distance);
      return distance;
    }
    #pragma omp end declare target


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

class NonLocalBall3D: public NonLocal {
  public:
    NonLocalBall3D(): NonLocal() {};
    ~NonLocalBall3D() {};

    #pragma omp begin declare target
    double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &radius) const {
      double distance  = 0.;
      for(unsigned k = 0; k < xc.size(); k++) {
        distance += (xp[k] - xc[k]) * (xp[k] - xc[k]);
      }
      distance = radius - sqrt(distance);
      return distance;
    };

    double GetInterfaceDistance_raw(const double* xc,
                                    const double* xp,
                                    unsigned dim,
                                    const double &radius) const {
      double distance = 0.;
      for (unsigned k = 0; k < dim; ++k) {
        double diff = xp[k] - xc[k];
        distance += diff * diff;
      }
      distance = radius - std::sqrt(distance);
      return distance;
    }
#pragma omp end declare target

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

    #pragma omp begin declare target
    double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &radius) const {
      double distance  = 0.;
      for(unsigned k = 0; k < xc.size(); k++) {
        distance += (xp[k] - xc[k]) * (xp[k] - xc[k]);
      }
      distance = radius - sqrt(distance);
      return distance;
    }
    #pragma omp end declare target

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
    #pragma omp begin declare target
    double GetInterfaceDistance(const std::vector < double>  &xc, const std::vector < double>  &xp, const double &halfSide) const{
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
    };
    #pragma omp end declare target

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


double NonLocal::Assembly2(const RefineElement & element1, const Region & region2, const std::vector<unsigned> &jelIndex,
                           const unsigned & nDof1, const vector < double > &xg1,
                           const double & twoWeigh1Kernel, const vector < double > &phi1, const vector < double >  &solu1,
                           const double & delta, const bool & printMesh) {

  double area = 0.;

  double solu1g = 0.;
  for(unsigned i = 0; i < nDof1; i++) {
    solu1g += solu1[i] * phi1[i];
  }

// const double *phi2;
//  const double *phi2pt;


  std::vector<std::vector< double > > mCphi2iSum(jelIndex.size());

  for(unsigned jj = 0; jj < jelIndex.size(); jj++) {
    unsigned jel = jelIndex[jj];
    const unsigned &nDof2 = region2.GetDofNumber(jel);
    mCphi2iSum[jj].assign(nDof2, 0.);
  }

  const double& eps = element1.GetEps();
  NonLocalBall* thisBall=dynamic_cast<NonLocalBall*> (this);

  const elem_type *fem = region2.GetFem(0);
  std::vector<double*> phi2(fem->GetGaussPointNumber());

  std::vector<std::vector<double>> U(jelIndex.size(),std::vector<double>(fem->GetGaussPointNumber(),0.));
     
  for(unsigned jj = 0; jj < jelIndex.size(); jj++) {
    unsigned jel = jelIndex[jj];
    const std::vector < std::vector <double> >  &xg2 = region2.GetGaussCoordinates(jel);
    for(unsigned jg = 0; jg < fem->GetGaussPointNumber(); jg++) {
      U[jj][jg] = element1.GetSmoothStepFunction(thisBall->GetInterfaceDistance(xg1, xg2[jg], delta));
    }
  }     
  for(unsigned jg = 0; jg < fem->GetGaussPointNumber(); jg++) {
    phi2[jg] = fem->GetPhi(jg);
  }

//  unsigned N = jelIndex.size();
// unsigned threads_per_team = 64;                // or 128
// unsigned numTeams = (N == 0) ? 1 : std::min(N, 256u);  // cap at some max if you like

// #pragma omp target teams distribute parallel for num_teams(numTeams) thread_limit(threads_per_team)
 // #pragma omp target teams distribute parallel for num_teams(456) thread_limit(256)
// #pragma omp parallel for
  for(unsigned jj = 0; jj < jelIndex.size(); jj++) {
    const double *phi2pt;
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
      region2.GetGaussSolution(jel).data();
      const std::vector <double >  &weight2 = region2.GetGaussWeight(jel);
      const std::vector < std::vector <double> >  &xg2 = region2.GetGaussCoordinates(jel);

      for(unsigned jg = 0; jg < fem->GetGaussPointNumber(); jg++) {
        if( U[jj][jg] > 0.) {
//          double C =  U[jj] * GetGamma(xg1, xg2[jg]) *  weight2[jg] * twoWeigh1Kernel;
	  double C =  U[jj][jg] * weight2[jg] * twoWeigh1Kernel;
	  double *jac22pt = &_jac22[jel][0];

          for(unsigned i = 0; i < nDof2; i++) {
            double cPhi2i = C * phi2[jg][i];
            mCphi2iSum[jj][i] -= cPhi2i;
            unsigned j = 0;
            for(phi2pt = phi2[jg]; j < nDof2; j++, phi2pt++, jac22pt++) {
              *jac22pt -= cPhi2i * (*phi2pt);
            }
            _res2[jel][i] += cPhi2i * solu2g[jg];
          }
        }//end if U > 0.
      }//end jg loop


      unsigned ijIndex = 0;
      for(unsigned i = 0; i < nDof2; i++) {
        for(unsigned j = 0; j < nDof1; j++, ijIndex++) {
          _jac21[jel][ijIndex] -= mCphi2iSum[jj][i] * phi1[j];
        }
        _res2[jel][i] += mCphi2iSum[jj][i] * solu1g;
      }
    }
  }

  return area;
}



// Small POD-style helper: does all per-(task, jel) work.
// No std::vector, works on raw pointers only.
inline void nonLocalInnerElementKernel(
    const RegionDeviceData&   D,
    unsigned                  jel,
    const double*             xg1,            // length dim
    unsigned                  nDof1,
    const double*             phi1,           // length nDof1
    double                    twoWeigh1Kernel,
    double                    solu1g,
    double                    delta,
    double                    eps,
    const double*             phi2Flat,       // [nGauss2_ref * nDof2_ref]
    unsigned                  nDof2_ref,
    double*                   mCphi2i,        // length nDof2(jel)
    double*                   jac21_jel,      // &_jac21[jel][0]
    double*                   jac22_jel,      // &_jac22[jel][0]
    double*                   res2_jel,
    const SmoothStepData&     stepData )
{
    const unsigned dim     = D.dim[jel];
    const unsigned nGauss2 = D.nGauss2[jel];
    const unsigned nDof2   = D.nDof2[jel];

    const unsigned baseMinMax = D.x2MinMaxOffset[jel];

    bool coarseIntersectionTest = true;
    for (unsigned k = 0; k < dim; ++k) {
      double xmin = D.x2MinMaxAll[baseMinMax + 2 * k    ];
      double xmax = D.x2MinMaxAll[baseMinMax + 2 * k + 1];

      if ((xg1[k] - xmax) > delta + eps || (xmin - xg1[k]) > delta + eps) {
        coarseIntersectionTest = false;
        break;
      }
    }

    if (!coarseIntersectionTest) {
      return;
    }

    // Offsets for Gauss data
    const unsigned baseXg2   = D.xg2Offset[jel];
    const unsigned baseW2    = D.w2Offset[jel];
    const unsigned baseSolu2 = D.solu2Offset[jel];

    // Loop over Gauss points jg
    for (unsigned jg = 0; jg < nGauss2; ++jg) {
      double xg2_jg[3] = {0.0, 0.0, 0.0};  // up to 3D
      for (unsigned k = 0; k < dim; ++k) {
        xg2_jg[k] = D.xg2All[baseXg2 + jg * dim + k];
      }

      // const SmoothStepData s = element1.GetSmoothStepData();

      double dg1 = interface_distance_ball_raw(xg1, xg2_jg, dim, delta);
      // Smooth cut function U(jj,jg)
      double U_jjjg = SmoothStepEval(dg1, stepData);

      // const double U_jjjg = element1.GetSmoothStepFunction(interface_distance_ball_raw(xg1, xg2_jg, dim, delta));

      if (U_jjjg <= 0.0) continue;

      const double w2    = D.w2All[baseW2    + jg];
      const double solu2 = D.solu2All[baseSolu2 + jg];

      const double C = U_jjjg * w2 * twoWeigh1Kernel;

      // phi2 at this Gauss point (from flat array)
      const double* phi2_jg = &phi2Flat[jg * nDof2_ref];

      // pointer into jac22 (accumulated over i,j)
      double* jac22pt = jac22_jel;

      // i-loop (shape functions of jel)
      for (unsigned i = 0; i < nDof2; ++i) {
        const double cPhi2i = C * phi2_jg[i];
        mCphi2i[i] -= cPhi2i;

        // j-loop for jac22
        const double* phi2pt = phi2_jg;
        for (unsigned j = 0; j < nDof2; ++j, ++phi2pt, ++jac22pt) {
            *jac22pt -= cPhi2i * (*phi2pt);
        }

        res2_jel[i] += cPhi2i * solu2;
      }
    } // end jg loop

    // Finalize jac21 and res2 (coupling with element1)
    unsigned ijIndex = 0;
    for (unsigned i = 0; i < nDof2; ++i) {
        const double mSum = mCphi2i[i];
        for (unsigned j = 0; j < nDof1; ++j, ++ijIndex) {
            jac21_jel[ijIndex] -= mSum * phi1[j];
        }
        res2_jel[i] += mSum * solu1g;
    }
}

#pragma omp declare target
inline double interface_distance_ball_raw(const double* xc,
                                          const double* xp,
                                          unsigned dim,
                                          double radius)
{
  double distance = 0.0;
  for (unsigned k = 0; k < dim; ++k) {
    double diff = xp[k] - xc[k];
    distance += diff * diff;
  }
  distance = radius - std::sqrt(distance);
  return distance;
}
#pragma omp end declare target

double NonLocal::Assembly2_flat_CPU(const RefineElement& element1,
                                    const RegionDeviceData& D,
                                    const unsigned* jelIndex,
                                    unsigned jelCount,
                                    unsigned nDof1,
                                    const double* xg1,
                                    double twoWeigh1Kernel,
                                    const double* phi1,
                                    const double* solu1,
                                    double delta,
                                    bool printMesh,
                                    const double* phi2Flat,
                                    unsigned nGauss2_ref,
                                    unsigned nDof2_ref)
{
    double area = 0.0;

    double solu1g = 0.0;
    for (unsigned i = 0; i < nDof1; ++i) {
        solu1g += solu1[i] * phi1[i];
    }

    SmoothStepData stepData = element1.GetSmoothStepData();

    std::vector<std::vector<double>> mCphi2iSum(jelCount);
    for (unsigned jj = 0; jj < jelCount; ++jj) {
        unsigned jel = jelIndex[jj];
        unsigned nDof2 = D.nDof2[jel];
        mCphi2iSum[jj].assign(nDof2, 0.0);
    }

    const double eps = element1.GetEps();
    // NonLocalBall* thisBall = dynamic_cast<NonLocalBall*>(this);
    // assert(thisBall && "Assembly2_flat_CPU currently assumes NonLocalBall");

    // ----- Main loop over interacting elements jj, -----
    // ----- delegated to helper "nonLocalInnerElementKernel" -----
    for (unsigned jj = 0; jj < jelCount; ++jj) {
      const unsigned jel   = jelIndex[jj];
      double* mCphi2i      = mCphi2iSum[jj].data();
      double* jac21_jel    = _matView.jac21Flat.data() + _matView.offsetJac21[jel];
      double* jac22_jel    = _matView.jac22Flat.data() + _matView.offsetJac22[jel];
      double* res2_jel     = _matView.res2Flat .data() + _matView.offsetRes2 [jel];

      nonLocalInnerElementKernel(D, jel, xg1, nDof1, phi1, twoWeigh1Kernel, solu1g, delta, eps,
                                 phi2Flat, nDof2_ref, mCphi2i, jac21_jel, jac22_jel, res2_jel, stepData);
    }


    return area;
}

double NonLocal::Assembly2_flat_GPU(const RegionDeviceView& V,
                                    const unsigned* jelIndex,
                                    unsigned jelCount,
                                    unsigned nDof1,
                                    const double* xg1,
                                    double twoWeigh1Kernel,
                                    const double* phi1,
                                    const double* solu1,
                                    double delta,
                                    const double* phi2Flat,
                                    unsigned nGauss2_ref,
                                    unsigned nDof2_ref,
                                    const SmoothStepData& stepData,
                                    // sizes for mapping
                                    size_t dimCount, size_t nGauss2Count,
                                    size_t nDof2Count,
                                    size_t x2MinMaxOffsetCount,
                                    size_t x2MinMaxAllCount,
                                    size_t xg2OffsetCount, size_t xg2AllCount,
                                    size_t w2OffsetCount, size_t w2AllCount,
                                    size_t solu2OffsetCount, size_t solu2AllCount)
{
    double area = 0.0;

    // 1) solu1g on host (scalar)
    double solu1g = 0.0;
    for (unsigned i = 0; i < nDof1; ++i) {
        solu1g += solu1[i] * phi1[i];
    }

    std::vector<unsigned> offsetMC(jelCount + 1);
    offsetMC[0] = 0;
    for (unsigned jj = 0; jj < jelCount; ++jj) {
        const unsigned jel   = jelIndex[jj];
        const unsigned nDof2 = V.nDof2[jel];      // <--- use V
        offsetMC[jj + 1] = offsetMC[jj] + nDof2;
    }
    const unsigned totalMC = offsetMC[jelCount];
    std::vector<double> mCphi2All(totalMC, 0.0);

    const unsigned* dimPtr         = V.dim;
    const unsigned* nGauss2Ptr     = V.nGauss2;
    const unsigned* nDof2Ptr       = V.nDof2;
    const unsigned* x2MinMaxOffPtr = V.x2MinMaxOffset;
    const double*   x2MinMaxAllPtr = V.x2MinMaxAll;
    const unsigned* xg2OffPtr      = V.xg2Offset;
    const double*   xg2AllPtr      = V.xg2All;
    const unsigned* w2OffPtr       = V.w2Offset;
    const double*   w2AllPtr       = V.w2All;
    const unsigned* solu2OffPtr    = V.solu2Offset;
    const double*   solu2AllPtr    = V.solu2All;


    // 4) Pointers to flat matrices
    double* jac21FlatPtr = _matView.jac21Flat.data();
    double* jac22FlatPtr = _matView.jac22Flat.data();
    double* res2FlatPtr  = _matView.res2Flat.data();

    // 5) Offsets in flat matrices
    unsigned* offJac21Ptr = _matView.offsetJac21.data();
    unsigned* offJac22Ptr = _matView.offsetJac22.data();
    unsigned* offRes2Ptr  = _matView.offsetRes2.data();

    const size_t jac21Size   = _matView.jac21Flat.size();
    const size_t jac22Size   = _matView.jac22Flat.size();
    const size_t res2Size    = _matView.res2Flat.size();
    const size_t offJac21Size = _matView.offsetJac21.size();
    const size_t offJac22Size = _matView.offsetJac22.size();
    const size_t offRes2Size  = _matView.offsetRes2.size();

    // 6) Pointers for mCphi2
    unsigned* offsetMCptr = offsetMC.data();
    double*   mCphi2AllPtr = mCphi2All.data();

    const unsigned dimSpace = V.dim[0];
    const double   eps      = stepData.eps;

    const unsigned threads_per_team = 128; // or 256
    const unsigned numTeams = (jelCount == 0) ? 1 :
                              std::min(jelCount, 456u);// 7) Offload jj loop

    // #pragma omp target teams distribute parallel for num_teams(456) thread_limit(256)
    #pragma omp target teams distribute parallel for \
    num_teams(numTeams) thread_limit(threads_per_team) \
      map(to: jelIndex[0:jelCount], \
            V.dim[0:dimCount], \
            V.nGauss2[0:nGauss2Count], \
            V.nDof2[0:nDof2Count], \
            V.x2MinMaxOffset[0:x2MinMaxOffsetCount], \
            V.x2MinMaxAll[0:x2MinMaxAllCount], \
            V.xg2Offset[0:xg2OffsetCount], \
            V.xg2All[0:xg2AllCount], \
            V.w2Offset[0:w2OffsetCount], \
            V.w2All[0:w2AllCount], \
            V.solu2Offset[0:solu2OffsetCount], \
            V.solu2All[0:solu2AllCount], \
            phi2Flat[0:nGauss2_ref*nDof2_ref], \
            offsetMCptr[0:jelCount+1], \
            offJac21Ptr[0:offJac21Size], \
            offJac22Ptr[0:offJac22Size], \
            offRes2Ptr[0:offRes2Size], \
            phi1[0:nDof1], xg1[0:dimSpace], \
            stepData, solu1g, delta, twoWeigh1Kernel, nDof1, eps) \
        map(tofrom: jac21FlatPtr[0:jac21Size], \
                       jac22FlatPtr[0:jac22Size], \
                       res2FlatPtr[0:res2Size], \
                       mCphi2AllPtr[0:totalMC])
    for (unsigned jj = 0; jj < jelCount; ++jj) {

        const unsigned jel = jelIndex[jj];

        const unsigned nDof2 = nDof2Ptr[jel];
        const unsigned offMC = offsetMCptr[jj];
        double* mCphi2i      = mCphi2AllPtr + offMC;

        double* jac21_jel = jac21FlatPtr + offJac21Ptr[jel];
        double* jac22_jel = jac22FlatPtr + offJac22Ptr[jel];
        double* res2_jel  = res2FlatPtr  + offRes2Ptr [jel];

        const unsigned dim     = dimPtr[jel];
        const unsigned nGauss2 = nGauss2Ptr[jel];
        const unsigned baseMinMax = x2MinMaxOffPtr[jel];

        bool coarseIntersectionTest = true;
        for (unsigned k = 0; k < dim; ++k) {
            double xmin = x2MinMaxAllPtr[baseMinMax + 2 * k    ];
            double xmax = x2MinMaxAllPtr[baseMinMax + 2 * k + 1];

            if ((xg1[k] - xmax) > delta + eps || (xmin - xg1[k]) > delta + eps) {
                coarseIntersectionTest = false;
                break;
            }
        }

        if (!coarseIntersectionTest) continue;

        const unsigned baseXg2   = xg2OffPtr[jel];
        const unsigned baseW2    = w2OffPtr[jel];
        const unsigned baseSolu2 = solu2OffPtr[jel];

        // ensure mCphi2i is zeroed
        for (unsigned i = 0; i < nDof2; ++i) mCphi2i[i] = 0.0;

        for (unsigned jg = 0; jg < nGauss2; ++jg) {
            double xg2_jg[3] = {0.0, 0.0, 0.0};
            for (unsigned k = 0; k < dim; ++k) {
                xg2_jg[k] = xg2AllPtr[baseXg2 + jg * dim + k];
            }

            const double dg1    = interface_distance_ball_raw(xg1, xg2_jg, dim, delta);
            const double U_jjjg = SmoothStepEval(dg1, stepData);

            if (U_jjjg <= 0.0) continue;

            const double w2    = w2AllPtr[baseW2    + jg];
            const double solu2 = solu2AllPtr[baseSolu2 + jg];

            const double C = U_jjjg * w2 * twoWeigh1Kernel;

            const double* phi2_jg = &phi2Flat[jg * nDof2_ref];

            double* jac22pt = jac22_jel;

            for (unsigned i = 0; i < nDof2; ++i) {
                const double cPhi2i = C * phi2_jg[i];
                mCphi2i[i] -= cPhi2i;

                const double* phi2pt = phi2_jg;
                for (unsigned j = 0; j < nDof2; ++j, ++phi2pt, ++jac22pt) {
                    *jac22pt -= cPhi2i * (*phi2pt);
                }

                res2_jel[i] += cPhi2i * solu2;
            }
        }

        unsigned ijIndex = 0;
        for (unsigned i = 0; i < nDof2; ++i) {
            const double mSum = mCphi2i[i];
            for (unsigned j = 0; j < nDof1; ++j, ++ijIndex) {
                jac21_jel[ijIndex] -= mSum * phi1[j];
            }
            res2_jel[i] += mSum * solu1g;
        }
    }

    return area;
}





#endif






