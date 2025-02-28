#pragma once
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include "MultiLevelSolution.hpp"


using namespace femus;

class Region {
  private:
    unsigned _size;

    std::vector<std::vector<std::vector<double>>> _x;
    std::vector<std::vector<std::vector<double>>> _minmax;
    std::vector<std::vector<unsigned>> _l2Gmap;
    std::vector<std::vector<double>> _sol;
    std::vector < const elem_type *> _fem;

    std::vector<std::vector<double>> _weight;
    std::vector<std::vector<std::vector<double>>> _xg;
    std::vector<std::vector<double>> _solug;
    std::vector<std::vector<double>> _I2;
    std::vector<unsigned> _iel;


  public:

    void Reset() {
      _size = 0;
    }

    Region(const unsigned reserveSize = 0) {
      _size = 0;
      _x.reserve(reserveSize);
      _l2Gmap.reserve(reserveSize);
      _sol.reserve(reserveSize);
      _fem.reserve(reserveSize);
      _minmax.reserve(reserveSize);
      _weight.reserve(reserveSize);
      _I2.reserve(reserveSize);
      _iel.reserve(reserveSize);
      _xg.reserve(reserveSize);
      _solug.reserve(reserveSize);
    }

    void AddElement(const unsigned &iel,
                    const std::vector<std::vector<double>>&x,
                    const std::vector<unsigned> &l2GMap,
                    const std::vector<double> &sol,
                    const elem_type *fem,
                    const std::vector<std::vector<double>>&minmax,
                    const double &I2value = 1.) {
      _size++;

      _x.resize(_size);
      _l2Gmap.resize(_size);
      _sol.resize(_size);
      _fem.resize(_size);
      _minmax.resize(_size);
      _weight.resize(_size);
      _I2.resize(_size);
      _iel.resize(_size);
      _xg.resize(_size);
      _solug.resize(_size);

      unsigned jel = _size - 1;
      _iel[jel] = iel;
      _x[jel] = x;
      _l2Gmap[jel] = l2GMap;
      _sol[jel] = sol;
      _fem[jel] = fem;
      _minmax[jel] = minmax;

      _weight[jel].resize(fem->GetGaussPointNumber());
      _I2[jel].assign(fem->GetGaussPointNumber(), I2value);
      _xg[jel].resize(fem->GetGaussPointNumber());
      _solug[jel].resize(fem->GetGaussPointNumber());

      const double *phi;
      const double *phipt;
      std::vector<double>::const_iterator soluit;
      std::vector < std::vector<double> > ::const_iterator xvit;
      std::vector<double>::const_iterator xvkit;
      std::vector<double>::iterator xgit;

      for (unsigned jg = 0; jg < fem->GetGaussPointNumber(); jg++) {

        fem->GetGaussQuantities(x, jg, _weight[jel][jg], phi);

        _solug[jel][jg] = 0.;
        for (soluit = sol.begin(), phipt = phi; soluit != sol.end(); soluit++, phipt++) {
          _solug[jel][jg] += (*soluit) * (*phipt);
        }

        _xg[jel][jg].assign(x.size(), 0.);
        for (xgit = _xg[jel][jg].begin(), xvit = x.begin(); xgit != _xg[jel][jg].end(); xgit++, xvit++) {
          for (xvkit = (*xvit).begin(), phipt = phi;  xvkit != (*xvit).end(); phipt++, xvkit++) {
            *xgit += (*xvkit) * (*phipt);
          }
        }
      }
    }

    void AddElement(const unsigned &iel,
                    const std::vector<std::vector<double>>&x,
                    const std::vector<unsigned> &l2GMap,
                    const std::vector<double> &sol,
                    const elem_type *fem,
                    const std::vector<std::vector<double>>&minmax,
                    const double *I2) {

      AddElement(iel, x, l2GMap, sol, fem, minmax);
      unsigned jel = _size - 1;
      for (unsigned jg = 0; jg < _I2[jel].size(); jg++) _I2[jel][jg] = I2[jg];

    }

    const unsigned& GetElementNumber(const unsigned &jel) const {
      return _iel[jel];
    }

    const std::vector<std::vector<double>>& GetGaussCoordinates(const unsigned &jel) const {
      return _xg[jel];
    }

    const std::vector<double>& GetGaussWeight(const unsigned &jel) const {
      return _weight[jel];
    }

    const std::vector<double>& GetI2(const unsigned &jel) const {
      return _I2[jel];
    }

    void AddI2(const unsigned &jel, const unsigned &ig, const double &I2) {
      _I2[jel][ig] += I2;
    }


    const std::vector<double>& GetGaussSolution(const unsigned &jel) const {
      return _solug[jel];
    }

    const std::vector<std::vector<double>>& GetCoordinates(const unsigned &jel) const {
      return _x[jel];
    }

    const std::vector<std::vector<double>>& GetMinMax(const unsigned &jel) const {
      return _minmax[jel];
    }

    const std::vector<unsigned>& GetMapping(const unsigned &jel) const {
      return _l2Gmap[jel];
    }

    const std::vector<double>& GetSolution(const unsigned &jel) const {
      return _sol[jel];
    }

    const elem_type *GetFem(const unsigned &jel) const {
      return _fem[jel];
    }

    const unsigned &size() const {
      return _size;
    }

    const unsigned GetDimension(const unsigned &jel) const {
      return _x[jel].size();
    }

    const unsigned GetDofNumber(const unsigned &jel) const {
      return _l2Gmap[jel].size();
    }

};


#include <boost/math/special_functions/sign.hpp>
#include "RefineElement.hpp"
#include "NonLocal.hpp"

//THIS IS THE 2D ASSEMBLY FOR THE NONLOCAL DIFFUSION PROBLEM with ADAPTIVE QUADRATURE RULE





using namespace femus;

// double GetExactSolutionValue(const std::vector < double >& x) {
//   double pi = acos(-1.);
//   return cos(pi * x[0]) * cos(pi * x[1]);
// };
//
// void GetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
//   double pi = acos(-1.);
//   solGrad[0]  = -pi * sin(pi * x[0]) * cos(pi * x[1]);
//   solGrad[1] = -pi * cos(pi * x[0]) * sin(pi * x[1]);
// };
//
// double GetExactSolutionLaplace ( const std::vector < double >& x )
// {
//     double pi = acos ( -1. );
//     return -pi * pi * cos ( pi * x[0] ) * cos ( pi * x[1] ) - pi * pi * cos ( pi * x[0] ) * cos ( pi * x[1] );
// };

bool nonLocalAssembly = true;

//DELTA sizes: martaTest1: 0.4, martaTest2: 0.01, martaTest3: 0.53, martaTest4: 0.2, maxTest1: both 0.4, maxTest2: both 0.01, maxTest3: both 0.53, maxTest4: both 0.2, maxTest5: both 0.1, maxTest6: both 0.8,  maxTest7: both 0.05, maxTest8: both 0.025, maxTest9: both 0.0125, maxTest10: both 0.00625

//double delta1 = 0.2; //cubic, quartic, consistency
double delta1 = 0.2; //parallel
double delta2 = 0.2;
// double epsilon = ( delta1 > delta2 ) ? delta1 : delta2;
double kappa1 = 1.;
double kappa2 = 1.;

double A1 = 1. / 16.;
double B1 = - 1. / 8.;
double A2 = 1. / 16.;
double B2 = - 1. / 24.;

// === New variables for adaptive assembly ===
double xc = -1;
double yc = -1;

const unsigned mySwap[4][9] = {
  {0, 1, 2, 3, 4, 5, 6, 7, 8},
  {3, 0, 1, 2, 7, 4, 5, 6, 8},
  {2, 3, 0, 1, 6, 7, 4, 5, 8},
  {1, 2, 3, 0, 5, 6, 7, 4, 8}
};

const unsigned mySwapI[4][9] = {
  {0, 1, 2, 3, 4, 5, 6, 7, 8},
  {1, 2, 3, 0, 5, 6, 7, 4, 8},
  {2, 3, 0, 1, 6, 7, 4, 5, 8},
  {3, 0, 1, 2, 7, 4, 5, 6, 8}
};

// === New variables for adaptive assembly ===

void GetBoundaryFunctionValue(double &value, const std::vector < double >& x) {

  //   double u1 = A1 + B1 * x[0] - 1. / (2. * kappa1) * x[0] * x[0] ;
  //   double u2 = A2 + B2 * x[0] - 1. / (2. * kappa2) * x[0] * x[0] ;

  double u1 = (A1 + B1 * x[0] - 1. / (2. * kappa1) * x[0] * x[0]) * (1. + x[0] * x[0]) * cos(x[1]) ;
  double u2 = (A2 + B2 * x[0] - 1. / (2. * kappa2) * x[0] * x[0]) * cos(x[0]) * cos(x[1]);

  value = (x[0] < 0.) ? u1 : u2;

//     value = 0.;
//     value = x[0];
//     value = x[0] * x[0];
//     value = ( x[0] < 0. ) ? x[0] * x[0] * x[0] : 3 * x[0] * x[0] * x[0];
//     value = x[0] * x[0] * x[0] + x[1] * x[1] * x[1];
//     value = x[0] * x[0] * x[0] * x[0] + 0.1 * x[0] * x[0];
//     value = x[0] * x[0] * x[0] * x[0];
//     value =  2 * x[0] + x[0] * x[0] * x[0] * x[0] * x[0]; //this is 2x + x^5


}

unsigned ReorderElement(std::vector < int > &dofs, std::vector < double > & sol, std::vector < std::vector < double > > & x);

void RectangleAndBallRelation(bool &theyIntersect, const std::vector<double> &ballCenter, const double &ballRadius, const std::vector < std::vector < double> > &elementCoordinates,  std::vector < std::vector < double> > &newCoordinates);

void RectangleAndBallRelation2(bool & theyIntersect, const std::vector<double> &ballCenter, const double & ballRadius, const std::vector < std::vector < double> > &elementCoordinates, std::vector < std::vector < double> > &newCoordinates);


//BEGIN New functions: GetIntegral on refined mesh (needs the RefineElement class)

void AssembleNonLocalRefined(MultiLevelProblem& ml_prob) {

  LinearImplicitSystem* mlPdeSys;

  mlPdeSys = &ml_prob.get_system<LinearImplicitSystem> ("NonLocal");

  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*                    msh = ml_prob._ml_msh->GetLevel(level);
  elem*                     el = msh->el;

  MultiLevelSolution*    mlSol = ml_prob._ml_sol;
  Solution*                sol = ml_prob._ml_sol->GetSolutionLevel(level);

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
  SparseMatrix*            KK = pdeSys->_KK;
  NumericVector*           RES = pdeSys->_RES;

  Region region2(10);

  const unsigned  dim = msh->GetDimension();

  unsigned iproc = msh->processor_id(); // get the process_id (for parallel computation)
  unsigned nprocs = msh->n_processors(); // get the noumber of processes (for parallel computation)

  unsigned soluIndex;
  unsigned soluPdeIndex;

  soluIndex = mlSol->GetIndex("u");    // get the position of "u" in the ml_sol object
  soluPdeIndex = mlPdeSys->GetSolPdeIndex("u");    // get the position of "u" in the pdeSys object
  unsigned soluType = mlSol->GetSolutionType(soluIndex);    // get the finite element type for "u"

  unsigned cntIndex = mlSol->GetIndex("cnt");    // get the position of "u" in the ml_sol object

  std::vector < double >  solu1; // local solution for the nonlocal assembly
  std::vector < double >  solu2; // local solution for the nonlocal assembly

  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)
  std::vector < vector < double > > x1(dim);
  std::vector < vector < double > > x2(dim);

  std::vector< unsigned > l2GMap1; // local to global mapping
  std::vector< unsigned > l2GMap2; // local to global mapping

  std::vector < std::vector < double > > xg1;
  std::vector <double> weight1;
  std::vector <const double *> phi1x;

  std::vector < std::pair<std::vector<double>::iterator, std::vector<double>::iterator> > x1MinMax(dim);
  std::vector < std::pair<std::vector<double>::iterator, std::vector<double>::iterator> > x2MinMax(dim);
  std::vector < std::vector <double> > minmax2;

  //BEGIN setup for adaptive integration

  //unsigned lmax1 = 2; // cubic or quartic
  unsigned lmin1 = 1;
  if (lmin1 > lmax1 - 1) lmin1 = lmax1 - 1;


//   //consistency
//   double dMax = 0.1;
//   double eps = 0.125 * dMax *  pow(0.75, lmax1 - 3);

  //cubic
  double dMax = (cutFem) ? 0. : 0.1 * pow(2. / 3., level - 1); //marta4, tri unstructured
// double dMax = 0.1 * pow(2./3., level + 1); //marta4Fine
  double eps = 0.125 * dMax;

  //quartic
  //double dMax = 0.1 * pow(2./3., level - 1); //marta4, tri unstructured
  //double dMax = 0.1 * pow(2./3., level + 1); //marta4Fine
  //double eps = 0.125 * dMax;

  //parallel
  //double dMax = 0.0125 * pow(1./2., level); //marta4finer
  //double eps = 0.5 * dMax;



// //consistency 3D
  // double dMax = 0.1;
  // double eps = 0.125 * dMax *  pow(0.75, lmax1 - 3);


//   //convergence 3D
//   double dMax = 0.1 * pow(2. / 3., level - 1); //marta4-3D
//   //double dMax = 0.1 * pow(2. / 3., level); //marta4-3D-fine
//   double eps = 0.125 * dMax;

  double areaEl = pow(0.1 * pow(1. / 2., level - 1), dim);

  std::cout << "level = " << level << " ";

  std::cout << "EPS = " << eps << " " << "delta1 = " << delta1 + eps << " " << " lmax1 = " << lmax1 << " lmin1 = " << lmin1 << std::endl;

  RefineElement *refineElement[6][3];
  //RefineElement *refineElementCF[6][3];

  NonLocal *nonlocal;

  if (dim == 3) {
    refineElement[0][0] = new RefineElement(lmax1, "hex", "linear", "third", "third", "legendre");
    refineElement[0][1] = new RefineElement(lmax1, "hex", "quadratic", "third", "third", "legendre");
    refineElement[0][2] = new RefineElement(lmax1, "hex", "biquadratic", "third", "third", "legendre");

    refineElement[1][0] = new RefineElement(lmax1, "tet", "linear", "third", "third", "legendre");
    refineElement[1][1] = new RefineElement(lmax1, "tet", "quadratic", "third", "third", "legendre");
    refineElement[1][2] = new RefineElement(lmax1, "tet", "biquadratic", "third", "third", "legendre");

    refineElement[0][soluType]->SetConstants(eps);
    refineElement[1][soluType]->SetConstants(eps);

    nonlocal = new NonLocalBall3D();

  }
  else if (dim == 2) {
    refineElement[3][0] = new RefineElement(lmax1, "quad", "linear", "fifth", "fifth", "legendre");
    refineElement[3][1] = new RefineElement(lmax1, "quad", "quadratic", "fifth", "fifth", "legendre");
    refineElement[3][2] = new RefineElement(lmax1, "quad", "biquadratic", "fifth", "fifth", "legendre");

    refineElement[4][0] = new RefineElement(lmax1, "tri", "linear", "fifth", "fifth", "legendre");
    refineElement[4][1] = new RefineElement(lmax1, "tri", "quadratic", "fifth", "fifth", "legendre");
    refineElement[4][2] = new RefineElement(lmax1, "tri", "biquadratic", "fifth", "fifth", "legendre");

    refineElement[3][soluType]->SetConstants(eps);
    refineElement[4][soluType]->SetConstants(eps);

    nonlocal = new NonLocalBall();
    //nonlocal = new NonLocalBox();
  }

  nonlocal->SetKernel(kappa1, delta1, correctConstant * eps);

  unsigned offset = msh->_elementOffset[iproc];
  unsigned offsetp1 = msh->_elementOffset[iproc + 1];

  MyVector <unsigned> rowSize(msh->_elementOffset);
  for (unsigned jel = offset; jel < offsetp1; jel++) {
    unsigned jelGeom = msh->GetElementType(jel);
    const elem_type* fem2 = refineElement[jelGeom][soluType]->GetFem2();
    rowSize[jel] = fem2->GetGaussPointNumber();
  }
  MyMatrix <double> I2(rowSize);

  std::vector <double> kprocMinMax(nprocs * dim * 2);
  for (unsigned k = 0; k < dim; k++) {
    unsigned kk = iproc * (dim * 2) + k * 2;
    kprocMinMax[kk] = 1.0e10;
    kprocMinMax[kk + 1] = -1.0e10;
  }
  for (unsigned iel = offset; iel < offsetp1; iel++) {
    unsigned nDof  = msh->GetElementDofNumber(iel, xType);
    for (unsigned i = 0; i < nDof; i++) {
      unsigned iDof  = msh->GetSolutionDof(i, iel, xType);
      for (unsigned k = 0; k < dim; k++) {
        unsigned kk = iproc * (dim * 2) + k * 2;
        double xk = (*msh->_topology->_Sol[k])(iDof);
        if (xk < kprocMinMax[kk]) kprocMinMax[kk] = xk;
        if (xk > kprocMinMax[kk + 1]) kprocMinMax[kk + 1] = xk;
      }
    }
  }

  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    MPI_Bcast(&kprocMinMax[kproc * dim * 2], dim * 2, MPI_DOUBLE, kproc, PETSC_COMM_WORLD);
  }

  //BEGIN Search and Exchange of overlapping quantities and RHS evaluation

  std::vector < std::vector < unsigned > > orElements(nprocs);
  std::vector < unsigned > orCntSend(nprocs, 0);
  std::vector < unsigned > orCntRecv(nprocs, 0);
  std::vector < unsigned > orSizeSend(nprocs, 0);
  std::vector < unsigned > orSizeRecv(nprocs, 0);
  std::vector < std::vector < unsigned > > orGeomSend(nprocs);
  std::vector < std::vector < unsigned > > orGeomRecv(nprocs);

  std::vector < std::vector < unsigned > > orDofsSend(nprocs);
  std::vector < std::vector < unsigned > > orDofsRecv(nprocs);

  std::vector < std::vector < double > > orSolSend(nprocs);
  std::vector < std::vector < double > > orSolRecv(nprocs);

  std::vector < std::vector < std::vector < double > > > orXSend(nprocs);
  std::vector < std::vector < std::vector < double > > > orXRecv(nprocs);

  unsigned sizeAll = (offsetp1 - offset) * pow(3, dim);

  time_t exchangeTime = clock();

  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    orElements[kproc].resize(offsetp1 - offset);
    orGeomSend[kproc].resize(offsetp1 - offset);
    orDofsSend[kproc].resize(sizeAll);
    orSolSend[kproc].resize(sizeAll);

    orXSend[kproc].resize(dim);
    orXRecv[kproc].resize(dim);

    for (unsigned k = 0; k < dim; k++) {
      orXSend[kproc][k].resize(sizeAll);
    }
  }

  RES->zero();

  for (unsigned iel = offset; iel < offsetp1; iel++) {

    unsigned ielGeom = msh->GetElementType(iel);
    unsigned nDof1  = msh->GetElementDofNumber(iel, soluType);

    l2GMap1.resize(nDof1);
    solu1.resize(nDof1);
    for (unsigned k = 0; k < dim; k++) {
      x1[k].resize(nDof1);
    }

    for (unsigned i = 0; i < nDof1; i++) {

      unsigned uDof = msh->GetSolutionDof(i, iel, soluType);
      solu1[i] = (*sol->_Sol[soluIndex])(uDof);

      l2GMap1[i] = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);

      unsigned xDof  = msh->GetSolutionDof(i, iel, xType);
      for (unsigned k = 0; k < dim; k++) {
        x1[k][i] = (*msh->_topology->_Sol[k])(xDof);
      }
    }

    refineElement[ielGeom][soluType]->InitElement1(x1, lmax1);

    //assemble and store RHS
    std::vector <double> res1(nDof1, 0.);
    double weight1;
    const double *phi1;
    const elem_type* fem1 = refineElement[ielGeom][soluType]->GetFem1();
    for (unsigned ig = 0; ig < fem1->GetGaussPointNumber(); ig++) {
      fem1->GetGaussQuantities(x1, ig, weight1, phi1);
      std::vector< double > x1g(dim, 0);
      for (unsigned i = 0; i < nDof1; i++) {
        for (unsigned k = 0; k < dim; k++) {
          x1g[k] += x1[k][i] * phi1[i];
        }
      }
      for (unsigned i = 0; i < nDof1; i++) {

        for (unsigned k = 0; k < dim; k++) {
          // res1[i] -= -2 * phi1[i] * weight1; // consistency
          res1[i] -= -6.* x1g[k] * phi1[i] * weight1; //cubic
//         res1[i] -= ( -12.* x1g[k] * x1g[k] - delta1 * delta1 ) * phi1[i] * weight1; //quartic
        }
      }
    }
    RES->add_vector_blocked(res1, l2GMap1);

    for (unsigned k = 0; k < dim; k++) {
      x1MinMax[k] = std::minmax_element(x1[k].begin(), x1[k].end());
    }

    for (unsigned kproc = 0; kproc < nprocs; kproc++) {
      bool coarseIntersectionTest = true;
      for (unsigned k = 0; k < dim; k++) {
        unsigned kk = kproc * dim * 2 + k * 2;
        if ((*x1MinMax[k].first  - kprocMinMax[kk + 1]) > delta1 + eps  || (kprocMinMax[kk]  - *x1MinMax[k].second) > delta1 + eps) {
          coarseIntersectionTest = false;
          break;
        }
      }
      if (coarseIntersectionTest) {
        orElements[kproc][orCntSend[kproc]] = iel;
        orGeomSend[kproc][orCntSend[kproc]] = ielGeom;
        orCntSend[kproc]++;
        for (unsigned i = 0; i < nDof1; i++) {
          orDofsSend[kproc][orSizeSend[kproc] + i] = l2GMap1[i];
          orSolSend[kproc][orSizeSend[kproc] + i] = solu1[i];
          for (unsigned k = 0; k < dim; k++) {
            orXSend[kproc][k][orSizeSend[kproc] + i] = x1[k][i];
          }
        }
        orSizeSend[kproc] += nDof1;
      }
    }
  }

  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    orElements[kproc].resize(orCntSend[kproc]);
    orGeomSend[kproc].resize(orCntSend[kproc]);
    orDofsSend[kproc].resize(orSizeSend[kproc]);
    orSolSend[kproc].resize(orSizeSend[kproc]);
    for (unsigned k = 0; k < dim; k++) {
      orXSend[kproc][k].resize(orSizeSend[kproc]);
    }
  }

  std::vector < std::vector < MPI_Request > >  reqsSend(nprocs) ;
  std::vector < std::vector < MPI_Request > >  reqsRecv(nprocs) ;
  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    reqsSend[kproc].resize(3 + dim);
    reqsRecv[kproc].resize(3 + dim);
  }

  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    MPI_Irecv(&orCntRecv[kproc], 1, MPI_UNSIGNED, kproc, 0, PETSC_COMM_WORLD, &reqsRecv[kproc][0]);
    MPI_Irecv(&orSizeRecv[kproc], 1, MPI_UNSIGNED, kproc, 1, PETSC_COMM_WORLD, &reqsRecv[kproc][1]);
  }

  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    MPI_Isend(&orCntSend[kproc], 1, MPI_UNSIGNED, kproc, 0, PETSC_COMM_WORLD, &reqsSend[kproc][0]);
    MPI_Isend(&orSizeSend[kproc], 1, MPI_UNSIGNED, kproc, 1, PETSC_COMM_WORLD, &reqsSend[kproc][1]);
  }

  //wait and check that all the sends and receives have been completed successfully
  MPI_Status status;
  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    for (unsigned m = 0; m < 2; m++) {
      int test = MPI_Wait(&reqsRecv[kproc][m], &status);
      if (test != MPI_SUCCESS) {
        abort();
      }
      test = MPI_Wait(&reqsSend[kproc][m], &status);
      if (test != MPI_SUCCESS) {
        abort();
      }
    }
  }

  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    orGeomRecv[kproc].resize(orCntRecv[kproc]);
    orDofsRecv[kproc].resize(orSizeRecv[kproc]);
    orSolRecv[kproc].resize(orSizeRecv[kproc]);
    for (unsigned k = 0; k < dim; k++) {
      orXRecv[kproc][k].resize(orSizeRecv[kproc]);
    }
  }

  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    MPI_Irecv(orGeomRecv[kproc].data(), orGeomRecv[kproc].size(), MPI_UNSIGNED, kproc, 0, PETSC_COMM_WORLD, &reqsRecv[kproc][0]);
    MPI_Irecv(orDofsRecv[kproc].data(), orDofsRecv[kproc].size(), MPI_UNSIGNED, kproc, 1, PETSC_COMM_WORLD, &reqsRecv[kproc][1]);
    MPI_Irecv(orSolRecv[kproc].data(), orSolRecv[kproc].size(), MPI_DOUBLE, kproc, 2, PETSC_COMM_WORLD, &reqsRecv[kproc][2]);
    for (unsigned k = 0; k < dim; k++) {
      MPI_Irecv(orXRecv[kproc][k].data(), orXRecv[kproc][k].size(), MPI_DOUBLE, kproc, 3 + k, PETSC_COMM_WORLD, &reqsRecv[kproc][3 + k]);
    }
  }

  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    MPI_Isend(orGeomSend[kproc].data(), orGeomSend[kproc].size(), MPI_UNSIGNED, kproc, 0, PETSC_COMM_WORLD, &reqsSend[kproc][0]);
    MPI_Isend(orDofsSend[kproc].data(), orDofsSend[kproc].size(), MPI_UNSIGNED, kproc, 1, PETSC_COMM_WORLD, &reqsSend[kproc][1]);
    MPI_Isend(orSolSend[kproc].data(), orSolSend[kproc].size(), MPI_DOUBLE, kproc, 2, PETSC_COMM_WORLD, &reqsSend[kproc][2]);
    for (unsigned k = 0; k < dim; k++) {
      MPI_Isend(orXSend[kproc][k].data(), orXSend[kproc][k].size(), MPI_DOUBLE, kproc, 3 + k, PETSC_COMM_WORLD, &reqsSend[kproc][3 + k]);
    }
  }

  //wait and check that all the sends and receives have been completed successfully
  for (unsigned kproc = 0; kproc < nprocs; kproc++) {
    for (unsigned m = 0; m < 3 + dim; m++) {
      int test = MPI_Wait(&reqsRecv[kproc][m], &status);
      if (test != MPI_SUCCESS) {
        abort();
      }
      test = MPI_Wait(&reqsSend[kproc][m], &status);
      if (test != MPI_SUCCESS) {
        abort();
      }
    }
  }
  std::cout << "[" << iproc << "]  ";
  std::cout << "Parallel Exchange Time and RHS evaluation = " << static_cast<double>(clock() - exchangeTime) / CLOCKS_PER_SEC << std::endl;
  std::cout << std::endl;

  //END Search and Exchange of overlapping quantities


  if (cutFem * correctConstant) {
    //BEGIN corrective moment Constant evaluation
    std::cout << "Corrective moment constant evaluation\n";

    time_t pSearchTime = 0.;
    time_t pAssemblyTime = 0.;

    for (unsigned kproc = 0; kproc < nprocs; kproc++) {
      unsigned cnt1 = 0;
      for (unsigned iel = 0; iel < orGeomRecv[kproc].size(); iel++) { // these elements are not own by iproc

        std::cout << "\r[" << iproc << "] " << iel << " out of " << orGeomRecv[kproc].size() <<  " on [" << kproc << "]" << std::flush;

        short unsigned ielGeom = orGeomRecv[kproc][iel];
        unsigned nDof1  = el->GetNVE(ielGeom, soluType);

        l2GMap1.resize(nDof1);
        solu1.resize(nDof1);
        for (unsigned k = 0; k < dim; k++) {
          x1[k].resize(nDof1);
        }

        for (unsigned i = 0; i < nDof1; i++) {
          solu1[i] = orSolRecv[kproc][cnt1 + i];
          l2GMap1[i] = orDofsRecv[kproc][cnt1 + i];
          for (unsigned k = 0; k < dim; k++) {
            x1[k][i] =  orXRecv[kproc][k][cnt1 + i];
          }
        }

        refineElement[ielGeom][soluType]->InitElement1(x1, lmax1);
        for (unsigned k = 0; k < dim; k++) {
          x1MinMax[k] = std::minmax_element(x1[k].begin(), x1[k].end());
        }

        cnt1 += nDof1;
        time_t start = clock();

        region2.Reset();

        unsigned cnt2 = 0;
        for (unsigned jel = 0; jel < orGeomSend[kproc].size(); jel++) { // these elements are own by iproc

          short unsigned jelGeom = orGeomSend[kproc][jel];
          unsigned nDof2  = el->GetNVE(jelGeom, soluType);

          for (unsigned k = 0; k < dim; k++) {
            x2[k].assign(nDof2, 0.);
          }
          for (unsigned j = 0; j < nDof2; j++) {
            for (unsigned k = 0; k < dim; k++) {
              x2[k][j] = orXSend[kproc][k][cnt2 + j];
            }
          }
          minmax2.resize(dim);
          for (unsigned k = 0; k < dim; k++) {
            minmax2[k].resize(2);
            x2MinMax[k] = std::minmax_element(x2[k].begin(), x2[k].end());
            minmax2[k][0] = *x2MinMax[k].first;
            minmax2[k][1] = *x2MinMax[k].second;
          }
          bool coarseIntersectionTest = true;
          for (unsigned k = 0; k < dim; k++) {
            if ((*x1MinMax[k].first  - *x2MinMax[k].second) > delta1 + eps  || (*x2MinMax[k].first  - *x1MinMax[k].second) > delta1 + eps) {
              coarseIntersectionTest = false;
              break;
            }
          }

          if (coarseIntersectionTest) {
            l2GMap2.resize(nDof2);
            solu2.resize(nDof2);
            for (unsigned j = 0; j < nDof2; j++) {
              solu2[j] = orSolSend[kproc][cnt2 + j];
              l2GMap2[j] = orDofsSend[kproc][cnt2 + j];
            }
            region2.AddElement(orElements[kproc][jel], x2, l2GMap2, solu2, refineElement[jelGeom][soluType]->GetFem2(), minmax2, 0.);
          }
          cnt2 += nDof2;
        }

        pSearchTime += clock() - start;
        start = clock();

        if (region2.size() > 0) {
          nonlocal->ZeroLocalQuantities(nDof1, region2, lmax1);
          bool printMesh = false;

          std::vector<unsigned>jelIndex(region2.size());
          for (unsigned j = 0; j < jelIndex.size(); j++) {
            jelIndex[j] = j;
          }

          nonlocal->AssemblyCutFemI2(0, lmin1, lmax1, 0,
                                     refineElement[ielGeom][soluType]->GetOctTreeElement1(), refineElement[ielGeom][soluType]->GetOctTreeElement1CF(),
                                     *refineElement[ielGeom][soluType], region2, jelIndex, solu1, kappa1, delta1, printMesh);

          for (unsigned jel = 0; jel < region2.size(); jel++) {
            const std::vector<double>& I2jel = region2.GetI2(jel);
            for (unsigned jg = 0; jg < region2.GetFem(jel)->GetGaussPointNumber(); jg++) {
              I2[region2.GetElementNumber(jel)][jg] += I2jel[jg];
            }
          }
        }
        pAssemblyTime += clock() - start;
      }//end iel loop
      if(orGeomRecv[kproc].size()!=0) std::cout << std::endl;

    }

    std::cout << "[" << iproc << "] ";
    std::cout << "I2 Search Time = " << static_cast<double>(pSearchTime) / CLOCKS_PER_SEC << std::endl;
    std::cout << "[" << iproc << "] ";
    std::cout << "I2 Assembly Time = " << static_cast<double>(pAssemblyTime) / CLOCKS_PER_SEC << std::endl;
    std::cout << std::endl;

    double I2real = (dim == 2) ? 0.5 * M_PI * pow(delta1, 4) : 4. / 5. * M_PI * pow(delta1, 5);

    for (unsigned jel = offset; jel < offsetp1; jel++) {
      unsigned jelGroup = msh->GetElementGroup(jel);
      if (jelGroup == 5) {
        for (unsigned j = 0; j < I2.size(jel); j++) {
          I2[jel][j] = 1.;
        }
      }
      else {
        for (unsigned j = 0; j < I2.size(jel); j++) {
          I2[jel][j] = I2real / I2[jel][j];
        }
      }
    }
    // std::cout<<I2;
    //END parallel corrective moment Constant evaluation
  }

  std::cout << "Nonlocal Assembly\n";
  sol->_Sol[cntIndex]->zero();
  KK->zero(); // Set to zero all the entries of the Global Matrix
  //BEGIN nonlocal assembly

  time_t pSearchTime = 0.;
  time_t pAssemblyTime = 0.;

  std::vector<unsigned > procOrder(nprocs);

  procOrder[0] = iproc;
  for (unsigned i = 1; i <= iproc; i++) {
    procOrder[i] = i - 1;
  }
  for (unsigned i = iproc + 1; i < procOrder.size(); i++) {
    procOrder[i] = i;
  }

  for (unsigned i = 1; i < procOrder.size() - 1; i++) {
    for (unsigned j = i + 1; j < procOrder.size(); j++) {
      if (orGeomRecv[procOrder[i]].size() < orGeomRecv[procOrder[j]].size()) {
        unsigned procOrderi = procOrder[i];
        procOrder[i] = procOrder[j];
        procOrder[j] = procOrderi;
      }
    }
  }

  for (unsigned lproc = 0; lproc < nprocs; lproc++) {
    unsigned kproc = procOrder[lproc];
    unsigned cnt1 = 0;
    for (unsigned iel = 0; iel < orGeomRecv[kproc].size(); iel++) { // these elements are not own by iproc

      std::cout << "\r[" << iproc << "] " << iel << " out of " << orGeomRecv[kproc].size() <<  " on [" << kproc << "]" << std::flush;

      short unsigned ielGeom = orGeomRecv[kproc][iel];
      unsigned nDof1  = el->GetNVE(ielGeom, soluType);

      l2GMap1.resize(nDof1);
      solu1.resize(nDof1);
      for (unsigned k = 0; k < dim; k++) {
        x1[k].resize(nDof1);
      }

      for (unsigned i = 0; i < nDof1; i++) {
        solu1[i] = orSolRecv[kproc][cnt1 + i];
        l2GMap1[i] = orDofsRecv[kproc][cnt1 + i];
        for (unsigned k = 0; k < dim; k++) {
          x1[k][i] =  orXRecv[kproc][k][cnt1 + i];
        }
      }

      refineElement[ielGeom][soluType]->InitElement1(x1, lmax1);
      for (unsigned k = 0; k < dim; k++) {
        x1MinMax[k] = std::minmax_element(x1[k].begin(), x1[k].end());
      }

      cnt1 += nDof1;
      time_t start = clock();

      region2.Reset();

      unsigned cnt2 = 0;
      for (unsigned jel = 0; jel < orGeomSend[kproc].size(); jel++) { // these elements are own by iproc

        short unsigned jelGeom = orGeomSend[kproc][jel];
        unsigned nDof2  = el->GetNVE(jelGeom, soluType);

        for (unsigned k = 0; k < dim; k++) {
          x2[k].assign(nDof2, 0.);
        }
        for (unsigned j = 0; j < nDof2; j++) {
          for (unsigned k = 0; k < dim; k++) {
            x2[k][j] = orXSend[kproc][k][cnt2 + j];
          }
        }
        minmax2.resize(dim);
        for (unsigned k = 0; k < dim; k++) {
          minmax2[k].resize(2);
          x2MinMax[k] = std::minmax_element(x2[k].begin(), x2[k].end());
          minmax2[k][0] = *x2MinMax[k].first;
          minmax2[k][1] = *x2MinMax[k].second;
        }
        bool coarseIntersectionTest = true;
        for (unsigned k = 0; k < dim; k++) {
          if ((*x1MinMax[k].first  - *x2MinMax[k].second) > delta1 + eps  || (*x2MinMax[k].first  - *x1MinMax[k].second) > delta1 + eps) {
            coarseIntersectionTest = false;
            break;
          }
        }

        if (coarseIntersectionTest) {

          sol->_Sol[cntIndex]->add(orElements[kproc][jel], 1);

          l2GMap2.resize(nDof2);
          solu2.resize(nDof2);
          for (unsigned j = 0; j < nDof2; j++) {
            solu2[j] = orSolSend[kproc][cnt2 + j];
            l2GMap2[j] = orDofsSend[kproc][cnt2 + j];
          }
          if (cutFem * correctConstant) region2.AddElement(orElements[kproc][jel], x2, l2GMap2, solu2, refineElement[jelGeom][soluType]->GetFem2(), minmax2, I2[orElements[kproc][jel]]);
          else region2.AddElement(orElements[kproc][jel], x2, l2GMap2, solu2, refineElement[jelGeom][soluType]->GetFem2(), minmax2);
        }
        cnt2 += nDof2;
      }

      pSearchTime += clock() - start;
      start = clock();

      if (region2.size() > 0) {
        nonlocal->ZeroLocalQuantities(nDof1, region2, lmax1);
        bool printMesh = false;

        std::vector<unsigned>jelIndex(region2.size());
        for (unsigned j = 0; j < jelIndex.size(); j++) {
          jelIndex[j] = j;
        }


        if (!cutFem) {
          nonlocal->Assembly1(0, lmin1, lmax1, 0, refineElement[ielGeom][soluType]->GetOctTreeElement1(),
                              *refineElement[ielGeom][soluType], region2, jelIndex,
                              solu1, kappa1, delta1, printMesh);
        }
        else {
          nonlocal->AssemblyCutFem1(0, lmin1, lmax1, 0,
                                    refineElement[ielGeom][soluType]->GetOctTreeElement1(), refineElement[ielGeom][soluType]->GetOctTreeElement1CF(),
                                    *refineElement[ielGeom][soluType], region2, jelIndex, solu1, kappa1, delta1, printMesh);
        }
        for (unsigned jel = 0; jel < region2.size(); jel++) {
          /* The rows of J21, J22 and Res2 are mostly own by iproc, while the columns of J21 and J22 are mostly own by kproc
             This is okay, since the rows of the global matrix KK and residual RES belong to iproc, and this should optimize
             the bufferization and exchange of information when closing the KK matrix and the RES vector */

          std::vector<double> & J21 = nonlocal->GetJac21(jel);
          for (unsigned ii = 0; ii < J21.size(); ii++) { // assembly only if one of the entries is different from zero
            if (fabs(J21[ii]) > 1.0e-12 * areaEl) {
              KK->add_matrix_blocked(J21, region2.GetMapping(jel), l2GMap1);
              break;
            }
          }

          KK->add_matrix_blocked(nonlocal->GetJac22(jel), region2.GetMapping(jel), region2.GetMapping(jel));
          RES->add_vector_blocked(nonlocal->GetRes2(jel), region2.GetMapping(jel));
        }
      }
      pAssemblyTime += clock() - start;
    }//end iel loop
    if(orGeomRecv[kproc].size()!=0) std::cout << std::endl;
    KK->flush();
  }

  std::cout << "[" << iproc << "] ";
  std::cout << "Search Time = " << static_cast<double>(pSearchTime) / CLOCKS_PER_SEC << std::endl;
  std::cout << "[" << iproc << "] ";
  std::cout << "Assembly Time = " << static_cast<double>(pAssemblyTime) / CLOCKS_PER_SEC << std::endl;
  std::cout << std::endl;

  time_t start = clock();
  RES->close();
  KK->close();
  std::cout << "[" << iproc << "] ";
  std::cout << "Closing Time = " << static_cast<double>(clock() - start) / CLOCKS_PER_SEC << std::endl;
  std::cout << std::endl;

  //END nonlocal assembly

  sol->_Sol[cntIndex]->close();

  double tolerance = 1.0e-12 * KK->linfty_norm();
  KK->RemoveZeroEntries(tolerance);

//   KK->draw();

  abort();

  delete nonlocal;
  if (dim == 3) {
    delete refineElement[0][0];
    delete refineElement[0][1];
    delete refineElement[0][2];
    delete refineElement[1][0];
    delete refineElement[1][1];
    delete refineElement[1][2];
  }
  else if (dim == 2) {
    delete refineElement[3][0];
    delete refineElement[3][1];
    delete refineElement[3][2];
    delete refineElement[4][0];
    delete refineElement[4][1];
    delete refineElement[4][2];
  }
  // ***************** END ASSEMBLY *******************
}

void AssembleLocalSys(MultiLevelProblem& ml_prob) {

  LinearImplicitSystem* mlPdeSys  = &ml_prob.get_system<LinearImplicitSystem> ("Local");
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*                    msh = ml_prob._ml_msh->GetLevel(level);
  elem*                     el = msh->el;

  MultiLevelSolution*    mlSol = ml_prob._ml_sol;
  Solution*                sol = ml_prob._ml_sol->GetSolutionLevel(level);

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
  SparseMatrix*             KK = pdeSys->_KK;
  NumericVector*           RES = pdeSys->_RES;

  const unsigned  dim = msh->GetDimension();
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27

  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)
  unsigned    nprocs = msh->n_processors(); // get the noumber of processes (for parallel computation)

  unsigned soluIndex = mlSol->GetIndex("u_local");    // get the position of "u" in the ml_sol object
  unsigned soluType = mlSol->GetSolutionType(soluIndex);    // get the finite element type for "u"

  unsigned soluPdeIndex;
  soluPdeIndex = mlPdeSys->GetSolPdeIndex("u_local");    // get the position of "u" in the pdeSys object

  vector < double >  solu; // local solution for the local assembly (it uses adept)
  solu.reserve(maxSize);

  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  vector < vector < double > > x1(dim);

  for(unsigned k = 0; k < dim; k++) {
    x1[k].reserve(maxSize);
  }

  vector <double> phi;  // local test function
  vector <double> phi_x; // local test function first order partial derivatives
  double weight; // gauss point weight

  phi.reserve(maxSize);
  phi_x.reserve(maxSize * dim);

  vector< double > aRes; // local redidual vector
  aRes.reserve(maxSize);

  vector< int > l2GMap1; // local to global mapping
  l2GMap1.reserve(maxSize);

  vector< double > Res1; // local redidual vector
  Res1.reserve(maxSize);

  KK->zero(); // Set to zero all the entries of the Global Matrix

  //BEGIN local assembly

  std::vector<double> Jac;

  // element loop: each process loops only on the elements that owns
  for(int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

    short unsigned ielGeom = msh->GetElementType(iel);
    unsigned nDofu  = msh->GetElementDofNumber(iel, soluType);
    unsigned nDofx = msh->GetElementDofNumber(iel, xType);

    // resize local arrays
    l2GMap1.resize(nDofu);
    solu.resize(nDofu);
    Jac.assign(nDofu * nDofu, 0.);

    for(int i = 0; i < dim; i++) {
      x1[i].resize(nDofx);
    }

    aRes.assign(nDofu, 0);

    // local storage of solution
    for(unsigned i = 0; i < nDofu; i++) {
      unsigned solDof = msh->GetSolutionDof(i, iel, soluType);
      solu[i] = (*sol->_Sol[soluIndex])(solDof);
      l2GMap1[i] = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);
    }

    // local storage of coordinates
    for(unsigned i = 0; i < nDofx; i++) {
      unsigned xDof  = msh->GetSolutionDof(i, iel, xType);
      for(unsigned jdim = 0; jdim < dim; jdim++) {
        x1[jdim][i] = (*msh->_topology->_Sol[jdim])(xDof);
      }
    }

    // start a new recording of all the operations involving adept::adouble variables


    // *** Gauss point loop ***
    for(unsigned ig = 0; ig < msh->_finiteElement[ielGeom][soluType]->GetGaussPointNumber(); ig++) {
      // *** get gauss point weight, test function and test function partial derivatives ***
      msh->_finiteElement[ielGeom][soluType]->Jacobian(x1, ig, weight, phi, phi_x, boost::none);

      // evaluate the solution, the solution derivatives and the coordinates in the gauss point

      vector < double > gradSolu_gss(dim, 0.);
      vector < double > x_gss(dim, 0.);

      for(unsigned i = 0; i < nDofu; i++) {
        for(unsigned jdim = 0; jdim < dim; jdim++) {
          gradSolu_gss[jdim] += phi_x[i * dim + jdim] * solu[i];
          x_gss[jdim] += x1[jdim][i] * phi[i];
        }
      }

      double aCoeff = (x_gss[0] < 0) ? kappa1 : kappa2;

      // *** phi_i loop ***
      for(unsigned i = 0; i < nDofu; i++) {

        double laplace = 0.;

        for(unsigned jdim = 0; jdim < dim; jdim++) {
          laplace   +=  aCoeff * phi_x[i * dim + jdim] * gradSolu_gss[jdim];
        }

        double srcTerm = 0.;

        for(unsigned k = 0; k < dim; k++) {
          // srcTerm +=  -2. ; // so f = - 2 //consistency
          srcTerm +=  -6. * x_gss[k] ; // cubic
//          srcTerm +=  -12.* x_gss[k] * x_gss[k]; //quartic
        }
        aRes[i] -= (-srcTerm * phi[i] + laplace) * weight;

        for(unsigned j = 0; j < nDofu; j++) {
          for(unsigned jdim = 0; jdim < dim; jdim++) {
            Jac[i * nDofu + j] += aCoeff * phi_x[i * dim + jdim] * phi_x[j * dim + jdim] * weight;
          }
        }

      } // end phi_i loop
    } // end gauss point loop

    //--------------------------------------------------------------------------------------------------------
    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store

    RES->add_vector_blocked(aRes, l2GMap1);
    KK->add_matrix_blocked(Jac, l2GMap1, l2GMap1);

  } //end element loop for each process

  //END local assembly

  RES->close();
  KK->close();

  // ***************** END ASSEMBLY *******************
}








void RectangleAndBallRelation(bool & theyIntersect, const std::vector<double> &ballCenter, const double & ballRadius, const std::vector < std::vector < double> > &elementCoordinates, std::vector < std::vector < double> > &newCoordinates) {

  //theyIntersect = true : element and ball intersect
  //theyIntersect = false : element and ball are disjoint

  //elementCoordinates are the coordinates of the vertices of the element

  theyIntersect = false; //by default we assume the two sets are disjoint

  unsigned dim = 2;
  unsigned nDofs = elementCoordinates[0].size();

  std::vector< std::vector < double > > ballVerticesCoordinates(dim);
  newCoordinates.resize(dim);


  for (unsigned n = 0; n < dim; n++) {
    newCoordinates[n].resize(nDofs);
    ballVerticesCoordinates[n].resize(4);

    for (unsigned i = 0; i < nDofs; i++) {
      newCoordinates[n][i] = elementCoordinates[n][i]; //this is just an initalization, it will be overwritten
    }
  }

  double xMinElem = elementCoordinates[0][0];
  double yMinElem = elementCoordinates[1][0];
  double xMaxElem = elementCoordinates[0][2];
  double yMaxElem = elementCoordinates[1][2];


  for (unsigned i = 0; i < 4; i++) {
    if (elementCoordinates[0][i] < xMinElem) xMinElem = elementCoordinates[0][i];

    if (elementCoordinates[0][i] > xMaxElem) xMaxElem = elementCoordinates[0][i];

    if (elementCoordinates[1][i] < yMinElem) yMinElem = elementCoordinates[1][i];

    if (elementCoordinates[1][i] > yMaxElem) yMaxElem = elementCoordinates[1][i];
  }

  //bottom left corner of ball (south west)
  ballVerticesCoordinates[0][0] =  ballCenter[0] - ballRadius;
  ballVerticesCoordinates[1][0] =  ballCenter[1] - ballRadius;

  //top right corner of ball (north east)
  ballVerticesCoordinates[0][2] = ballCenter[0] + ballRadius;
  ballVerticesCoordinates[1][2] = ballCenter[1] + ballRadius;

  newCoordinates[0][0] = (ballVerticesCoordinates[0][0] >= xMinElem) ? ballVerticesCoordinates[0][0] : xMinElem;
  newCoordinates[1][0] = (ballVerticesCoordinates[1][0] >= yMinElem) ? ballVerticesCoordinates[1][0] : yMinElem;

  newCoordinates[0][2] = (ballVerticesCoordinates[0][2] >= xMaxElem) ? xMaxElem : ballVerticesCoordinates[0][2];
  newCoordinates[1][2] = (ballVerticesCoordinates[1][2] >= yMaxElem) ? yMaxElem : ballVerticesCoordinates[1][2];

  if (newCoordinates[0][0] < newCoordinates[0][2] && newCoordinates[1][0] < newCoordinates[1][2]) {   //ball and rectangle intersect

    theyIntersect = true;

    newCoordinates[0][1] = newCoordinates[0][2];
    newCoordinates[1][1] = newCoordinates[1][0];

    newCoordinates[0][3] = newCoordinates[0][0];
    newCoordinates[1][3] = newCoordinates[1][2];

    if (nDofs > 4) {   //TODO the quadratic case has not yet been debugged

      newCoordinates[0][4] = 0.5 * (newCoordinates[0][0] + newCoordinates[0][1]);
      newCoordinates[1][4] = newCoordinates[1][0];

      newCoordinates[0][5] = newCoordinates[0][1];
      newCoordinates[1][5] = 0.5 * (newCoordinates[1][1] + newCoordinates[1][2]);

      newCoordinates[0][6] = newCoordinates[0][4];
      newCoordinates[1][6] = newCoordinates[1][2];

      newCoordinates[0][7] = newCoordinates[0][0];
      newCoordinates[1][7] = newCoordinates[1][5];

      if (nDofs > 8) {

        newCoordinates[0][8] = newCoordinates[0][4];
        newCoordinates[1][8] = newCoordinates[1][5];

      }

    }

  }

}



unsigned ReorderElement(std::vector < int > &dofs, std::vector < double > & sol, std::vector < std::vector < double > > & x) {

  unsigned type = 0;

  if (fabs(x[0][0] - x[0][1]) > 1.e-10) {
    if (x[0][0] - x[0][1] > 0) {
      type = 2;
    }
  }

  else {
    type = 1;

    if (x[1][0] - x[1][1] > 0) {
      type = 3;
    }
  }

  if (type != 0) {
    std::vector < int > dofsCopy = dofs;
    std::vector < double > solCopy = sol;
    std::vector < std::vector < double > > xCopy = x;

    for (unsigned i = 0; i < dofs.size(); i++) {
      dofs[i] = dofsCopy[mySwap[type][i]];
      sol[i] = solCopy[mySwap[type][i]];

      for (unsigned k = 0; k < x.size(); k++) {
        x[k][i] = xCopy[k][mySwap[type][i]];
      }
    }
  }

  return type;
}


void RectangleAndBallRelation2(bool & theyIntersect, const std::vector<double> &ballCenter, const double & ballRadius, const std::vector < std::vector < double> > &elementCoordinates, std::vector < std::vector < double> > &newCoordinates) {

  theyIntersect = false; //by default we assume the two sets are disjoint

  unsigned dim = 2;
  unsigned nDofs = elementCoordinates[0].size();

  newCoordinates.resize(dim);

  for (unsigned i = 0; i < dim; i++) {
    newCoordinates[i].resize(nDofs);
  }

  double xMin = elementCoordinates[0][0];
  double xMax = elementCoordinates[0][2];
  double yMin = elementCoordinates[1][0];
  double yMax = elementCoordinates[1][2];

  if (xMin > xMax || yMin > yMax) {
    std::cout << "error" << std::endl;

    for (unsigned i = 0; i < nDofs; i++) {
      std::cout <<  elementCoordinates[0][i] << " " << elementCoordinates[1][i] << std::endl;
    }

    exit(0);
  }


  double xMinBall = ballCenter[0] - ballRadius;
  double xMaxBall = ballCenter[0] + ballRadius;
  double yMinBall = ballCenter[1] - ballRadius;
  double yMaxBall = ballCenter[1] + ballRadius;


  xMin = (xMin > xMinBall) ? xMin : xMinBall;
  xMax = (xMax < xMaxBall) ? xMax : xMaxBall;
  yMin = (yMin > yMinBall) ? yMin : yMinBall;
  yMax = (yMax < yMaxBall) ? yMax : yMaxBall;

  if (xMin < xMax && yMin < yMax) {   //ball and rectangle intersect

    theyIntersect = true;

    //std::cout<< xMin <<" "<<xMax<<" "<<yMin<<" "<<yMax<<std::endl;

    newCoordinates[0][0] = xMin;
    newCoordinates[0][1] = xMax;
    newCoordinates[0][2] = xMax;
    newCoordinates[0][3] = xMin;

    newCoordinates[1][0] = yMin;
    newCoordinates[1][1] = yMin;
    newCoordinates[1][2] = yMax;
    newCoordinates[1][3] = yMax;

    if (nDofs > 4) {   //TODO the quadratic case has not yet been debugged

      double xMid = 0.5 * (xMin + xMax);
      double yMid = 0.5 * (yMin + yMax);

      newCoordinates[0][4] = xMid;
      newCoordinates[0][5] = xMax;
      newCoordinates[0][6] = xMid;
      newCoordinates[0][7] = xMin;

      newCoordinates[1][4] = yMin;
      newCoordinates[1][5] = yMid;
      newCoordinates[1][6] = yMax;
      newCoordinates[1][7] = yMid;

      if (nDofs > 8) {

        newCoordinates[0][8] = xMid;
        newCoordinates[1][8] = yMid;

      }

    }

  }



}















