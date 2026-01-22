/** \file Ex13.cpp
 *  \brief This example shows how to set and solve the weak form
 *   of the time dependent Stress-Strain Equation, for a beam with pressure
 *
 *  \nabla \cdot \sigma + \mass*acceleration = F
 *  in a Beam domain (in 2D and 3D) clamped on one end
 *
 *  \author Eugenio Aulisa
 */


#include "FemusInit.hpp"
#include "MultiLevelSolution.hpp"
#include "MultiLevelProblem.hpp"
#include "NumericVector.hpp"
#include "VTKWriter.hpp"
#include "GMVWriter.hpp"
#include "LinearImplicitSystem.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "TransientSystem.hpp"
#include "adept.h"

#include <cstdint>
#include <limits>
#include <cmath>

// Pair: minimize dist2, break ties by smaller gid
struct MinPair {
  double dist2;
  std::uint64_t gid;
};

static void mpi_minpair_op(void* inVec, void* inOutVec, int* len, MPI_Datatype* /*dtype*/) {
  auto* in    = static_cast<MinPair*>(inVec);
  auto* inout = static_cast<MinPair*>(inOutVec);
  for (int i = 0; i < *len; ++i) {
    const bool take_in =
    (in[i].dist2 < inout[i].dist2) ||
    (in[i].dist2 == inout[i].dist2 && in[i].gid < inout[i].gid);
    if (take_in) inout[i] = in[i];
  }
}

const unsigned DIM = 2;

double dt = 0.025;
const unsigned n_timesteps = 400;
unsigned cascadeIterations = 2;

unsigned jTMP = 0;

bool withDisturbance = true;

// static double PStar = 0.0;
std::vector<unsigned> g_controlNodeDofs;
unsigned gdof;

struct Indices {
    std::vector<unsigned> P, PStar, CStar;
    std::vector<unsigned> Z, ZOld, E;
    unsigned Zi, Xi, Yi, Ei, Ztot, R, C, S, S1;
    unsigned d; // only if withDisturbance
  };

static Indices g_idx;


std::vector<double> x1_vec;
std::vector<double> y1_vec;
std::vector<double> u1;
std::vector<std::vector<double>> w;
std::vector<std::vector<double>> wOld;

double alpha = 1.0e-7;
double beta = 0.125;
double h = 1.;
double a = -5.;
double b = 5.;

double mu = 1.;

static double g_bc_time = 0.0;

using namespace std;
using namespace femus;


struct WNodeIDs {
  unsigned W0;  // node for W-equation #1
  unsigned X1;  // node for W-equation #2
  unsigned Y1;  // node for W-equation #3
};

struct ODEParams {
  double dt;
  double alpha;
  double beta;
  double a;
  double b;
  double h;
};

struct ODEStageOut {
  std::vector<double> x1;   // size nCtrl
  std::vector<double> y1;   // size nCtrl
  std::vector<double> u1;   // size nCtrl
  std::vector<double> w;    // size nCtrl (updated w for this stage)
};

static inline void ensure_size(std::vector<double>& v, std::size_t n) {
  if (v.size() != n) v.assign(n, 0.0);
}

ODEStageOut computeODESystemStage( const Solution& sol, unsigned idxZi, unsigned idxXi, unsigned idxEi, const std::vector<unsigned>& idxP,
  const std::vector<unsigned>& idxPStar, const std::vector<unsigned>& idxCStar, const std::vector<double>& wOldStage, const ODEParams& par, MPI_Comm comm);

static void computeODESystem( const Solution& sol, const Indices& idx, const std::vector<unsigned>& controlNodeDofs, const ODEParams& par,
  unsigned jTMP, std::vector<std::vector<double>>& w, const std::vector<std::vector<double>>& wOld, std::vector<double>& x1_vec,
  std::vector<double>& y1_vec, std::vector<double>& u1);


static unsigned MapMeshDofToSystemRowP(const Mesh* msh, LinearEquationSolver* pdeSys, unsigned pIndex, unsigned pPdeIndex, unsigned pType, unsigned meshDofP);


double SetVariableTimeStep(const double time) {
  return dt;
}


struct RegionBox {
  double xMin, xMax;
  double yMin, yMax;
};


void SetRegions(Solution *sol, /*const RegionBox& boxB,*/ const RegionBox& boxC);

void SetPrescribedFields(Solution* sol, const double& time, const std::string& R);

bool SetBoundaryCondition(const std::vector < double >& x, const char SolName[], double& value, const int facename, const double time) {
  bool dirichlet = true;
  value = 0.;

  if(withDisturbance){
    if(!strcmp(SolName, "s")) {  // where s is the name of the variable
      if(4 == facename) { // 0 is the face ( it could be 2)
        value = - M_PI * cos(g_bc_time * M_PI); // d_t(x,t)
      }
      else { // all other faces
        value = 0.;
      }
    }
    else if(!strcmp(SolName, "s1")) {  // where s is the name of the variable
      if(4 == facename) { // 0 is the face ( it could be 2)
        value = sin(x[1] * M_PI) + sin(g_bc_time * M_PI); // d(x,t)
      }
      else { // all other faces
        value = 0.;
      }
    }
  }

  return dirichlet;
}


void AssembleResAD(MultiLevelProblem& ml_prob);
void AssembleResADReduced(MultiLevelProblem& ml_prob);
void AssembleResP(MultiLevelProblem& ml_prob);

void AssembleLaplacian_s (MultiLevelProblem& ml_prob);
void AssembleLaplacian_s1(MultiLevelProblem& ml_prob);

static void AssembleLaplacianCore(MultiLevelProblem& ml_prob, const char* system_name, const char* var_name);


std::vector<double> PrecomputePstarIntegrals(Solution* sol);


std::vector<double> ComputeL2NormCascadeOverC(Solution* sol, unsigned cascadeIterations);

std::vector<unsigned> GetControlNodeIndices(const Mesh* msh, const std::vector<std::vector<double>>& points);

// std::vector<unsigned> GetGlobalNodeIDsForW(const Mesh* msh, unsigned elemID, const std::vector<unsigned>& localNodeIDs);
WNodeIDs GetGlobalNodeIDsForW(const Mesh* msh, unsigned elemID);

int main(int argc, char** args) {
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  MultiLevelMesh mlMsh;

  // unsigned nx = 10;
  unsigned nx = 8;
  unsigned ny = 8;
  unsigned nz = 1;

  double length = 1;
  double lengthx = M_PI;

  std::vector<std::vector<double>> controlPoints = {
    {M_PI / 1.5, 0.5},{M_PI / 2.5,0.5}
  };

  if (DIM == 2) {
    mlMsh.GenerateCoarseBoxMesh(nx, ny, 0, 0., lengthx, 0., length, 0., 0., QUAD9, "seventh");
  }
  else if (DIM == 3) {
    nz = ny;
    mlMsh.GenerateCoarseBoxMesh(nx, ny, nz, 0., lengthx, 0., length, 0., length, HEX27, "seventh");
  }

  unsigned numberOfUniformLevels = 1;
  unsigned numberOfSelectiveLevels = 0;
  mlMsh.RefineMesh(numberOfUniformLevels, numberOfUniformLevels + numberOfSelectiveLevels, NULL);
  mlMsh.EraseCoarseLevels(numberOfUniformLevels - 1);
  mlMsh.PrintInfo();

  MultiLevelSolution mlSol(&mlMsh);

  const unsigned level = mlMsh.GetNumberOfLevels() - 1;
  Solution* sol = mlSol.GetSolutionLevel(level);
  Mesh* msh = sol->GetMesh();

  g_controlNodeDofs = GetControlNodeIndices(msh, controlPoints);

  w.assign(cascadeIterations,std::vector<double>(g_controlNodeDofs.size(),0.));
  wOld.assign(cascadeIterations,std::vector<double>(g_controlNodeDofs.size(),0.));

  for (unsigned j = 0; j < g_controlNodeDofs.size(); j++) {
    std::string Pj = "P" + std::to_string(j);
    std::string PStarj = "PStar" + std::to_string(j);
    std::string CStarCPStarj = "CStarCPStar" + std::to_string(j);

    mlSol.AddSolution(Pj.c_str(), LAGRANGE, SECOND, false);
    mlSol.AddSolution(PStarj.c_str(), LAGRANGE, SECOND, false);
    mlSol.AddSolution(CStarCPStarj.c_str(), LAGRANGE, SECOND, false);
  }

  mlSol.AddSolution("Zi", LAGRANGE, SECOND, 2);
  mlSol.AddSolution("Xi", LAGRANGE, SECOND);
  mlSol.AddSolution("Yi", LAGRANGE, SECOND);
  mlSol.AddSolution("Ei", LAGRANGE, SECOND, false);
  mlSol.AddSolution("Z", LAGRANGE, SECOND, false);

  mlSol.AddSolution("P", LAGRANGE, SECOND, false);

  mlSol.AddSolution("PStar",       LAGRANGE, SECOND, false);
  mlSol.AddSolution("CStarCPStar", LAGRANGE, SECOND, false);

  for (unsigned j = 0; j < cascadeIterations; j++) {
    std::string Zj = "Z" + std::to_string(j);
    std::string ZjOld = "Z" + std::to_string(j) + "Old";
    std::string Ej = "E" + std::to_string(j);
    mlSol.AddSolution(Zj.c_str(), LAGRANGE, SECOND, false);
    mlSol.AddSolution(ZjOld.c_str(), LAGRANGE, SECOND, false);
    mlSol.AddSolution(Ej.c_str(), LAGRANGE, SECOND, false);
  }

  mlSol.AddSolution("R", LAGRANGE, SECOND, false);
  // mlSol.AddSolution("B", DISCONTINUOUS_POLYNOMIAL, ZERO, false);
  mlSol.AddSolution("C", DISCONTINUOUS_POLYNOMIAL, ZERO, false);
  if(withDisturbance) {
    mlSol.AddSolution("d", LAGRANGE, SECOND, false);
    mlSol.AddSolution("s",  LAGRANGE, SECOND, false);
    mlSol.AddSolution("s1", LAGRANGE, SECOND, false);
  }
  mlSol.Initialize("All");
  mlSol.AttachSetBoundaryConditionFunction(SetBoundaryCondition);
  mlSol.GenerateBdc("All");

  const unsigned nCtrl = g_controlNodeDofs.size();
  g_idx.P.resize(nCtrl);
  g_idx.PStar.resize(nCtrl);
  g_idx.CStar.resize(nCtrl);

  for (unsigned p = 0; p < nCtrl; ++p) {
    g_idx.P[p]     = mlSol.GetIndex(("P" + std::to_string(p)).c_str());
    g_idx.PStar[p] = mlSol.GetIndex(("PStar" + std::to_string(p)).c_str());
    g_idx.CStar[p] = mlSol.GetIndex(("CStarCPStar" + std::to_string(p)).c_str());
  }

  g_idx.Z.resize(cascadeIterations);
  g_idx.ZOld.resize(cascadeIterations);
  g_idx.E.resize(cascadeIterations);

  for (unsigned j = 0; j < cascadeIterations; ++j) {
    g_idx.Z[j]    = mlSol.GetIndex(("Z" + std::to_string(j)).c_str());
    g_idx.ZOld[j] = mlSol.GetIndex(("Z" + std::to_string(j) + "Old").c_str());
    g_idx.E[j]    = mlSol.GetIndex(("E" + std::to_string(j)).c_str());
  }

  // singletons
  g_idx.Zi   = mlSol.GetIndex("Zi");
  g_idx.Xi   = mlSol.GetIndex("Xi");
  g_idx.Yi   = mlSol.GetIndex("Yi");
  g_idx.Ei   = mlSol.GetIndex("Ei");
  g_idx.Ztot = mlSol.GetIndex("Z");
  g_idx.R    = mlSol.GetIndex("R");
  g_idx.C    = mlSol.GetIndex("C");

  if (withDisturbance) {
    g_idx.S = mlSol.GetIndex("s");
    g_idx.S1 = mlSol.GetIndex("s1");
  }

  auto bad = [](unsigned v) { return v == static_cast<unsigned>(-1); };

  // controls
  for (unsigned p = 0; p < nCtrl; ++p) {
    if (bad(g_idx.P[p]) || bad(g_idx.PStar[p]) || bad(g_idx.CStar[p])) {
      std::cerr << "Bad control indices at p=" << p << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // cascade
  for (unsigned j = 0; j < cascadeIterations; ++j) {
    if (bad(g_idx.Z[j]) || bad(g_idx.ZOld[j]) || bad(g_idx.E[j])) {
      std::cerr << "Bad cascade indices at j=" << j << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  // singletons
  if (bad(g_idx.Zi) || bad(g_idx.Xi) || bad(g_idx.Yi) || bad(g_idx.Ei) ||
      bad(g_idx.Ztot) || bad(g_idx.R) || bad(g_idx.C)) {
    std::cerr << "Bad singleton indices\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }


  // ---------- ODE parameters (constant for the run) ----------
  ODEParams odePar;
  odePar.dt    = dt;
  odePar.alpha = alpha;
  odePar.beta  = beta;
  odePar.a     = a;
  odePar.b     = b;
  odePar.h     = h;

  // ---------- cache indices / references once ----------
  const unsigned idxZi   = g_idx.Zi;
  const unsigned idxXi   = g_idx.Xi;
  const unsigned idxYi   = g_idx.Yi;
  const unsigned idxEi   = g_idx.Ei;
  const unsigned idxZtot = g_idx.Ztot;
  const unsigned idxR    = g_idx.R;
  const unsigned idxC    = g_idx.C;

  const unsigned idxS  = withDisturbance ? g_idx.S  : static_cast<unsigned>(-1);
  const unsigned idxS1 = withDisturbance ? g_idx.S1 : static_cast<unsigned>(-1);

  const std::vector<unsigned>& idxP     = g_idx.P;
  const std::vector<unsigned>& idxPStar = g_idx.PStar;
  const std::vector<unsigned>& idxCStar = g_idx.CStar;

  const unsigned nCtrl_cached = static_cast<unsigned>(idxP.size());


  MultiLevelProblem mlProb(&mlSol);

  // New stationary system for P
  LinearImplicitSystem& systemP = mlProb.add_system<LinearImplicitSystem>("LP");
  systemP.AddSolutionToSystemPDE("P");
  systemP.SetAssembleFunction(AssembleResP);
  systemP.init();

  TransientLinearImplicitSystem& lap_s = mlProb.add_system<TransientLinearImplicitSystem>("Lap_s");
  TransientLinearImplicitSystem& lap_s1 = mlProb.add_system<TransientLinearImplicitSystem>("Lap_s1");
  TransientNonlinearImplicitSystem& system = mlProb.add_system<TransientNonlinearImplicitSystem>("LSAT");

  if(withDisturbance){
    lap_s.AddSolutionToSystemPDE("s");
    lap_s.SetAssembleFunction(AssembleLaplacian_s);
    lap_s.init();

    lap_s1.AddSolutionToSystemPDE("s1");
    lap_s1.SetAssembleFunction(AssembleLaplacian_s1);
    lap_s1.init();

    system.AddSolutionToSystemPDE("Zi");
    system.AddSolutionToSystemPDE("Xi");
    system.AddSolutionToSystemPDE("Yi");
    system.SetAssembleFunction(AssembleResAD);
    system.AttachGetTimeIntervalFunction(SetVariableTimeStep);
    system.init();
    system.SetOuterSolver(PREONLY);
  }

  TransientNonlinearImplicitSystem& systemR = mlProb.add_system<TransientNonlinearImplicitSystem>("LSAT_reduced");
  systemR.AddSolutionToSystemPDE("Zi");
  systemR.AddSolutionToSystemPDE("Xi");
  systemR.SetAssembleFunction(AssembleResADReduced);
  systemR.AttachGetTimeIntervalFunction(SetVariableTimeStep);
  systemR.init();
  systemR.SetOuterSolver(PREONLY);


  // RegionBox boxB{M_PI/3., 2*M_PI/3., 0., 1};
  RegionBox boxC{(M_PI / 4.) - 0.001, (3.*M_PI / 4.) + 0.001, 0.249, 0.751};
  SetRegions(sol, boxC);
  // else SetRegions(sol, boxB, boxC);

  sol->_Sol[mlSol.GetIndex("P")]->zero();

  std::vector<std::string> variablesToBePrinted;
  variablesToBePrinted.push_back("All");

  VTKWriter vtkIO(&mlSol);
  vtkIO.SetDebugOutput(false);
  vtkIO.Write(DEFAULT_OUTPUTDIR, "biquadratic", variablesToBePrinted, 0);

  for(unsigned j = 0; j < g_controlNodeDofs.size(); j++){
    std::string Pj = "P" + std::to_string(j);

    gdof = g_controlNodeDofs[j];

    systemP.MGsolve();

    *(sol->_Sol[mlSol.GetIndex(Pj.c_str())]) = *(sol->_Sol[(mlSol.GetIndex("P"))]);
  }

  std::vector<double> IntegralP = PrecomputePstarIntegrals(sol);

  // ****here we solve the system for P****

  for (unsigned j = 0; j < cascadeIterations; ++j) {
    sol->_Sol[g_idx.ZOld[j]]->zero();
  }

  int world_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  std::ofstream errorFile;
  if (world_rank == 0) {
    errorFile.open("error.dat", std::ios::out | std::ios::trunc);
    if (!errorFile.is_open()) {
      std::cerr << "ERROR: could not open error.dat for writing\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    errorFile.setf(std::ios::scientific);
    errorFile << "# time";
    for (unsigned j = 0; j < cascadeIterations; ++j) errorFile << "   E" << j;
    errorFile << "\n";
    errorFile << std::setprecision(8);
  }

  std::ofstream wFile;
  if (world_rank == 0) {
    wFile.open("w.dat", std::ios::out | std::ios::trunc);
    if (!wFile.is_open()) {
      std::cerr << "ERROR: could not open error.dat for writing\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    wFile.setf(std::ios::scientific);
    wFile << "# time";
    for (unsigned j = 0; j < cascadeIterations; ++j) wFile << "   w" << j << "   x1" << j << "   y1" << j;
    wFile << "\n";
    wFile << std::setprecision(8);
  }

  // BEGIN Time loop
  for (unsigned t = 1; t <= n_timesteps; t++) {
    const double time = t * dt;

    g_bc_time = time;

    if(withDisturbance){
      mlSol.GenerateBdc("s");
      mlSol.GenerateBdc("s1");
    }


    SetPrescribedFields(sol, t * dt, "R");

    sol->_Sol[g_idx.Ztot]->zero();
    *(sol->_Sol[g_idx.Ei]) = *(sol->_Sol[g_idx.R]);

    if(world_rank == 0) wFile << std::setw(14) << t * dt;

    if(withDisturbance){
      lap_s.MGsolve();
      lap_s1.MGsolve();
    }

    for (unsigned j = 0; j < cascadeIterations; j++) {

      jTMP = j;

      //For withDisturbance, and j = 1,2,.., we set the disturbance to be zero
      if (withDisturbance && j > 0) sol->_Sol[mlSol.GetIndex("d")]->zero();

      *(sol->_Sol[g_idx.Zi]) = *(sol->_Sol[g_idx.ZOld[j]]);
      *(sol->_SolOld[g_idx.Zi]) = *(sol->_Sol[g_idx.ZOld[j]]);

      computeODESystem(*sol, g_idx, g_controlNodeDofs, odePar, jTMP, w, wOld, x1_vec, y1_vec, u1);

      if(withDisturbance && j ==0) system.MGsolve();
      else systemR.MGsolve();

      //std::cout<<"j = " << j << " w = " << w[j] <<std::endl;

      wOld[j] = w[j];

      *(sol->_Sol[g_idx.Ztot]) += *(sol->_Sol[g_idx.Zi]);

      for (unsigned p=0; p<nCtrl; ++p) {
        sol->_Sol[g_idx.Ztot]->add(w[j][p], *(sol->_Sol[g_idx.P[p]]));
      }
      if(withDisturbance && j == 0) sol->_Sol[g_idx.Ztot]->add(+1.0, *(sol->_Sol[g_idx.S1]));


      *(sol->_Sol[g_idx.Z[j]]) = *(sol->_Sol[g_idx.Zi]);
      *(sol->_Sol[g_idx.E[j]]) = *(sol->_Sol[g_idx.Ei]);
      *(sol->_Sol[g_idx.E[j]]) -= *(sol->_Sol[g_idx.Zi]);

      for (unsigned p = 0; p < g_controlNodeDofs.size(); p++) {
        sol->_Sol[g_idx.E[j]]->add(-w[j][p], *(sol->_Sol[g_idx.P[p]]));
      }
      if(withDisturbance && j == 0) sol->_Sol[g_idx.E[j]]->add(-1.0, *(sol->_Sol[g_idx.S1]));

      *(sol->_Sol[g_idx.Ei]) = *(sol->_Sol[g_idx.E[j]]);

      if (world_rank == 0) {
        for (unsigned p = 0; p < g_controlNodeDofs.size(); p++) {
        wFile << "  " << std::setw(14) << w[j][p] << " " << std::setw(14) << x1_vec[p] << " " << std::setw(14) << y1_vec[p];
        }
      }
    }

    if (world_rank == 0) {
      wFile << "\n";
      wFile.flush();
    }

    vtkIO.Write(DEFAULT_OUTPUTDIR, "biquadratic", variablesToBePrinted, t);

    std::vector<double> L2_E_Cascade = ComputeL2NormCascadeOverC(sol, cascadeIterations);
    if (world_rank == 0) {
      errorFile << std::setw(14) << t * dt;
      for (double val : L2_E_Cascade) errorFile << " " << std::setw(14) << val;
      errorFile << "\n";
      errorFile.flush();
    }

    for (unsigned j = 0; j < cascadeIterations; ++j) {
      *(sol->_Sol[g_idx.ZOld[j]]) = *(sol->_Sol[g_idx.Z[j]]);
    }
  }
  //END time loop

  MPI_Barrier(MPI_COMM_WORLD);
  if (world_rank == 0) errorFile.close();

  return 0;
}



double flc4hs(double const & x, double const & eps) {

  double r = x / eps;
  if (r < -1) {
    return 0.;
  }
  else if (r < 1.) {
    double r2 = r * r;
    double r3 = r * r2;
    double r5 = r3 * r2;
    double r7 = r5 * r2;
    double r9 = r7 * r2;
    return (128. + 315. * r - 420. * r3 + 378. * r5 - 180. * r7 + 35. * r9) / 256.;
  }
  else {
    return 1.;
  }
}

/*double GetTargetSolution(const std::vector<double> &xv, const double &time) {
  // return cos(xv[0] * xv[1]) * flc4hs(time - .5, 0.5) ;
  return xv[0] * (M_PI - xv[0]) * xv[1] * (1. - xv[1]) * sin(time) * flc4hs(time - .5, 0.5) ;
}*/

double GetTargetSolution(const std::vector<double> &xv, const double &time) {
  return sin(M_PI * xv[1]) * sin(xv[0] - time) * flc4hs(time - 2., 2.) ;
  // return /* flc4hs(time - 2., 2.)**/1.;
}


/*double GetDisturbanceSolution(const std::vector<double>& xv, const double& time) {
  return xv[1] * (1. - xv[1]) * sin(2. * time) * flc4hs(time - .5, 0.5);
}*/

double GetDisturbanceSolution(const std::vector<double>& xv, const double& time) {
  return (xv[0] - 2 * M_PI / 3.0) * (M_PI - xv[0]) * xv[1] * (1 - xv[1]) * sin(2 * time) * flc4hs(time - 0.5, 0.5);
}

void SetPrescribedFields(Solution* sol, const double& time, const std::string& R) {

  Mesh* msh = sol->GetMesh();
  const unsigned dim = msh->GetDimension();
  unsigned iproc = msh->processor_id();

  unsigned rIndex = sol->GetIndex(R.c_str());
  unsigned rType = sol->GetSolutionType(rIndex);

  bool hasDisturbance = false;
  unsigned dIndex = 0;

  std::vector<double> xv(dim);
  unsigned xType = 2;  // coordinates always quadratic

  for (unsigned iel = msh->_elementOffset[iproc];
       iel < msh->_elementOffset[iproc + 1]; iel++) {

    unsigned nDofsR = msh->GetElementDofNumber(iel, rType);

    for (unsigned i = 0; i < nDofsR; i++) {
      unsigned xDof = msh->GetSolutionDof(i, iel, xType);
      for (unsigned k = 0; k < dim; k++)
        xv[k] = (*msh->_topology->_Sol[k])(xDof);

      unsigned uDofR = msh->GetSolutionDof(i, iel, rType);
      double rVal = GetTargetSolution(xv, time);
      sol->_Sol[rIndex]->set(uDofR, rVal);
    }
  }

  sol->_Sol[rIndex]->close();
}

bool CheckIfInside(const std::vector<double>& xv, const RegionBox& box) {
  return (xv[0] > box.xMin && xv[0] < box.xMax &&
          xv[1] > box.yMin && xv[1] < box.yMax);
}

// double CheckIfInsideB(const std::vector<double> &xv) {
//   return (xv[0] > 0.5 && xv[0] < 1. && xv[1] > 0.25 && xv[1] < 0.75);
// }
//
// double CheckIfInsideC(const std::vector<double> &xv) {
//   return (xv[0] > 0.5 && xv[0] < 1. && xv[1] > 0.25 && xv[1] < 0.75);
// }

void SetRegions(Solution* sol,
                // const RegionBox& boxB,
                const RegionBox& boxC) {

  Mesh* msh = sol->GetMesh();
  const unsigned dim = msh->GetDimension();
  unsigned iproc = msh->processor_id();

  // unsigned IndexB = sol->GetIndex("B");
  unsigned IndexC = sol->GetIndex("C");

  std::vector<double> xv(dim);
  unsigned xType = 2;

  for (unsigned iel = msh->_elementOffset[iproc];
       iel < msh->_elementOffset[iproc + 1]; iel++) {

    unsigned nDofs = msh->GetElementDofNumber(iel, 1);
    // bool elementInB = true;
    bool elementInC = true;

    for (unsigned i = 0; i < nDofs; ++i) {
      unsigned xDof = msh->GetSolutionDof(i, iel, xType);
      for (unsigned k = 0; k < dim; ++k)
        xv[k] = (*msh->_topology->_Sol[k])(xDof);

      // elementInB  = elementInB  && CheckIfInside(xv, boxB);
      elementInC  = elementInC  && CheckIfInside(xv, boxC);
    }

    // sol->_Sol[IndexB]->set(iel, elementInB ? 1.0 : 0.0);
    sol->_Sol[IndexC]->set(iel, elementInC ? 1.0 : 0.0);
  }

  // sol->_Sol[IndexB]->close();
  sol->_Sol[IndexC]->close();

}





//Assemble Residual using A to update D amd V
void AssembleResADReduced(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled
  //  levelMax is the Maximum level of the MultiLevelProblem
  //  assembleMatrix is a flag that tells if only the residual or also the matrix should be assembled


  adept::Stack& s = FemusInit::_adeptStack;

  //  extract pointers to the several objects that we are going to use
  TransientNonlinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<TransientNonlinearImplicitSystem> ("LSAT_reduced");   // pointer to the linear implicit system named "Beam"
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*          msh          = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object

  MultiLevelSolution*  mlSol        = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*    sol        = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object


  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  SparseMatrix*    KK         = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*   RES          = pdeSys->_RES; // pointer to the global residual std::vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem

  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)

  double dt =  mlPdeSys->GetIntervalTime();

  //solution variable
  // Note: for P* use PStar = IntegralP;
  // PStarNodes = std::move(WeightsP);
  unsigned solIndexZ = mlSol->GetIndex("Zi");
  unsigned solIndexX = mlSol->GetIndex("Xi");

  unsigned solIndexE = mlSol->GetIndex("Ei");

  // unsigned solIndexB = mlSol->GetIndex("B");
  unsigned solIndexC = mlSol->GetIndex("C");


  unsigned solType = mlSol->GetSolutionType(solIndexZ);


  unsigned solPdeIndexZ = mlPdeSys->GetSolPdeIndex("Zi");
  unsigned solPdeIndexX = mlPdeSys->GetSolPdeIndex("Xi");

  std::vector < double > solZOld;    // local solution

  std::vector < double > solZdouble;    // local solution
  std::vector < double > solXdouble;

  std::vector < adept::adouble > solZ;    // local solution
  std::vector < adept::adouble > solX;

  std::vector < double > solE;
  std::vector<std::vector < double > > solP(g_controlNodeDofs.size());

  std::vector < std::vector < double > > coordX(dim);    // local coordinates
  unsigned coordXType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  std::vector <double> phi;  // local test function for velocity
  std::vector <double> gradPhi; // local test function first order partial derivatives
  double weight; // gauss point weight

  std::vector < unsigned > sysDof; // local to global pdeSys dofs
  std::vector < adept::adouble > aRes;
  std::vector < double > res; // local redidual std::vector
  std::vector < double > Jac;

  RES->zero(); // Set to zero all the entries of the Global Residual std::vector
  KK->zero(); // Set to zero all the entries of the Global Matrix

  if (jTMP >= w.size()) {
    std::cerr << "AssembleResADReduced: jTMP out of range\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const unsigned nCtrl = g_controlNodeDofs.size();
  if (w[jTMP].size() != nCtrl || y1_vec.size() != nCtrl || u1.size() != nCtrl) {
    std::cerr << "AssembleResADReduced: ODE arrays not ready or wrong size\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // element loop: each process loops only on the elements that owns
  for (unsigned iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

    short unsigned ielGeom = msh->GetElementType(iel);

    // double BBs = (*sol->_Sol[solIndexB])(iel);
    double CsC = (*sol->_Sol[solIndexC])(iel);

    unsigned nDofs = msh->GetElementDofNumber(iel, solType);    // number of solution element dofs

    unsigned nUnkn = 2;
    unsigned nDofsAll = nUnkn * nDofs;

    sysDof.resize(nDofsAll);
    aRes.assign(nDofsAll, 0.);

    solZOld.resize(nDofs);
    solZ.resize(nDofs);
    solX.resize(nDofs);
    solE.resize(nDofs);

    for(unsigned j = 0; j < g_controlNodeDofs.size(); j++) solP[j].resize(nDofs);

    for (unsigned  k = 0; k < dim; k++) {
      coordX[k].resize(nDofs);
    }

    // local storage of global mapping and solution
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned iDof = msh->GetSolutionDof(i, iel, solType);

      solZOld[i] = (*sol->_SolOld[solIndexZ])(iDof);
      solZ[i] = (*sol->_Sol[solIndexZ])(iDof);
      solX[i] = (*sol->_Sol[solIndexX])(iDof);

      for(unsigned j = 0; j < g_controlNodeDofs.size(); j++){
        solP[j][i] = (*sol->_Sol[g_idx.P[j]])(iDof);
      }

      solE[i] = (*sol->_Sol[solIndexE])(iDof);

      for (unsigned k = 0; k < nUnkn; k++) {
        unsigned solIndex = (k == 0) ? solIndexZ : solIndexX;
        unsigned solPdeIndex = (k == 0) ? solPdeIndexZ : solPdeIndexX;
        sysDof[k * nDofs + i] = pdeSys->GetSystemDof(solIndex, solPdeIndex, i, iel);
      }
    }

    // local storage of coordinates
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned coordXDof  = msh->GetSolutionDof(i, iel, coordXType);    // local to global mapping between coordinates node and coordinate dof
      for (unsigned k = 0; k < dim; k++) {
        coordX[k][i] = (*msh->_topology->_Sol[k])(coordXDof);      // global extraction and local storage for the element coordinates
      }
    }

    s.new_recording();

    // *** Gauss point loop ***
    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solType]->GetGaussPointNumber(); ig++) {
      // *** get gauss point weight, test function and test function partial derivatives ***
      msh->_finiteElement[ielGeom][solType]->Jacobian(coordX, ig, weight, phi, gradPhi);

      double ZOldg = 0.;
      adept::adouble Zg = 0.;
      adept::adouble Xg = 0.;
      std::vector<double> Pg (g_controlNodeDofs.size(), 0.);

      double r = 0.;
      double d = 0.;

      std::vector < adept::adouble > gradZg(dim, 0.);
      std::vector < adept::adouble > gradXg(dim, 0.);

      for (unsigned i = 0; i < nDofs; i++) {

        ZOldg += solZOld[i] * phi[i];
        Zg += solZ[i] * phi[i];
        Xg += solX[i] * phi[i];
        for(unsigned j = 0; j < g_controlNodeDofs.size(); j++) Pg[j] += solP[j][i] * phi[i];

        r += solE[i] * phi[i];

        for (unsigned j = 0; j < dim; j++) {
          gradZg[j] += solZ[i] * gradPhi[i * dim + j];
          gradXg[j] += solX[i] * gradPhi[i * dim + j];
        }
      }


      // *** phiA_i loop ***
      for (unsigned i = 0; i < nDofs; i++) {
        unsigned coordXDof  = msh->GetSolutionDof(i, iel, coordXType);

        adept::adouble aResZ = (Zg - ZOldg) / dt * phi[i];
        adept::adouble aResX = - (CsC > 0.5) * (r - (1. - beta) * Zg) * phi[i];

        for(unsigned j = 0; j < g_controlNodeDofs.size(); j++){
          aResZ += h * a * Pg[j] * w[jTMP][j] * phi[i] + h * b * Pg[j] * u1[j] * phi[i];
          aResX += (CsC > 0.5) * ( (1. - beta) * h * Pg[j] * w[jTMP][j] - beta * b * h * Pg[j] * u1[j] / a) * phi[i];
        }


        for (unsigned d = 0; d < dim; d++) { // second index j in each equation
          aResZ +=  mu * gradPhi[i * dim + d] * gradZg[d]; // diffusion
          aResX +=  mu * gradPhi[i * dim + d] * gradXg[d]; // diffusion
        }


        aRes[0 * nDofs + i] += aResZ * weight;
        aRes[1 * nDofs + i] += aResX * weight;

      } // end phiA_i loop
    }

    //--------------------------------------------------------------------------------------------------------
    // Add the local Matrix/Vector into the global Matrix/Vector

    res.resize(nDofsAll);
    //copy the value of the adept::adoube mRes in double Res and store
    for (int i = 0; i < nDofsAll; i++) {
      res[i] = -aRes[i].value();
    }

    // define the dependent variables
    s.dependent(aRes.data(), nDofsAll);
    s.independent(solZ.data(), nDofs);
    s.independent(solX.data(), nDofs);

    Jac.assign(nDofsAll * nDofsAll, 0.);
    // get the jacobian matrix (ordered by column)
    s.jacobian(Jac.data(), true);

    RES->add_vector_blocked(res, sysDof);
    KK->add_matrix_blocked(Jac, sysDof, sysDof);

    s.clear_independents();
    s.clear_dependents();

  } //end element loop for each process

  RES->close();
  KK->close();
  //KK->draw();

  // double a;
  // std::cin>>a;

}



//Assemble Residual using A to update D amd V
void AssembleResAD(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled
  //  levelMax is the Maximum level of the MultiLevelProblem
  //  assembleMatrix is a flag that tells if only the residual or also the matrix should be assembled


  adept::Stack& s = FemusInit::_adeptStack;

  //  extract pointers to the several objects that we are going to use
  TransientNonlinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<TransientNonlinearImplicitSystem> ("LSAT");   // pointer to the linear implicit system named "Beam"
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*          msh          = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object

  MultiLevelSolution*  mlSol        = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*    sol        = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object


  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  SparseMatrix*    KK         = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*   RES          = pdeSys->_RES; // pointer to the global residual std::vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem

  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)

  double dt =  mlPdeSys->GetIntervalTime();

  //solution variable
  // Note: for P* use PStar = IntegralP;
  // PStarNodes = std::move(WeightsP);
  unsigned solIndexZ = mlSol->GetIndex("Zi");
  unsigned solIndexX = mlSol->GetIndex("Xi");
  unsigned solIndexY = mlSol->GetIndex("Yi");

  unsigned solIndexE = mlSol->GetIndex("Ei");

  // unsigned solIndexB = mlSol->GetIndex("B");
  unsigned solIndexC = mlSol->GetIndex("C");

  unsigned solIndexS = mlSol->GetIndex("s");
  unsigned solIndexS1 = mlSol->GetIndex("s1");

  unsigned solType = mlSol->GetSolutionType(solIndexZ);


  unsigned solPdeIndexZ = mlPdeSys->GetSolPdeIndex("Zi");
  unsigned solPdeIndexX = mlPdeSys->GetSolPdeIndex("Xi");
  unsigned solPdeIndexY = mlPdeSys->GetSolPdeIndex("Yi");

  std::vector < double > solZOld;    // local solution

  std::vector < double > solZdouble;    // local solution
  std::vector < double > solXdouble;
  std::vector < double > solYdouble;

  std::vector < adept::adouble > solZ;    // local solution
  std::vector < adept::adouble > solX;
  std::vector < adept::adouble > solY;

  // double PstarZ = 0;
  // double PstarX = 0;
  // double PstarY = 0;
  // double CstarCPstarZ = 0;
  // double CstarCPstarY = 0;
  // double CstarCPstarP = 0;
  // double CstarCPstarE = 0.;
  // x1 = 0;
  // double y1 = 0;
  //
  // w[jTMP] = 0.;

  std::vector < double > solE;
  std::vector < double > solS;
  std::vector < double > solS1;
  std::vector<std::vector < double > > solP(g_controlNodeDofs.size());

  std::vector < std::vector < double > > coordX(dim);    // local coordinates
  unsigned coordXType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  std::vector <double> phi;  // local test function for velocity
  std::vector <double> gradPhi; // local test function first order partial derivatives
  double weight; // gauss point weight

  std::vector < unsigned > sysDof; // local to global pdeSys dofs
  std::vector < adept::adouble > aRes;
  std::vector < double > res; // local redidual std::vector
  std::vector < double > Jac;

  RES->zero(); // Set to zero all the entries of the Global Residual std::vector
  KK->zero(); // Set to zero all the entries of the Global Matrix

  if (jTMP >= w.size()) {
    std::cerr << "AssembleResAD: jTMP out of range\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const unsigned nCtrl = g_controlNodeDofs.size();
  if (w[jTMP].size() != nCtrl || y1_vec.size() != nCtrl || u1.size() != nCtrl) {
    std::cerr << "AssembleResAD: ODE arrays not ready or wrong size\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // element loop: each process loops only on the elements that owns
  for (unsigned iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

    short unsigned ielGeom = msh->GetElementType(iel);

    // double BBs = (*sol->_Sol[solIndexB])(iel);
    double CsC = (*sol->_Sol[solIndexC])(iel);

    unsigned nDofs = msh->GetElementDofNumber(iel, solType);    // number of solution element dofs

    unsigned nUnkn = 3;
    unsigned nDofsAll = nUnkn * nDofs;

    sysDof.resize(nDofsAll);
    aRes.assign(nDofsAll, 0.);

    solZOld.resize(nDofs);
    solZ.resize(nDofs);
    solX.resize(nDofs);
    solY.resize(nDofs);
    solE.resize(nDofs);
    solS.resize(nDofs);
    solS1.resize(nDofs);

    for(unsigned j = 0; j < g_controlNodeDofs.size(); j++) solP[j].resize(nDofs);

    for (unsigned  k = 0; k < dim; k++) {
      coordX[k].resize(nDofs);
    }

    // local storage of global mapping and solution
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned iDof = msh->GetSolutionDof(i, iel, solType);

      solZOld[i] = (*sol->_SolOld[solIndexZ])(iDof);
      solZ[i] = (*sol->_Sol[solIndexZ])(iDof);
      solX[i] = (*sol->_Sol[solIndexX])(iDof);
      solY[i] = (*sol->_Sol[solIndexY])(iDof);

      for(unsigned j = 0; j < g_controlNodeDofs.size(); j++){
        solP[j][i] = (*sol->_Sol[g_idx.P[j]])(iDof);
      }

      solE[i] = (*sol->_Sol[solIndexE])(iDof);
      solS[i] = (*sol->_Sol[solIndexS])(iDof);
      solS1[i] = (*sol->_Sol[solIndexS1])(iDof);

      for (unsigned k = 0; k < nUnkn; k++) {
        unsigned solIndex = (k == 0) ? solIndexZ :
        (k == 1) ? solIndexX : solIndexY;
        unsigned solPdeIndex = (k == 0) ? solPdeIndexZ :
        (k == 1) ? solPdeIndexX : solPdeIndexY;
        sysDof[k * nDofs + i] = pdeSys->GetSystemDof(solIndex, solPdeIndex, i, iel);
      }
    }

    // local storage of coordinates
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned coordXDof  = msh->GetSolutionDof(i, iel, coordXType);    // local to global mapping between coordinates node and coordinate dof
      for (unsigned k = 0; k < dim; k++) {
        coordX[k][i] = (*msh->_topology->_Sol[k])(coordXDof);      // global extraction and local storage for the element coordinates
      }
    }


    s.new_recording();

    // *** Gauss point loop ***
    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solType]->GetGaussPointNumber(); ig++) {
      // *** get gauss point weight, test function and test function partial derivatives ***
      msh->_finiteElement[ielGeom][solType]->Jacobian(coordX, ig, weight, phi, gradPhi);

      double ZOldg = 0.;
      adept::adouble Zg = 0.;
      adept::adouble Xg = 0.;
      adept::adouble Yg = 0.;
      std::vector<double> Pg(g_controlNodeDofs.size(), 0.);

      double r = 0.;
      double d = 0.;
      double s = 0.;
      double s1 = 0.;

      std::vector < adept::adouble > gradZg(dim, 0.);
      std::vector < adept::adouble > gradXg(dim, 0.);
      std::vector < adept::adouble > gradYg(dim, 0.);

      for (unsigned i = 0; i < nDofs; i++) {

        ZOldg += solZOld[i] * phi[i];
        Zg += solZ[i] * phi[i];
        Xg += solX[i] * phi[i];
        Yg += solY[i] * phi[i];
        for(unsigned j = 0; j < g_controlNodeDofs.size(); j++) Pg[j] += solP[j][i] * phi[i];

        r += solE[i] * phi[i];

        s += solS[i] * phi[i];
        s1 += solS1[i] * phi[i];

        for (unsigned d = 0; d < dim; d++) {
          gradZg[d] += solZ[i] * gradPhi[i * dim + d];
          gradXg[d] += solX[i] * gradPhi[i * dim + d];
          gradYg[d] += solY[i] * gradPhi[i * dim + d];
        }
      }


      // *** phiA_i loop ***
      for (unsigned i = 0; i < nDofs; i++) {
        unsigned coordXDof  = msh->GetSolutionDof(i, iel, coordXType);

        adept::adouble aResZ = (Zg - ZOldg) / dt* phi[i];
        adept::adouble aResX = - (CsC > 0.5) * (r - (1. - beta) * Zg - beta * Yg) * phi[i];
        adept::adouble aResY = 0;

        for(unsigned j = 0; j < g_controlNodeDofs.size(); j++){
          aResZ += h * a * Pg[j] * w[jTMP][j] * phi[i] + h * b * Pg[j] * u1[j] * phi[i];
          aResX += (CsC > 0.5) * ( (1. - beta) * h * Pg[j] * w[jTMP][j] + beta * (+ h * Pg[j] * y1_vec[j])) * phi[i];
          aResY +=  h * a * Pg[j] * y1_vec[j] * phi[i] + h * b * Pg[j] * u1[j] * phi[i];
        }


        for (unsigned d = 0; d < dim; d++) { // second index j in each equation
          aResZ +=  mu * gradPhi[i * dim + d] * gradZg[d]; // diffusion
          aResX +=  mu * gradPhi[i * dim + d] * gradXg[d]; // diffusion
          aResY +=  mu * gradPhi[i * dim + d] * gradYg[d]; // diffusion
        }

        if(withDisturbance){
          aResZ += - s * phi[i];
          aResX += (CsC > 0.5) * s1 * phi[i];
          aResY += - s * phi[i];
        }


        aRes[0 * nDofs + i] += aResZ * weight;
        aRes[1 * nDofs + i] += aResX * weight;
        aRes[2 * nDofs + i] += aResY * weight;

      } // end phiA_i loop
    }

    //--------------------------------------------------------------------------------------------------------
    // Add the local Matrix/Vector into the global Matrix/Vector

    res.resize(nDofsAll);
    //copy the value of the adept::adoube mRes in double Res and store
    for (int i = 0; i < nDofsAll; i++) {
      res[i] = -aRes[i].value();
    }

    // define the dependent variables
    s.dependent(aRes.data(), nDofsAll);
    s.independent(solZ.data(), nDofs);
    s.independent(solX.data(), nDofs);
    s.independent(solY.data(), nDofs);

    Jac.assign(nDofsAll * nDofsAll, 0.);
    // get the jacobian matrix (ordered by column)
    s.jacobian(Jac.data(), true);

    RES->add_vector_blocked(res, sysDof);
    KK->add_matrix_blocked(Jac, sysDof, sysDof);

    s.clear_independents();
    s.clear_dependents();

  } //end element loop for each process

  RES->close();
  KK->close();

}


void AssembleResP(MultiLevelProblem& ml_prob) {
  adept::Stack& s = FemusInit::_adeptStack;

  auto* mlPdeSys = &ml_prob.get_system<LinearImplicitSystem>("LP");
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh* msh = ml_prob._ml_msh->GetLevel(level);
  MultiLevelSolution* mlSol = ml_prob._ml_sol;
  Solution* sol = ml_prob._ml_sol->GetSolutionLevel(level);

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
  SparseMatrix* KK = pdeSys->_KK;
  NumericVector* RES = pdeSys->_RES;

  RES->zero();
  KK->zero();

  const unsigned pIndex    = mlSol->GetIndex("P");
  const unsigned pType     = mlSol->GetSolutionType(pIndex);
  const unsigned pPdeIndex = mlPdeSys->GetSolPdeIndex("P");

  const unsigned dim        = msh->GetDimension();
  const unsigned coordXType = 2;
  const unsigned iproc      = msh->processor_id();

  // ------------- element loop -------------
  for (unsigned iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; ++iel) {

    const short unsigned ielGeom = msh->GetElementType(iel);
    const unsigned nDofs = msh->GetElementDofNumber(iel, pType);

    // coordinates at element nodes
    std::vector<std::vector<double>> X(dim, std::vector<double>(nDofs));
    for (unsigned i = 0; i < nDofs; ++i) {
      const unsigned xd = msh->GetSolutionDof(i, iel, coordXType);
      for (unsigned k = 0; k < dim; ++k) X[k][i] = (*msh->_topology->_Sol[k])(xd);
    }

    // global dof map
    std::vector<unsigned> sysDof(nDofs);
    for (unsigned i = 0; i < nDofs; ++i) {
      sysDof[i] = pdeSys->GetSystemDof(pIndex, pPdeIndex, i, iel);
    }

    // local unknowns as adoubles
    std::vector<adept::adouble> p(nDofs);
    for (unsigned i = 0; i < nDofs; ++i) {
      const unsigned gd = msh->GetSolutionDof(i, iel, pType);
      p[i] = (*sol->_Sol[pIndex])(gd);
    }

    s.new_recording();

    // local residual
    std::vector<adept::adouble> aRes(nDofs, 0.0);

    // Gauss integration: a(u,v) = ∫ grad p · grad v
    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][pType]->GetGaussPointNumber(); ++ig) {
      double w;
      std::vector<double> phi, dphi;
      msh->_finiteElement[ielGeom][pType]->Jacobian(X, ig, w, phi, dphi);

      // grad p at GP
      std::vector<adept::adouble> gradPg(dim, 0.0);
      for (unsigned a = 0; a < nDofs; ++a)
        for (unsigned k = 0; k < dim; ++k)
          gradPg[k] += p[a] * dphi[a * dim + k];

      // test with grad v_i
      for (unsigned i = 0; i < nDofs; ++i) {
        adept::adouble gi = 0.0;
        for (unsigned k = 0; k < dim; ++k) gi += dphi[i * dim + k] * gradPg[k];
        aRes[i] += gi * w;
      }
    }

    // assemble local residual and Jacobian
    std::vector<double> Re(nDofs);
    for (unsigned i = 0; i < nDofs; ++i) Re[i] = -aRes[i].value();   // move to RHS

    s.dependent(aRes.data(), nDofs);
    s.independent(p.data(),  nDofs);

    std::vector<double> Je(nDofs * nDofs, 0.0);
    s.jacobian(Je.data(), true);

    RES->add_vector_blocked(Re, sysDof);
    KK->add_matrix_blocked(Je, sysDof, sysDof);

    s.clear_independents();
    s.clear_dependents();
  }

  const unsigned meshDofP = gdof;  // this is what GetControlNodeIndices gave you (global mesh dof)
  const unsigned sysRow = MapMeshDofToSystemRowP(msh, pdeSys, pIndex, pPdeIndex, pType, meshDofP);

  if (sysRow != static_cast<unsigned>(-1)) {
    RES->add(sysRow, 1.0);
  }

  RES->close();
  KK->close();
}

void AssembleLaplacian_s(MultiLevelProblem& ml_prob) {
  AssembleLaplacianCore(ml_prob, "Lap_s", "s");
}

void AssembleLaplacian_s1(MultiLevelProblem& ml_prob) {
  AssembleLaplacianCore(ml_prob, "Lap_s1", "s1");
}

static void AssembleLaplacianCore(MultiLevelProblem& ml_prob,
                                  const char* system_name,
                                  const char* var_name) {
  adept::Stack& s = FemusInit::_adeptStack;

  auto* mlPdeSys = &ml_prob.get_system<TransientLinearImplicitSystem>(system_name);
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh* msh = ml_prob._ml_msh->GetLevel(level);
  MultiLevelSolution* mlSol = ml_prob._ml_sol;
  Solution* sol = mlSol->GetSolutionLevel(level);

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
  SparseMatrix* KK = pdeSys->_KK;
  NumericVector* RES = pdeSys->_RES;

  RES->zero();
  KK->zero();

  const unsigned uIndex    = mlSol->GetIndex(var_name);
  const unsigned uType     = mlSol->GetSolutionType(uIndex);
  const unsigned uPdeIndex = mlPdeSys->GetSolPdeIndex(var_name);

  const unsigned dim = msh->GetDimension();
  const unsigned coordXType = 2;
  const unsigned iproc = msh->processor_id();

  for (unsigned iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; ++iel) {
    const short unsigned ielGeom = msh->GetElementType(iel);
    const unsigned nDofs = msh->GetElementDofNumber(iel, uType);

    std::vector<std::vector<double>> X(dim, std::vector<double>(nDofs));
    for (unsigned i = 0; i < nDofs; ++i) {
      const unsigned xd = msh->GetSolutionDof(i, iel, coordXType);
      for (unsigned k = 0; k < dim; ++k) X[k][i] = (*msh->_topology->_Sol[k])(xd);
    }

    std::vector<unsigned> sysDof(nDofs);
    std::vector<adept::adouble> u(nDofs);
    for (unsigned i = 0; i < nDofs; ++i) {
      sysDof[i] = pdeSys->GetSystemDof(uIndex, uPdeIndex, i, iel);
      const unsigned gd = msh->GetSolutionDof(i, iel, uType);
      u[i] = (*sol->_Sol[uIndex])(gd);
    }

    s.new_recording();
    std::vector<adept::adouble> aRes(nDofs, 0.0);

    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][uType]->GetGaussPointNumber(); ++ig) {
      double wgt;
      std::vector<double> phi, dphi;
      msh->_finiteElement[ielGeom][uType]->Jacobian(X, ig, wgt, phi, dphi);

      std::vector<adept::adouble> gradUg(dim, 0.0);
      for (unsigned a = 0; a < nDofs; ++a)
        for (unsigned k = 0; k < dim; ++k)
          gradUg[k] += u[a] * dphi[a * dim + k];

      for (unsigned i = 0; i < nDofs; ++i) {
        adept::adouble gi = 0.0;
        for (unsigned k = 0; k < dim; ++k) gi += dphi[i * dim + k] * gradUg[k];
        aRes[i] += mu * gi * wgt;   // mu is your global diffusion coefficient
      }
    }

    std::vector<double> Re(nDofs);
    for (unsigned i = 0; i < nDofs; ++i) Re[i] = -aRes[i].value();

    s.dependent(aRes.data(), nDofs);
    s.independent(u.data(),  nDofs);

    std::vector<double> Je(nDofs * nDofs, 0.0);
    s.jacobian(Je.data(), true);

    RES->add_vector_blocked(Re, sysDof);
    KK->add_matrix_blocked(Je, sysDof, sysDof);

    s.clear_independents();
    s.clear_dependents();
  }

  RES->close();
  KK->close();
}














std::vector<double> PrecomputePstarIntegrals(Solution* sol) {
  Mesh* msh = sol->GetMesh();
  const unsigned dim = msh->GetDimension();
  unsigned iproc = msh->processor_id();

  const unsigned cIndex = sol->GetIndex("C");  // element-wise indicator CsC

  std::vector<double> globalIntegral(g_controlNodeDofs.size(), 0.);

  // nodal weights:
  //   PStar(i)       = ∫Ω       P φ_i dx
  //   CStarCPStar(i) = ∫Ω_C     P φ_i dx   with Ω_C selected by CsC > 0.5


  for(unsigned j = 0; j < g_controlNodeDofs.size(); j++){

    // global scalar integral ∫Ω P dx (for PStar scalar)
    double localIntegral = 0.0;

    const unsigned pIndexj = g_idx.P[j];
    const unsigned pType     = sol->GetSolutionType(pIndexj);
    const unsigned coordXType = 2;  // quadratic coordinates

    const unsigned pStarIndexj       = g_idx.PStar[j];
    const unsigned cStarCPStarIndexj = g_idx.CStar[j];


    sol->_Sol[pStarIndexj]->zero();
    sol->_Sol[cStarCPStarIndexj]->zero();



  // element loop (each process owns its range)
  for (unsigned iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; ++iel) {
    short unsigned ielGeom = msh->GetElementType(iel);
    unsigned nDofs = msh->GetElementDofNumber(iel, pType);

    // element-wise indicator CsC (discontinuous 0/1 on elements)
    double CsC = (*sol->_Sol[cIndex])(iel);
    const double inC = (CsC > 0.5) ? 1.0 : 0.0;

    // coordinates at element nodes
    std::vector<std::vector<double>> coordX(dim, std::vector<double>(nDofs));
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned coordXDof = msh->GetSolutionDof(i, iel, coordXType);
      for (unsigned k = 0; k < dim; k++)
        coordX[k][i] = (*msh->_topology->_Sol[k])(coordXDof);
    }

    // nodal P values on this element
    std::vector<double> pVal(nDofs);
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned pDof = msh->GetSolutionDof(i, iel, pType);
      pVal[i] = (*sol->_Sol[pIndexj])(pDof);
    }

    // Gauss integration
    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][pType]->GetGaussPointNumber(); ig++) {
      double weight;
      std::vector<double> phi, gradPhi;
      msh->_finiteElement[ielGeom][pType]->Jacobian(coordX, ig, weight, phi, gradPhi);

      double Pg = 0.0;
      for (unsigned i = 0; i < nDofs; i++) Pg += pVal[i] * phi[i];

      // global integral ∫Ω P dx
      localIntegral += Pg * weight;

      // nodal weights
      for (unsigned i = 0; i < nDofs; i++) {
        unsigned pDof = msh->GetSolutionDof(i, iel, pType);

        const double contrib = Pg * phi[i] * weight;

        // nodeWeightsP[pDof] += contrib;              // full domain Ω
        // nodeWeightsC[pDof] += inC * contrib;        // restricted to C

        sol->_Sol[pStarIndexj]->add(pDof,contrib);
        sol->_Sol[cStarCPStarIndexj]->add(pDof,inC*contrib);
      }
    }
  }


  // MPI reductions TODO
  MPI_Allreduce(&localIntegral, &globalIntegral[j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // // Reduce nodal weights in-place
  // MPI_Allreduce(MPI_IN_PLACE, nodeWeightsP.data(),
  //               static_cast<int>(nodeWeightsP.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //
  // MPI_Allreduce(MPI_IN_PLACE, nodeWeightsC.data(),
  //               static_cast<int>(nodeWeightsC.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // // Scatter to FEMuS solution fields PStar and CStarCPStar
  // for (std::size_t gdof = 0; gdof < nDofsGlobal; ++gdof) {
  //   sol->_Sol[pStarIndex]->set(static_cast<unsigned>(gdof), nodeWeightsP[gdof]);
  //   sol->_Sol[cStarCPStarIndex]->set(static_cast<unsigned>(gdof), nodeWeightsC[gdof]);
  // }

  sol->_Sol[pStarIndexj]->close();
  sol->_Sol[cStarCPStarIndexj]->close();
  }

  return globalIntegral;
}





std::vector<double> ComputeL2NormCascadeOverC(Solution* sol, unsigned cascadeIterations) {
  Mesh* msh = sol->GetMesh();
  const unsigned dim = msh->GetDimension();
  unsigned iproc = msh->processor_id();

  unsigned solIndexC = sol->GetIndex("C");
  unsigned coordXType = 2;

  std::vector<unsigned> solIndexE(cascadeIterations);
  std::vector<unsigned> solTypeE(cascadeIterations);
  for (unsigned j = 0; j < cascadeIterations; j++) {
    std::string Ej = "E" + std::to_string(j);
    solIndexE[j] = sol->GetIndex(Ej.c_str());
    solTypeE[j] = sol->GetSolutionType(solIndexE[j]);
  }

  std::vector<double> local_integral(cascadeIterations, 0.0);

  for (unsigned iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {
    double CsC = (*sol->_Sol[solIndexC])(iel);
    if (CsC < 0.5) continue;  // outside C

    short unsigned ielGeom = msh->GetElementType(iel);
    unsigned nDofs = msh->GetElementDofNumber(iel, solTypeE[0]);

    std::vector<std::vector<double>> coordX(dim, std::vector<double>(nDofs));
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned coordXDof = msh->GetSolutionDof(i, iel, coordXType);
      for (unsigned k = 0; k < dim; k++)
        coordX[k][i] = (*msh->_topology->_Sol[k])(coordXDof);
    }

    // store E_j values for each cascade at element dofs
    std::vector<std::vector<double>> solE(cascadeIterations, std::vector<double>(nDofs));
    for (unsigned j = 0; j < cascadeIterations; j++) {
      for (unsigned i = 0; i < nDofs; i++) {
        unsigned eDof = msh->GetSolutionDof(i, iel, solTypeE[j]);
        solE[j][i] = (*sol->_Sol[solIndexE[j]])(eDof);
      }
    }

    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solTypeE[0]]->GetGaussPointNumber(); ig++) {
      double weight;
      std::vector<double> phi, gradPhi;
      msh->_finiteElement[ielGeom][solTypeE[0]]->Jacobian(coordX, ig, weight, phi, gradPhi);

      for (unsigned j = 0; j < cascadeIterations; j++) {
        double Eg = 0.;
        for (unsigned i = 0; i < nDofs; i++) Eg += solE[j][i] * phi[i];
        local_integral[j] += Eg * Eg * weight;
      }
    }
  }

  std::vector<double> global_integral(cascadeIterations, 0.0);
  MPI_Allreduce(local_integral.data(), global_integral.data(), cascadeIterations,
                MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  for (double& val : global_integral) val = std::sqrt(val);
  return global_integral;
}


std::vector<unsigned> GetControlNodeIndices(const Mesh* msh,
                                            const std::vector<std::vector<double>>& points) {
  const unsigned dim = msh->GetDimension();
  if (points.empty()) return {};

  // Coordinate vectors (distributed)
  const NumericVector* X0 = msh->_topology->_Sol[0];
  const NumericVector* X1 = (dim > 1) ? msh->_topology->_Sol[1] : nullptr;
  const NumericVector* X2 = (dim > 2) ? msh->_topology->_Sol[2] : nullptr;

  // Owned global index range on THIS rank (unique ownership)
  const unsigned first = X0->first_local_index();
  const unsigned last  = X0->last_local_index();

  // Create MPI datatype + op once per call (cheap; you can also cache globally)
  MPI_Datatype MPI_MinPair;
  MPI_Op       MPI_MinPairOp;

  // MinPair is {double, uint64} with potential padding; define explicit MPI struct
  {
    MinPair tmp;
    MPI_Aint displs[2];
    int      blens[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_UNSIGNED_LONG_LONG};

    MPI_Aint base;
    MPI_Get_address(&tmp, &base);
    MPI_Get_address(&tmp.dist2, &displs[0]);
    MPI_Get_address(&tmp.gid,   &displs[1]);
    displs[0] -= base;
    displs[1] -= base;

    MPI_Type_create_struct(2, blens, displs, types, &MPI_MinPair);
    MPI_Type_commit(&MPI_MinPair);

    MPI_Op_create(&mpi_minpair_op, /*commute=*/1, &MPI_MinPairOp);
  }

  std::vector<unsigned> controlIndices;
  controlIndices.reserve(points.size());

  for (const auto& x0 : points) {
    if (x0.size() < dim) {
      std::cerr << "GetControlNodeIndices: point has wrong dimension\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MinPair local;
    local.dist2 = std::numeric_limits<double>::infinity();
    local.gid   = std::numeric_limits<std::uint64_t>::max();

    // Search only owned nodes
    for (unsigned gdof = first; gdof < last; ++gdof) {
      const double dx = (*X0)(gdof) - x0[0];
      double d2 = dx * dx;

      if (dim > 1) {
        const double dy = (*X1)(gdof) - x0[1];
        d2 += dy * dy;
      }
      if (dim > 2) {
        const double dz = (*X2)(gdof) - x0[2];
        d2 += dz * dz;
      }

      // local argmin, tie-break by smaller gdof
      if (d2 < local.dist2 || (d2 == local.dist2 && std::uint64_t(gdof) < local.gid)) {
        local.dist2 = d2;
        local.gid   = gdof;
      }
    }

    MinPair global = local;
    MPI_Allreduce(&local, &global, 1, MPI_MinPair, MPI_MinPairOp, MPI_COMM_WORLD);

    if (global.gid == std::numeric_limits<std::uint64_t>::max() ||
      !std::isfinite(global.dist2)) {
      std::cerr << "GetControlNodeIndices: failed to find a global closest node\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
      }

      controlIndices.push_back(static_cast<unsigned>(global.gid));
  }

  MPI_Op_free(&MPI_MinPairOp);
  MPI_Type_free(&MPI_MinPair);

  return controlIndices;
}


WNodeIDs GetGlobalNodeIDsForW(const Mesh* msh, unsigned elemID) {
  const unsigned solType = 2; // quadratic coordinates

  WNodeIDs ids;
  ids.W0 = msh->GetSolutionDof(0, elemID, solType);
  ids.X1 = msh->GetSolutionDof(1, elemID, solType);
  ids.Y1 = msh->GetSolutionDof(2, elemID, solType);

  return ids;
}



ODEStageOut computeODESystemStage(
  const Solution& sol,
  unsigned idxZi,
  unsigned idxXi,
  unsigned idxEi,
  const std::vector<unsigned>& idxP,       // size nCtrl
  const std::vector<unsigned>& idxPStar,   // size nCtrl
  const std::vector<unsigned>& idxCStar,   // size nCtrl
  const std::vector<double>& wOldStage,    // size nCtrl
  const ODEParams& par,
  MPI_Comm comm)
{
  const std::size_t nCtrl = idxP.size();
  ODEStageOut out;
  out.x1.assign(nCtrl, 0.0);
  out.y1.assign(nCtrl, 0.0);
  out.u1.assign(nCtrl, 0.0);
  out.w.assign(nCtrl, 0.0);

  if (idxPStar.size() != nCtrl || idxCStar.size() != nCtrl || wOldStage.size() != nCtrl) {
    std::cerr << "computeODESystemStage: inconsistent nCtrl sizes\n";
    MPI_Abort(comm, 1);
  }

  const NumericVector* ZVec = sol._Sol[idxZi];
  const NumericVector* XVec = sol._Sol[idxXi];
  const NumericVector* EVec = sol._Sol[idxEi];

  // One Allreduce for all controls:
  // For each control p: [PstarZ, CstarZ, PstarX, CstarP, CstarE] => 5 scalars.
  std::vector<double> local(5 * nCtrl, 0.0), global(5 * nCtrl, 0.0);

  for (std::size_t p = 0; p < nCtrl; ++p) {
    const NumericVector* PStarVecp       = sol._Sol[idxPStar[p]];
    const NumericVector* CStarCPStarVecp = sol._Sol[idxCStar[p]];
    const NumericVector* PVecp           = sol._Sol[idxP[p]];

    const unsigned first = PStarVecp->first_local_index();
    const unsigned last  = PStarVecp->last_local_index();

    double PstarZ = 0.0;
    double CstarZ = 0.0;
    double PstarX = 0.0;
    double CstarP = 0.0;
    double CstarE = 0.0;

    for (unsigned gdof = first; gdof < last; ++gdof) {
      const double wP  = (*PStarVecp)(gdof);
      const double wPC = (*CStarCPStarVecp)(gdof);

      const double Zi = (*ZVec)(gdof);
      const double Xi = (*XVec)(gdof);
      const double Ei = (*EVec)(gdof);
      const double Pi = (*PVecp)(gdof);

      PstarZ += wP  * Zi;
      CstarZ += wPC * Zi;
      PstarX += wP  * Xi;
      CstarP += wPC * Pi;
      CstarE += wPC * Ei;
    }

    local[5 * p + 0] = PstarZ;
    local[5 * p + 1] = CstarZ;
    local[5 * p + 2] = PstarX;
    local[5 * p + 3] = CstarP;
    local[5 * p + 4] = CstarE;
  }

  MPI_Allreduce(local.data(), global.data(),
                static_cast<int>(global.size()),
                MPI_DOUBLE, MPI_SUM, comm);

  // Now compute x1,w,y1,u1 per control p
  const double dt    = par.dt;
  const double alpha = par.alpha;
  const double beta  = par.beta;
  const double a     = par.a;
  const double b     = par.b;
  const double h     = par.h;

  const double one_minus_beta = 1.0 - beta;
  const double denom_adt = (1.0 - a * dt);

  // Basic safety checks to avoid NaNs
  if (std::abs(denom_adt) < 1e-14) {
    std::cerr << "computeODESystemStage: 1 - a*dt too small\n";
    MPI_Abort(comm, 1);
  }
  if (std::abs(a) < 1e-14) {
    std::cerr << "computeODESystemStage: a too small (division by a)\n";
    MPI_Abort(comm, 1);
  }
  if (std::abs(alpha) < 1e-30) {
    std::cerr << "computeODESystemStage: alpha too small\n";
    MPI_Abort(comm, 1);
  }

  for (std::size_t p = 0; p < nCtrl; ++p) {
    const double PstarZ = global[5 * p + 0];
    const double CstarZ = global[5 * p + 1];
    const double PstarX = global[5 * p + 2];
    const double CstarP = global[5 * p + 3];
    const double CstarE = global[5 * p + 4];

    const double lhs =
    -a + (h * h * b * b * CstarP / alpha) *
    ((dt * one_minus_beta / denom_adt) - (beta / a));

    // Guard against divide-by-zero
    if (std::abs(lhs) < 1e-14) {
      std::cerr << "computeODESystemStage: lhs too small at p=" << p << "\n";
      MPI_Abort(comm, 1);
    }

    const double rhs1 =
    -h * h * CstarP *
    ( one_minus_beta * (wOldStage[p] / denom_adt)
    + (h * b * b / alpha) * (-(one_minus_beta * dt / denom_adt) + (beta / a)) * PstarX );

    const double rhs2 =
    -h * a * PstarX + h * (CstarE - one_minus_beta * CstarZ);

    const double rhs = rhs1 + rhs2;

    const double x1 = rhs / lhs;
    const double wnew =
    (dt / denom_adt) * (wOldStage[p] / dt + (b * b / alpha) * (x1 - h * PstarX));
    const double y1 = -(b * b / (a * alpha)) * (-h * PstarX + x1);
    const double u1 = (-h * b * PstarX + b * x1) / alpha;

    out.x1[p] = x1;
    out.w[p]  = wnew;
    out.y1[p] = y1;
    out.u1[p] = u1;
  }

  return out;
}


static void computeODESystem(
  const Solution& sol,
  const Indices& idx,
  const std::vector<unsigned>& controlNodeDofs,
  const ODEParams& par,
  unsigned jTMP,
  std::vector<std::vector<double>>& w,
  const std::vector<std::vector<double>>& wOld,
  std::vector<double>& x1_vec,
  std::vector<double>& y1_vec,
  std::vector<double>& u1
) {
  const std::size_t nCtrl = controlNodeDofs.size();

  // sizes
  x1_vec.assign(nCtrl, 0.0);
  y1_vec.assign(nCtrl, 0.0);
  u1.assign(nCtrl, 0.0);
  w[jTMP].assign(nCtrl, 0.0);

  // compute in one shot
  ODEStageOut out = computeODESystemStage(
    sol,
    idx.Zi, idx.Xi, idx.Ei,
    idx.P, idx.PStar, idx.CStar,
    wOld[jTMP],
    par,
    MPI_COMM_WORLD);

  x1_vec = std::move(out.x1);
  y1_vec = std::move(out.y1);
  u1     = std::move(out.u1);
  w[jTMP]= std::move(out.w);
}


static unsigned MapMeshDofToSystemRowP(const Mesh* msh, LinearEquationSolver* pdeSys, unsigned pIndex,
                                       unsigned pPdeIndex, unsigned pType, unsigned meshDofP)
{
  // We search elements owned by this rank. For a point-load you only need
  // the system row on the owning rank; RES->add handles parallel assembly.
  const unsigned iproc = msh->processor_id();

  for (unsigned iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; ++iel) {
    const unsigned nDofs = msh->GetElementDofNumber(iel, pType);
    for (unsigned i = 0; i < nDofs; ++i) {
      const unsigned md = msh->GetSolutionDof(i, iel, pType);
      if (md == meshDofP) {
        return pdeSys->GetSystemDof(pIndex, pPdeIndex, i, iel);
      }
    }
  }

  // Not found on this rank (likely not owned). Return invalid.
  return static_cast<unsigned>(-1);
}




