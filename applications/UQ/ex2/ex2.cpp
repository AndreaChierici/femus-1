
#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "TransientSystem.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "NumericVector.hpp"
#include "adept.h"

#include "petsc.h"
#include "petscmat.h"
#include "PetscMatrix.hpp"

#include "slepceps.h"

#include "../include/sfem_assembly.hpp"

using namespace femus;


bool SetBoundaryCondition(const std::vector < double >& x, const char SolName[], double& value, const int facename, const double time)
{
  bool dirichlet = true; //dirichlet
  value = 0.;
  return dirichlet;
}

void GetEigenPair(MultiLevelProblem& ml_prob, const int& numberOfEigPairs, std::vector < std::pair<double, double> >& eigenvalues);

void GetQuantityOfInterest(MultiLevelProblem& ml_prob, std::vector < double >&  QoI, const unsigned& m, const double& domainMeasure);

void GetStochasticData(std::vector <double>& QoI);

void PlotStochasticData();

//BEGIN stochastic data
double L = 4; // correlation length of the covariance function
double domainMeasure = 1.; //measure of the domain
unsigned totMoments = 6;
std::vector <double> moments(totMoments, 0.); //initialization
std::vector <double> cumulants(totMoments, 0.); //initialization
double meanQoI = 0.; //initialization
double varianceQoI = 0.; //initialization
double stdDeviationQoI = 0.; //initialization
unsigned M = 10; //number of samples for the Monte Carlo
//END

unsigned numberOfUniformLevels = 3;

int main(int argc, char** argv)
{


  //BEGIN eigenvalue problem instances
  PetscErrorCode ierr;
  ierr = SlepcInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL);

  numberOfEigPairs = 1; //number of eigenpairs desired

  eigenvalues.resize(numberOfEigPairs); //this is where we store the eigenvalues

  //END


  //BEGIN deterministic FEM instances

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, argv, MPI_COMM_WORLD);

  MultiLevelMesh mlMsh;
  double scalingFactor = 1.;
  unsigned numberOfSelectiveLevels = 0;
  mlMsh.ReadCoarseMesh("../input/square.neu", "fifth", scalingFactor);
  mlMsh.RefineMesh(numberOfUniformLevels + numberOfSelectiveLevels, numberOfUniformLevels , NULL);

  unsigned dim = mlMsh.GetDimension();

  MultiLevelSolution mlSol(&mlMsh);

  // add variables to mlSol
  mlSol.AddSolution("u", LAGRANGE, SECOND, 2);

  for(unsigned i = 0; i < numberOfEigPairs; i++) {
    char name[10];
    sprintf(name, "egnf%d", i);
    mlSol.AddSolution(name, LAGRANGE, SECOND, 0, false);
  }

  mlSol.Initialize("All");

  mlSol.AttachSetBoundaryConditionFunction(SetBoundaryCondition);

  // ******* Set boundary conditions *******
  mlSol.GenerateBdc("All");

  MultiLevelProblem ml_prob(&mlSol);

  // ******* Add FEM system to the MultiLevel problem *******
  LinearImplicitSystem& system = ml_prob.add_system < LinearImplicitSystem > ("UQ");
  system.AddSolutionToSystemPDE("u");

  // ******* System FEM Assembly *******
  system.SetAssembleFunction(AssembleUQSys);
  system.SetMaxNumberOfLinearIterations(1);
  //system.SetAssembleFunction(AssembleFEM);
  // ******* set MG-Solver *******
  system.SetMgType(V_CYCLE);

  system.SetAbsoluteLinearConvergenceTolerance(1.e-50);
  //   system.SetNonLinearConvergenceTolerance(1.e-9);
//   system.SetMaxNumberOfNonLinearIterations(20);

  system.SetNumberPreSmoothingStep(1);
  system.SetNumberPostSmoothingStep(1);

  // ******* Set Preconditioner *******
  system.SetMgSmoother(GMRES_SMOOTHER);

  system.init();

  // ******* Set Smoother *******
  system.SetSolverFineGrids(GMRES);

  system.SetPreconditionerFineGrids(ILU_PRECOND);

  system.SetTolerances(1.e-20, 1.e-20, 1.e+50, 100);
  //END


  GetEigenPair(ml_prob, numberOfEigPairs, eigenvalues); //solve the generalized eigenvalue problem and compute the eigenpairs

  for(int i = 0; i < numberOfEigPairs; i++) {
    std::cout << eigenvalues[i].first << " " << eigenvalues[i].second << std::endl;
  }


  std::vector <double> QoI(M, 0.);

  for(unsigned m = 0; m < M; m++) {

    system.MGsolve();

    GetQuantityOfInterest(ml_prob, QoI, m, domainMeasure);

  }



//   for(unsigned m = 0; m < M; m++) {
//     std::cout << "QoI[" << m << "] = " << QoI[m] << std::endl;
//   }

  GetStochasticData(QoI);

  PlotStochasticData();

  // ******* Print solution *******
  mlSol.SetWriter(VTK);
  std::vector<std::string> print_vars;
  print_vars.push_back("All");
  mlSol.GetWriter()->SetDebugOutput(true);
  mlSol.GetWriter()->Write(DEFAULT_OUTPUTDIR, "biquadratic", print_vars, 0);

  //ierr = SlepcFinalize();
  //CHKERRQ(ierr);

  return 0;

} //end main

void GetEigenPair(MultiLevelProblem& ml_prob, const int& numberOfEigPairs, std::vector < std::pair<double, double> >& eigenvalues)
{
//void GetEigenPair(MultiLevelProblem & ml_prob, Mat &CCSLEPc, Mat &MMSLEPc) {

  LinearImplicitSystem* mlPdeSys  = &ml_prob.get_system<LinearImplicitSystem> ("UQ");   // pointer to the linear implicit system named "Poisson"

  unsigned level = numberOfUniformLevels - 1;

  double varianceInput = stdDeviationInput * stdDeviationInput;

  Mesh*                    msh = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object
  elem*                     el = msh->el;  // pointer to the elem object in msh (level)

  MultiLevelSolution*    mlSol = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*                sol = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  SparseMatrix*             MM = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27

  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)
  unsigned    nprocs = msh->n_processors(); // get the process_id (for parallel computation)


  //solution variable
  unsigned soluIndex;
  soluIndex = mlSol->GetIndex("u");    // get the position of "u" in the ml_sol object
  unsigned solType = mlSol->GetSolutionType(soluIndex);    // get the finite element type for "u"

  unsigned soluPdeIndex;
  soluPdeIndex = mlPdeSys->GetSolPdeIndex("u");    // get the position of "u" in the pdeSys object

  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  vector < vector < double > > x1(dim);    // local coordinates
  vector < vector < double > > x2(dim);    // local coordinates
  for(unsigned k = 0; k < dim; k++) {
    x1[k].reserve(maxSize);
    x2[k].reserve(maxSize);
  }

  vector <double> phi_x; // local test function first order partial derivatives
  vector < double >* nullDoublePointer = NULL;

  phi_x.reserve(maxSize * dim);

  vector< int > l2GMap1; // local to global mapping
  vector< int > l2GMap2; // local to global mapping
  l2GMap1.reserve(maxSize);
  l2GMap2.reserve(maxSize);

  vector < double > MMlocal;
  MMlocal.reserve(maxSize * maxSize);

  vector < double > CClocal;
  CClocal.reserve(maxSize * maxSize);

  MM->zero(); // Set to zero all the entries of the Global Matrix

  int MM_size = msh->_dofOffset[solType][nprocs];
  int MM_local_size = msh->_dofOffset[solType][iproc + 1] - msh->_dofOffset[solType][iproc];

  SparseMatrix* CC;
  CC = SparseMatrix::build().release();
  CC->init(MM_size, MM_size, MM_local_size, MM_local_size, MM_local_size, MM_size - MM_local_size);
  CC->zero();

  for(int kproc = 0; kproc < nprocs; kproc++) {
    for(int jel = msh->_elementOffset[kproc]; jel < msh->_elementOffset[kproc + 1]; jel++) {

      short unsigned ielGeom2;
      unsigned nDof2;
      unsigned nDofx2;

      if(iproc == kproc) {
        ielGeom2 = msh->GetElementType(jel);
        nDof2  = msh->GetElementDofNumber(jel, solType);    // number of solution element dofs
        nDofx2 = msh->GetElementDofNumber(jel, xType);    // number of coordinate element dofs
      }

      MPI_Bcast(&ielGeom2, 1, MPI_UNSIGNED_SHORT, kproc, MPI_COMM_WORLD);
      MPI_Bcast(&nDof2, 1, MPI_UNSIGNED, kproc, MPI_COMM_WORLD);
      MPI_Bcast(&nDofx2, 1, MPI_UNSIGNED, kproc, MPI_COMM_WORLD);

      // resize local arrays
      l2GMap2.resize(nDof2);

      for(int k = 0; k < dim; k++) {
        x2[k].resize(nDofx2);
      }

      // local storage of global mapping and solution
      if(iproc == kproc) {
        for(unsigned j = 0; j < nDof2; j++) {
          l2GMap2[j] = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, j, jel);  // global to global mapping between solution node and pdeSys dof
        }
      }
      MPI_Bcast(&l2GMap2[0], nDof2, MPI_UNSIGNED, kproc, MPI_COMM_WORLD);

      // local storage of coordinates
      if(iproc == kproc) {
        for(unsigned j = 0; j < nDofx2; j++) {
          unsigned xDof  = msh->GetSolutionDof(j, jel, xType);  // global to global mapping between coordinates node and coordinate dof
          for(unsigned k = 0; k < dim; k++) {
            x2[k][j] = (*msh->_topology->_Sol[k])(xDof);  // global extraction and local storage for the element coordinates
          }
        }
      }
      for(unsigned k = 0; k < dim; k++) {
        MPI_Bcast(& x2[k][0], nDofx2, MPI_DOUBLE, kproc, MPI_COMM_WORLD);
      }

      unsigned jgNumber = msh->_finiteElement[ielGeom2][solType]->GetGaussPointNumber();
      vector < vector < double > > xg2(jgNumber);
      vector <double> weight2(jgNumber);
      vector < vector <double> > phi2(jgNumber);  // local test function

      for(unsigned jg = 0; jg < jgNumber; jg++) {
        msh->_finiteElement[ielGeom2][solType]->Jacobian(x2, jg, weight2[jg], phi2[jg], phi_x, *nullDoublePointer);

        xg2[jg].assign(dim, 0.);

        for(unsigned j = 0; j < nDof2; j++) {
          for(unsigned k = 0; k < dim; k++) {
            xg2[jg][k] += x2[k][j] * phi2[jg][j];
          }
        }
      }

      // element loop: each process loops only on the elements that owns
      for(int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

        short unsigned ielGeom1 = msh->GetElementType(iel);
        unsigned nDof1  = msh->GetElementDofNumber(iel, solType);    // number of solution element dofs
        unsigned nDofx1 = msh->GetElementDofNumber(iel, xType);    // number of coordinate element dofs

        // resize local arrays
        l2GMap1.resize(nDof1);

        for(int k = 0; k < dim; k++) {
          x1[k].resize(nDofx1);
        }
               
        // local storage of global mapping and solution
        for(unsigned i = 0; i < nDof1; i++) {
          l2GMap1[i] = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
        }

        // local storage of coordinates
        for(unsigned i = 0; i < nDofx1; i++) {
          unsigned xDof  = msh->GetSolutionDof(i, iel, xType);    // global to global mapping between coordinates node and coordinate dof
          for(unsigned k = 0; k < dim; k++) {
            x1[k][i] = (*msh->_topology->_Sol[k])(xDof);  // global extraction and local storage for the element coordinates
          }
        }

	if(iel == jel) MMlocal.assign(nDof1 * nDof1, 0.);  //resize
	CClocal.assign(nDof1 * nDof2, 0.);   //resize
        
	// *** Gauss point loop ***
	unsigned igNumber = msh->_finiteElement[ielGeom1][solType]->GetGaussPointNumber();
	double weight1;
        vector <double> phi1;  // local test function
	for(unsigned ig = 0; ig < igNumber; ig++) {

          msh->_finiteElement[ielGeom1][solType]->Jacobian(x1, ig, weight1, phi1, phi_x, *nullDoublePointer);

          // evaluate the solution, the solution derivatives and the coordinates in the gauss point
          vector < double > xg1(dim,0.);
          for(unsigned i = 0; i < nDof1; i++) {
            for(unsigned k = 0; k < dim; k++) {
              xg1[k] += x1[k][i] * phi1[i];
            }
          }

          if(iel == jel) { 
	    for(unsigned i = 0; i < nDof1; i++) {
	      for(unsigned i1 = 0; i1 < nDof1; i1++) {
		MMlocal[ i * nDof1 + i1 ] += phi1[i] * phi1[i1] * weight1;
	      }
	    }
	  }
	        
	  for(unsigned jg = 0; jg < jgNumber; jg++) {
	    double dist = 0.;
	    for(unsigned k = 0; k < dim; k++) {
	      dist += fabs(xg1[k] - xg2[jg][k]);
	    }
	    double C = varianceInput * exp(- dist / L);
	    for(unsigned i = 0; i < nDof1; i++) {
	      for(unsigned j = 0; j < nDof2; j++) {
		CClocal[i * nDof2 + j] += weight1 * phi1[i] * C * phi2[jg][j] * weight2[jg];
	      }//endl j loop
	    } //endl i loop
	  } //endl jg loop
	} //endl ig loop
	if(iel == jel) MM->add_matrix_blocked(MMlocal, l2GMap1, l2GMap1);
	CC->add_matrix_blocked(CClocal, l2GMap1, l2GMap2);
      } // end iel loop
    } //end jel loop
  } //end kproc loop

  MM->close();
  CC->close();

  //BEGIN solve the eigenvalue problem

  int ierr;
  EPS eps;
  PetscInt convergedSolns, numberOfIterations;

  ierr = EPSCreate(PETSC_COMM_WORLD, &eps);
  CHKERRABORT(MPI_COMM_WORLD, ierr);
  ierr = EPSSetOperators(eps, (static_cast<PetscMatrix*>(CC))->mat(), (static_cast<PetscMatrix*>(MM))->mat());
  CHKERRABORT(MPI_COMM_WORLD, ierr);
  ierr = EPSSetFromOptions(eps);
  CHKERRABORT(MPI_COMM_WORLD, ierr);

  //ierr = EPSSetDimensions(eps, numberOfEigPairs, 8 * numberOfEigPairs, 600);
  ierr = EPSSetDimensions(eps, numberOfEigPairs, PETSC_DEFAULT, PETSC_DEFAULT);
  CHKERRABORT(MPI_COMM_WORLD, ierr);
  ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE);
  CHKERRABORT(MPI_COMM_WORLD, ierr);

  ierr = EPSSolve(eps);
  CHKERRABORT(MPI_COMM_WORLD, ierr);

  ierr = EPSView(eps, PETSC_VIEWER_STDOUT_SELF);

  std::cout << " -----------------------------------------------------------------" << std::endl;

  ierr = EPSGetConverged(eps, &convergedSolns);
  CHKERRABORT(MPI_COMM_WORLD, ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, " Number of converged eigenpairs: %D\n\n", convergedSolns);
  CHKERRABORT(MPI_COMM_WORLD, ierr);

  if(convergedSolns > 0) {

    for(unsigned i = 0; i < numberOfEigPairs; i++) {

      char name[10];
      sprintf(name, "egnf%d", i);
      soluIndex = mlSol->GetIndex(name);    // get the position of "u" in the ml_sol object

      // Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and ki (imaginary part)

      ierr = EPSGetEigenpair(eps, i, &eigenvalues[i].first, &eigenvalues[i].second, (static_cast<PetscVector*>(sol->_Sol[soluIndex]))->vec(), NULL);
      CHKERRABORT(MPI_COMM_WORLD, ierr);

    }
  }

  ierr = EPSDestroy(&eps);
  CHKERRABORT(MPI_COMM_WORLD, ierr);

  delete CC;

  // ***************** END ASSEMBLY *******************
}


void GetQuantityOfInterest(MultiLevelProblem& ml_prob, std::vector < double >&  QoI, const unsigned& m, const double& domainMeasure)
{

  //  extract pointers to the several objects that we are going to use

  LinearImplicitSystem* mlPdeSys  = &ml_prob.get_system<LinearImplicitSystem> ("UQ");   // pointer to the linear implicit system named "Poisson"
  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*                    msh = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object
  elem*                     el = msh->el;  // pointer to the elem object in msh (level)

  MultiLevelSolution*    mlSol = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*                sol = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object

  LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27

  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)
  unsigned    nprocs = msh->n_processors(); // get the process_id (for parallel computation)

  //solution variable
  unsigned soluIndex;
  soluIndex = mlSol->GetIndex("u");    // get the position of "u" in the ml_sol object
  unsigned soluType = mlSol->GetSolutionType(soluIndex);    // get the finite element type for "u"

  unsigned soluPdeIndex;
  soluPdeIndex = mlPdeSys->GetSolPdeIndex("u");    // get the position of "u" in the pdeSys object

  vector < double >  solu; // local solution
  solu.reserve(maxSize);

  vector < vector < double > > x(dim);    // local coordinates
  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  for(unsigned i = 0; i < dim; i++) {
    x[i].reserve(maxSize);
  }

  vector <double> phi;  // local test function
  vector <double> phi_x; // local test function first order partial derivatives
  double weight; // gauss point weight

  phi.reserve(maxSize);
  phi_x.reserve(maxSize * dim);

  double quantityOfInterest = 0.;

  // element loop: each process loops only on the elements that owns

  for(int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) {

    short unsigned ielGeom = msh->GetElementType(iel);
    unsigned nDofu  = msh->GetElementDofNumber(iel, soluType);    // number of solution element dofs
    unsigned nDofx = msh->GetElementDofNumber(iel, xType);    // number of coordinate element dofs

    // resize local arrays
    solu.resize(nDofu);

    for(int i = 0; i < dim; i++) {
      x[i].resize(nDofx);
    }

    // local storage of global mapping and solution
    for(unsigned i = 0; i < nDofu; i++) {
      unsigned solDof = msh->GetSolutionDof(i, iel, soluType);    // global to global mapping between solution node and solution dof
      solu[i] = (*sol->_Sol[soluIndex])(solDof);      // global extraction and local storage for the solution
    }

    // local storage of coordinates
    for(unsigned i = 0; i < nDofx; i++) {
      unsigned xDof  = msh->GetSolutionDof(i, iel, xType);    // global to global mapping between coordinates node and coordinate dof

      for(unsigned jdim = 0; jdim < dim; jdim++) {
        x[jdim][i] = (*msh->_topology->_Sol[jdim])(xDof);      // global extraction and local storage for the element coordinates
      }
    }

    vector < double >* nullDoublePointer = NULL;
    // *** Gauss point loop ***
    for(unsigned ig = 0; ig < msh->_finiteElement[ielGeom][soluType]->GetGaussPointNumber(); ig++) {
      // *** get gauss point weight, test function and test function partial derivatives ***
      msh->_finiteElement[ielGeom][soluType]->Jacobian(x, ig, weight, phi, phi_x, *nullDoublePointer);

      double solu_gss = 0.;
      for(unsigned i = 0; i < nDofu; i++) {
        solu_gss += phi[i] * solu[i];
      }
      quantityOfInterest += solu_gss * weight / domainMeasure;


    } // end gauss point loop

  } //end element loop for each process

  QoI[m] = 0.;
  MPI_Allreduce(&quantityOfInterest, &QoI[m], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  //QoI[m] = quantityOfInterest;

//   // add the quantityOfInterest of all processes
//   NumericVector* norm_vec;
//   norm_vec = NumericVector::build().release();
//   norm_vec->init (msh->n_processors(), 1 , false, AUTOMATIC);
//
//   norm_vec->set (iproc, quantityOfInterest);
//   norm_vec->close();
//
//   quantityOfInterest = norm_vec->l1_norm();
//
//   delete norm_vec;
//
//   QoI[m] = quantityOfInterest;

}

void GetStochasticData(std::vector <double>& QoI)
{

  //let's standardize the quantity of interest after finding moments and standard deviation

  if(totMoments <= 0) {

    std::cout << "ERROR: total number of moments has to be a positive integer" << std::endl;

  }

  else {

    //BEGIN computation of the mean of QoI
    meanQoI = 0.;
    for(unsigned m = 0; m < M; m++) {
      meanQoI += QoI[m];
    }
    meanQoI /= M;
    //END


    //BEGIN computation of the variance and standard deviation of QoI
    varianceQoI = 0;
    for(unsigned m = 0; m < M; m++) {
      varianceQoI += pow(QoI[m] - meanQoI, 2);
    }
    varianceQoI /= M;

    stdDeviationQoI = sqrt(varianceQoI);
    //END


    //BEGIN standardization of QoI before computing the moments
//     for(unsigned m = 0; m < M; m++) {
//       QoI[m] = (QoI[m] - meanQoI) / stdDeviationQoI ;
//     }
    //END


    //BEGIN computation of the raw moments
    for(unsigned p = 0; p < totMoments; p++) {
      moments[p] = 0.;
      for(unsigned m = 0; m < M; m++) {
        moments[p] += pow(QoI[m], p + 1);
      }
      moments[p] /= M;
    }
    //END

    cumulants[0] = moments[0];

    if(totMoments > 1) {
      cumulants[1] = moments[1] - moments[0] * moments[0];
      if(totMoments > 2) {
        cumulants[2] = moments[2] - 3. * moments[1] * moments[0] + 2. * pow(moments[0], 3);
        if(totMoments > 3) {
          cumulants[3] = moments[3] - 4. * moments[2] * moments[0] - 3. * moments[1] * moments[1] + 12. * moments[1] * moments[0] * moments[0] - 6. * pow(moments[0], 4);
          if(totMoments > 4) {
            cumulants[4] = moments[4] - 5. * moments[3] * moments[0] - 10. * moments[2] * moments[1] + 20. * moments[2] * moments[0] * moments[0]
                           + 30. * moments[1] * moments[1] * moments[0] - 60. * moments[1] * pow(moments[0], 3) + 24. * pow(moments[0], 5);
            if(totMoments > 5) {
              cumulants[5] = moments[5] - 6. * moments[4] * moments[0] - 15. * moments[3] * moments[1] + 30. * moments[3] * moments[0] * moments[0]
                             - 10. * moments[2] * moments[2] + 120. * moments[2] * moments[1] * moments[0] - 120. * moments[2] * pow(moments[0], 3)
                             + 30. * pow(moments[1], 3) - 270. * pow(moments[1], 2) * pow(moments[0], 2) + 360. * moments[1] * pow(moments[0], 4) - 120. * pow(moments[0], 6);
            }
          }
        }
      }
    }
  }
}


void PlotStochasticData()
{

  std::cout.precision(14);
  std::cout << " the number of MC samples is " << M << std::endl;
  std::cout << " the mean is " << meanQoI << std::endl;
  std::cout << " the standard deviation is " << sqrt(varianceQoI) << std::endl;

  for(unsigned p = 0; p < totMoments; p++) {
//     printf("%d-th moment is %g\n", p + 1, moments[p]);
    std::cout << "the " << p + 1 << "-th moment is " << moments[p] << std::endl;
  }

  for(unsigned p = 0; p < totMoments; p++) {
//     printf("%d-th cumulant is %g\n", p + 1, cumulants[p]);
    std::cout << "the " << p + 1 << "-th cumulant is " << cumulants[p] << std::endl;
  }

  double gramCharlier2Terms = 0.;
  double gramCharlier3Terms = 0.;
  double gramCharlier4Terms = 0.;
  double gramCharlier5Terms = 0.;

  double edgeworth2Terms = 0.;
  double edgeworth3Terms = 0.;
  double edgeworth4Terms = 0.;
  double edgeworth5Terms = 0.;

  double lambda3 = 0.;
  double lambda4 = 0.;
  double lambda5 = 0.;
  double lambda6 = 0.;

  double t = meanQoI - stdDeviationQoI * 40000.;
  double dt = (80000. * stdDeviationQoI) / 300.;

  for(unsigned i = 0; i <= 300; i++) {
    std::cout << t << " ";
    double gaussian = 1. / (sqrt(2 * acos(- 1))) * exp(- 0.5 * (t * t)) ;
    std::cout << gaussian << " ";
    if(totMoments > 1) {

      double d3gaussian = (- 1.) * gaussian * (t * t * t - 3. * t) ;

      gramCharlier2Terms =  gaussian - cumulants[2] / 6. * d3gaussian ;

//       std::cout << -1. / 6 * d3gaussian << " ";

//       std::cout << gramCharlier2Terms << " ";

      lambda3 = cumulants[2] / pow(stdDeviationQoI, 3);

      edgeworth2Terms = gaussian - 1. / sqrt(M) * lambda3 * d3gaussian / 6.;

      std::cout << edgeworth2Terms << " ";

      if(totMoments > 2) {

        double d4gaussian = (1.) * gaussian * (t * t * t * t - 6. * t * t + 3.) ;
        double d6gaussian = (1.) * gaussian * (pow(t, 6) - 15 * t * t * t * t + 45 * t * t - 15);

        gramCharlier3Terms = gramCharlier2Terms + cumulants[3] / 24. * d4gaussian ;

//         std::cout << 1. / 24. * d4gaussian << " ";

//         std::cout << gramCharlier3Terms << " ";

        lambda4 = cumulants[3] / pow(stdDeviationQoI, 4);

        edgeworth3Terms = edgeworth2Terms + 1. / M * (1. / 24 * lambda4 * d4gaussian + 1. / 72 * lambda3 * lambda3 * d6gaussian);

        std::cout << edgeworth3Terms << " ";

        if(totMoments > 3) {

          double d5gaussian = (- 1.) * gaussian * (pow(t, 5) - 10. * t * t * t + 15. * t);
          double d7gaussian = (- 1.) * gaussian * (pow(t, 7) - 21. * pow(t, 5) + 105. * t * t * t -  105. * t) ;
          double d9gaussian = (- 1.) * gaussian * (pow(t, 9) - 36. * pow(t, 7) + 378. * pow(t, 5) - 1260. * t * t * t + 945. * t) ;

          gramCharlier4Terms = gramCharlier3Terms - cumulants[4] / 120. * d5gaussian;

//           std::cout << - 1. / 120. * d5gaussian << " ";

//           std::cout << gramCharlier4Terms << " " ;

          lambda5 = cumulants[4] / pow(stdDeviationQoI, 5);

          edgeworth4Terms = edgeworth3Terms - 1. / (pow(M, 1.5)) * (lambda5 * d5gaussian / 120. + lambda3 * lambda4 * d7gaussian / 144. + pow(lambda3, 3) * d9gaussian / 1296.) ;

          std::cout << edgeworth4Terms << "\n";

          if(totMoments > 4) {

            double d6gaussian = (1.) * gaussian * (pow(t, 6) - 15. * pow(t, 4) + 45. * t * t - 15.) ;

            gramCharlier5Terms = gramCharlier4Terms + (10. * cumulants[2] * cumulants[2] + cumulants[5]) / (120. * 6.) * d6gaussian;

// 	    std::cout << gramCharlier5Terms << "\n";

          }
        }
      }
    }

    t += dt;
  }

}





