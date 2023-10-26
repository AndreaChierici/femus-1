/** tutorial/Ex3
 * This example shows how to set and solve the weak form of the nonlinear problem
 *                     -\Delta^2 u = f(x) \text{ on }\Omega,
 *            u=0 \text{ on } \Gamma,
 *      \Delta u=0 \text{ on } \Gamma,
 * on a box domain $\Omega$ with boundary $\Gamma$,
 * by using a system of second order partial differential equation.
 * all the coarse-level meshes are removed;
 * a multilevel problem and an equation system are initialized;
 * a direct solver is used to solve the problem.
 **/

#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "VTKWriter.hpp"
#include "NumericVector.hpp"

#include "FE_convergence.hpp"

#include "adept.h"  //Need for Automatic Differentiation

#include "cmath"

using namespace femus;


/*
//ORGINAL -BEGIN
double GetExactSolutionValue(const std::vector < double >& x) {
  double pi = acos(-1.);
  return cos(pi * x[0]) * cos(pi * x[1]);
};


void GetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
  double pi = acos(-1.);
  solGrad[0]  = -pi * sin(pi * x[0]) * cos(pi * x[1]);
  solGrad[1] = -pi * cos(pi * x[0]) * sin(pi * x[1]);
};


double GetExactSolutionLaplace(const std::vector < double >& x) {
  double pi = acos(-1.);
  return -2.*pi * pi * cos(pi * x[0]) * cos(pi * x[1]);       // - pi*pi*cos(pi*x[0])*cos(pi*x[1]);
};



// for v - BEGIN ----
double LaplaceGetExactSolutionValue(const std::vector < double >& x) {
  double pi = acos(-1.);
  return -2.* pi * pi * cos(pi * x[0]) * cos(pi * x[1]);       // - pi*pi*cos(pi*x[0])*cos(pi*x[1]);
};

void LaplaceGetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
  double pi = acos(-1.);
  solGrad[0]  = 2. * pi * pi * pi * sin(pi * x[0]) * cos(pi * x[1]);
  solGrad[1] =  2. * pi * pi * pi * cos(pi * x[0]) * sin(pi * x[1]);
};
// for v - END ----

//ORGINAL-END*/

/*
//POLYNOMIAL-BEGIN

double GetExactSolutionValue(const std::vector < double >& x) {
  double pi = acos(-1.);
  return x[0] * powf((1-x[0]),4) + x[1] * pow((1-x[1]),4) ;
};


void GetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
  double pi = acos(-1.);
  solGrad[0]  = powf((1-x[0]),3) * (1 - 5*x[0]);
  solGrad[1] = powf((1-x[1]),3) * (1 - 5*x[1]);
};


double GetExactSolutionLaplace(const std::vector < double >& x) {
  double pi = acos(-1.);
  return 6 * (1 - x[0]) * (1 - pow(x[0],2)) + 6 * (1 - x[1]) * (1 - pow(x[1],2));       // - pi*pi*cos(pi*x[0])*cos(pi*x[1]);
};



// for v - BEGIN ----
double LaplaceGetExactSolutionValue(const std::vector < double >& x) {
  double pi = acos(-1.);
  return 6 * (1 - x[0]) * (1 - pow(x[0],2)) + 6 * (1 - x[1]) * (1 - pow(x[1],2));       // - pi*pi*cos(pi*x[0])*cos(pi*x[1]);
};

void LaplaceGetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
  double pi = acos(-1.);
  solGrad[0]  = 36 - 72 * (1 - x[0]) * (1 + 3*x[0]) * (1 - x[1]) * (1 - powf(x[1],2));
  solGrad[1] =  36 - 72 * (1 - x[1]) * (1 + 3*x[1]) * (1 - x[0]) * (1 - powf(x[0],2));

// for RHS- BEGIN
double LaplaceGetExactSolution_RHS(const std::vector < double >& x) {
    return 36 * x[0] + 36 * x[1] - 24;
// for RHS-END


};
// for v - END ----

//POLYNOMIAL-END*/


/*
//SIN -BEGIN
double GetExactSolutionValue(const std::vector < double >& x) {
  double pi = acos(-1.);
  return sin(2*pi * x[0]) * sin(2*pi * x[1]);
};


void GetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
  double pi = acos(-1.);
  solGrad[0]  = -2*pi * sin(2*pi * x[1]) * cos(2*pi * x[0]);
  solGrad[1] = -2*pi * cos(2*pi * x[1]) * sin(2*pi * x[0]);
};


double GetExactSolutionLaplace(const std::vector < double >& x) {
  double pi = acos(-1.);
  return -8.*pi * pi * sin(2*pi * x[0]) * sin(2*pi * x[1]);       // - pi*pi*cos(pi*x[0])*cos(pi*x[1]);
};



// for v - BEGIN ----
double LaplaceGetExactSolutionValue(const std::vector < double >& x) {
  double pi = acos(-1.);
  return -8.* pi * pi * sin(2*pi * x[0]) * sin(2*pi * x[1]);       // - pi*pi*cos(pi*x[0])*cos(pi*x[1]);
};

void LaplaceGetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
  double pi = acos(-1.);
  solGrad[0]  = -16. * pi * pi * pi * sin(2*pi * x[1]) * cos(2*pi * x[0]);
  solGrad[1] =  -16. * pi * pi * pi * cos(2*pi * x[1]) * sin(2*pi * x[0]);
};
// for v - END ----

//SIN-END*/



//1_D_PROBLEM -BEGIN
double GetExactSolutionValue(const std::vector < double >& x) {
  double pi = acos(-1.);
  return sin(2*pi * x[0]);
};


void GetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
  double pi = acos(-1.);
  solGrad[0]  = 2*pi * cos(2*pi * x[0]);
  //solGrad[1] = -pi * cos(pi * x[0]) * sin(pi * x[1]);
};


double GetExactSolutionLaplace(const std::vector < double >& x) {
  double pi = acos(-1.);
  return -4*pi * pi * sin(2*pi * x[0]) ;       // - pi*pi*cos(pi*x[0])*cos(pi*x[1]);
};



// for v - BEGIN ----
double LaplaceGetExactSolutionValue(const std::vector < double >& x) {
  double pi = acos(-1.);
  return -4*pi * pi * sin(2*pi * x[0]);       // - pi*pi*cos(pi*x[0])*cos(pi*x[1]);
};

void LaplaceGetExactSolutionGradient(const std::vector < double >& x, vector < double >& solGrad) {
  double pi = acos(-1.);
  solGrad[0]  =  -8* pi * pi * pi * cos(2*pi * x[0]);
  //solGrad[1] =  2. * pi * pi * pi * cos(pi * x[0]) * sin(pi * x[1]);
};
// for v - END ----

//1_D_PROBLEM-END




//ORGINAL_BC-BEGIN
bool SetBoundaryCondition(const std::vector < double >& x, const char SolName[], double& value, const int facename, const double time) {
  bool dirichlet = true; //dirichlet
  value = 0;
  return dirichlet;
}
//ORGINAL_BC-END


/*
//BC_SIMILAR_NS-BEGIN

bool SetBoundaryCondition(const std::vector < double >& x, const char SolName[], double& value, const int facename, const double time) {
  bool dirichlet = true; //dirichlet

  if (!strcmp(SolName, "U")) { // strcmp compares two string in lexiographic sense.
    value = 0.;
    if (facename == 1) {
      if (x[1] < 0.5 && x[1] > -0.5 && x[2] < 0.5 && x[2] > -0.5) value = 1.;
    }
  }
  else if (!strcmp(SolName, "V")) {
    value = 0.;
    //if (facename == 1) {
     // if (x[1] < 0.5 && x[1] > -0.5 && x[2] < 0.5 && x[2] > -0.5) value = 1.;
    //}
  }

  return dirichlet;
}

//BC_SIMILAR_NS-END*/



void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob);




int main(int argc, char** args) {

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);


  // define multilevel mesh
  MultiLevelMesh mlMsh;
  // read coarse level mesh and generate finers level meshes
  double scalingFactor = 1.;
  const std::string relative_path_to_build_directory =  "../../../../"; //Mention the level of directory

  //const std::string mesh_file = relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/02_2d/square/minus0p5-plus0p5_minus0p5-plus0p5/square_-0p5-0p5x-0p5-0p5_divisions_2x2.med"; //Orginal

  const std::string mesh_file = relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/01_1d/zzz_embedded_in_3d/segment/0-1/Mesh_1_y_all_dir.med";

  mlMsh.ReadCoarseMesh(mesh_file.c_str(), "seventh", scalingFactor);

  unsigned maxNumberOfMeshes = 5;

  vector < vector < double > > l2Norm;
  l2Norm.resize(maxNumberOfMeshes);

  vector < vector < double > > semiNorm;
  semiNorm.resize(maxNumberOfMeshes);

    std::vector<FEOrder> feOrder;  
    feOrder.push_back(FIRST);
    feOrder.push_back(SERENDIPITY);
    feOrder.push_back(SECOND);
    
    
  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {   // loop on the mesh level

    unsigned numberOfUniformLevels = i + 1;
    unsigned numberOfSelectiveLevels = 1;
    mlMsh.RefineMesh(numberOfUniformLevels , numberOfUniformLevels + numberOfSelectiveLevels, NULL);

    // erase all the coarse mesh levels
    mlMsh.EraseCoarseLevels(numberOfUniformLevels - 1);

    // print mesh info
    mlMsh.PrintInfo();

    l2Norm[i].resize( feOrder.size() );
    semiNorm[i].resize( feOrder.size() );

    for (unsigned j = 0; j < feOrder.size(); j++) {   // loop on the FE Order

      // define the multilevel solution and attach the mlMsh object to it
      MultiLevelSolution mlSol(&mlMsh);

      // add variables to mlSol
      mlSol.AddSolution("u", LAGRANGE, feOrder[j]);
      mlSol.AddSolution("v", LAGRANGE, feOrder[j]);
      mlSol.Initialize("All");

      // attach the boundary condition function and generate boundary data
      mlSol.AttachSetBoundaryConditionFunction(SetBoundaryCondition);
      mlSol.GenerateBdc("u");
      mlSol.GenerateBdc("v");

      // define the multilevel problem attach the mlSol object to it
      MultiLevelProblem mlProb(&mlSol);

      // add system Biharmonic in mlProb as a Linear Implicit System
      NonLinearImplicitSystem& system = mlProb.add_system < NonLinearImplicitSystem > ("Biharmonic");

      // add solution "u" to system
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("v");

      // attach the assembling function to system
      system.SetAssembleFunction(AssembleBilaplaceProblem_AD);

      // initialize and solve the system
      system.init();
      
      system.MGsolve();

      // convergence for u
      std::pair< double , double > norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "u", GetExactSolutionValue, GetExactSolutionGradient );
      // // convergence for v
      //std::pair< double , double > norm = GetErrorNorm_L2_H1_with_analytical_sol(&mlSol, "v", LaplaceGetExactSolutionValue, LaplaceGetExactSolutionGradient );
      
      
      l2Norm[i][j]  = norm.first;
      semiNorm[i][j] = norm.second;

      
      // print solutions
      std::vector < std::string > variablesToBePrinted;
      variablesToBePrinted.push_back("All");

      VTKWriter vtkIO(&mlSol);
      vtkIO.Write(DEFAULT_OUTPUTDIR, "biquadratic", variablesToBePrinted, i);

    }
  }

  // ======= L2 - BEGIN  ========================
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "l2 ERROR and ORDER OF CONVERGENCE:\n\n";
  std::cout << "LEVEL\tFIRST\t\t\tSERENDIPITY\t\tSECOND\n";

  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    std::cout << i + 1 << "\t";
    std::cout.precision(14);

    for (unsigned j = 0; j < feOrder.size(); j++) {
      std::cout << l2Norm[i][j] << "\t";
    }

    std::cout << std::endl;

    if (i < maxNumberOfMeshes - 1) {
      std::cout.precision(3);
      std::cout << "\t\t";

      for (unsigned j = 0; j < feOrder.size(); j++) {
        std::cout << log(l2Norm[i][j] / l2Norm[i + 1][j]) / log(2.) << "\t\t\t";
      }

      std::cout << std::endl;
    }

  }
  // ======= L2 - END  ========================

  
  
  // ======= H1 - BEGIN  ========================
  
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "SEMINORM ERROR and ORDER OF CONVERGENCE:\n\n";
  std::cout << "LEVEL\tFIRST\t\t\tSERENDIPITY\t\tSECOND\n";

  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    std::cout << i + 1 << "\t";
    std::cout.precision(14);

    for (unsigned j = 0; j < feOrder.size(); j++) {
      std::cout << semiNorm[i][j] << "\t";
    }

    std::cout << std::endl;

    if (i < maxNumberOfMeshes - 1) {
      std::cout.precision(3);
      std::cout << "\t\t";

      for (unsigned j = 0; j < feOrder.size(); j++) {
        std::cout << log(semiNorm[i][j] / semiNorm[i + 1][j]) / log(2.) << "\t\t\t";
      }

      std::cout << std::endl;
    }

  }

  // ======= H1 - END  ========================


  return 0;
}







/**
 * Given the non linear problem
 *
 *      \Delta^2 u  = f(x),
 *      u(\Gamma) = 0
 *      \Delta u(\Gamma) = 0
 *
 * in the unit box \Omega centered in the origin with boundary \Gamma, where
 *
 *                      f(x) = \Delta^2 u_e ,
 *                    u_e = \cos ( \pi * x ) * \cos( \pi * y ),
 *
 * the following function assembles the system:
 *
 *      \Delta u = v
 *      \Delta v = f(x) = 4. \pi^4 u_e
 *      u(\Gamma) = 0
 *      v(\Gamma) = 0
 *
 * using automatic differentiation
 **/


//AUTOMATIC_DIFFERENTIATION-BEGIN
void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled

  // call the adept stack object
  adept::Stack& s = FemusInit::_adeptStack;

  //  extract pointers to the several objects that we are going to use

  NonLinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<NonLinearImplicitSystem> ("Biharmonic");   // pointer to the linear implicit system named "Biharmonic"

  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*          msh          = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object
  elem*          el         = msh->el;  // pointer to the elem object in msh (level)

  MultiLevelSolution*  mlSol        = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*    sol        = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object

  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  SparseMatrix*    KK         = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*   RES          = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)

  //solution variable
  unsigned soluIndex = mlSol->GetIndex("u");    // get the position of "u" in the ml_sol object
  unsigned soluType = mlSol->GetSolutionType(soluIndex);    // get the finite element type for "u"
  unsigned soluPdeIndex = mlPdeSys->GetSolPdeIndex("u");    // get the position of "u" in the pdeSys object

  //adouble is a type inside adept library correpsonding to the double type. Need to be defined inside the library.
  vector < adept::adouble >  solu; // local solution

  unsigned solvIndex = mlSol->GetIndex("v");    // get the position of "v" in the ml_sol object
  unsigned solvType = mlSol->GetSolutionType(solvIndex);    // get the finite element type for "v"
  unsigned solvPdeIndex = mlPdeSys->GetSolPdeIndex("v");    // get the position of "v" in the pdeSys object

  vector < adept::adouble >  solv; // local solution


  vector < vector < double > > x(dim);    // local coordinates
  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  vector< int > sysDof; // local to global pdeSys dofs
  vector <double> phi;  // local test function
  vector <double> phi_x; // local test function first order partial derivatives
  vector <double> phi_xx; // local test function second order partial derivatives
  double weight; // gauss point weight

  vector< double > Res; // local redidual vector
  vector< adept::adouble > aResu; // local redidual vector
  vector< adept::adouble > aResv; // local redidual vector


  // reserve memory for the local standar vectors
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27
  solu.reserve(maxSize);
  solv.reserve(maxSize);

  for (unsigned i = 0; i < dim; i++)
    x[i].reserve(maxSize);

  sysDof.reserve(2 * maxSize);
  phi.reserve(maxSize);
  phi_x.reserve(maxSize * dim);
  unsigned dim2 = (3 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)
  phi_xx.reserve(maxSize * dim2);

  Res.reserve(2 * maxSize);
  aResu.reserve(maxSize);
  aResv.reserve(maxSize);

  vector < double > Jac; // local Jacobian matrix (ordered by column, adept)
  Jac.reserve(4 * maxSize * maxSize);


  KK->zero(); // Set to zero all the entries of the Global Matrix


  for (int iel = msh->_elementOffset[iproc]; iel < msh->_elementOffset[iproc + 1]; iel++) { //Loop for each of the element and computing the Jacobian matrix using "adept"

    short unsigned ielGeom = msh->GetElementType(iel); 
    unsigned nDofs  = msh->GetElementDofNumber(iel, soluType);    // number of solution element dofs
    unsigned nDofs2 = msh->GetElementDofNumber(iel, xType);    // number of coordinate element dofs
    
    // resize local arrays
    sysDof.resize(2 * nDofs);
    solu.resize(nDofs);
    solv.resize(nDofs);

    for (int i = 0; i < dim; i++) {
      x[i].resize(nDofs2);
    }

    aResu.assign(nDofs, 0.);    //resize
    aResv.assign(nDofs, 0.);    //resize

    // local storage of global mapping and solution
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned solDof = msh->GetSolutionDof(i, iel, soluType);    // global to global mapping between solution node and solution dof
      solu[i]          = (*sol->_Sol[soluIndex])(solDof);      // global extraction and local storage for the solution
      solv[i]          = (*sol->_Sol[solvIndex])(solDof);      // global extraction and local storage for the solution
      sysDof[i]         = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
      sysDof[nDofs + i] = pdeSys->GetSystemDof(solvIndex, solvPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
    }

    // local storage of coordinates
    for (unsigned i = 0; i < nDofs2; i++) {
      unsigned xDof  = msh->GetSolutionDof(i, iel, xType); // global to global mapping between coordinates node and coordinate dof

      for (unsigned jdim = 0; jdim < dim; jdim++) {
        x[jdim][i] = (*msh->_topology->_Sol[jdim])(xDof);  // global extraction and local storage for the element coordinates
      }
    }

    // start a new recording of all the operations involving adept::adouble variables
    s.new_recording();

    // *** Gauss point loop ***
    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][soluType]->GetGaussPointNumber(); ig++) {
      // *** get gauss point weight, test function and test function partial derivatives ***
      msh->_finiteElement[ielGeom][soluType]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);

      // evaluate the solution, the solution derivatives and the coordinates in the gauss point
      adept::adouble soluGauss = 0;
      vector < adept::adouble > soluGauss_x(dim, 0.);

      adept::adouble solvGauss = 0;
      vector < adept::adouble > solvGauss_x(dim, 0.);

      vector < double > xGauss(dim, 0.);

      for (unsigned i = 0; i < nDofs; i++) {
        soluGauss += phi[i] * solu[i];
        solvGauss += phi[i] * solv[i];

        for (unsigned jdim = 0; jdim < dim; jdim++) {
          soluGauss_x[jdim] += phi_x[i * dim + jdim] * solu[i];
          solvGauss_x[jdim] += phi_x[i * dim + jdim] * solv[i];
          xGauss[jdim] += x[jdim][i] * phi[i];
        }
      }

      // *** phi_i loop ***
      for (unsigned i = 0; i < nDofs; i++) {

        adept::adouble Laplace_u = 0.;
        adept::adouble Laplace_v = 0.;

        for (unsigned jdim = 0; jdim < dim; jdim++) {
          Laplace_u   +=  - phi_x[i * dim + jdim] * soluGauss_x[jdim];
          Laplace_v   +=  - phi_x[i * dim + jdim] * solvGauss_x[jdim];
        }

        double exactSolValue = GetExactSolutionValue(xGauss);

        //RHS part of the Biharmonic Equation
        double pi = acos(-1.);
        aResv[i] += (solvGauss * phi[i] -  Laplace_u) * weight;
        //aResu[i] += ( (36*x[0] + 36*x[1] - 24) * phi[i] -  Laplace_v) * weight;
        aResu[i] += (16.*pi * pi * pi * pi * exactSolValue * phi[i] -  Laplace_v) * weight; //Orginal
      } // end phi_i loop
    } // end gauss point loop


    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store
    Res.resize(2 * nDofs);

    for (int i = 0; i < nDofs; i++) {
      Res[i]         = -aResu[i].value();
      Res[nDofs + i] = -aResv[i].value();
    }

    RES->add_vector_blocked(Res, sysDof);

    Jac.resize( 4 * nDofs * nDofs );

    // define the dependent variables
    s.dependent(&aResu[0], nDofs);
    s.dependent(&aResv[0], nDofs);

    // define the independent variables
    s.independent(&solu[0], nDofs);
    s.independent(&solv[0], nDofs);

    // get the jacobian matrix (ordered by column)
    s.jacobian(&Jac[0], true);

    KK->add_matrix_blocked(Jac, sysDof, sysDof);

    s.clear_independents();
    s.clear_dependents();

  } //end element loop for each process

  RES->close();
  KK->close();


}
//AUTOMATIC_DIFFERENTIATION-END
