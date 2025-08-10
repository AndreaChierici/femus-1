#ifndef __femus_biharmonic_coupled_hpp__
#define __femus_biharmonic_coupled_hpp__

#include "FemusInit.hpp"  //for the adept stack

#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"

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

using namespace femus;


namespace karthik {

  class biharmonic_coupled_equation {

  public:



static void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob) {
  //  ml_prob is the global object from/to where get/set all the data
  //  level is the level of the PDE system to be assembled

  // call the adept stack object
  adept::Stack& s = FemusInit::_adeptStack;

  //  extract pointers to the several objects that we are going to use

  NonLinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<NonLinearImplicitSystem> (ml_prob.get_app_specs_pointer()->_system_name);

  const unsigned level = mlPdeSys->GetLevelToAssemble();

  Mesh*          msh          = ml_prob._ml_msh->GetLevel(level);    // pointer to the mesh (level) object
  elem*          el         = msh->GetMeshElements();  // pointer to the elem object in msh (level)

  MultiLevelSolution*  ml_sol        = ml_prob._ml_sol;  // pointer to the multilevel solution object
  Solution*    sol        = ml_prob._ml_sol->GetSolutionLevel(level);    // pointer to the solution (level) object

  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level]; // pointer to the equation (level) object
  SparseMatrix*    KK         = pdeSys->_KK;  // pointer to the global stifness matrix object in pdeSys (level)
  NumericVector*   RES          = pdeSys->_RES; // pointer to the global residual vector object in pdeSys (level)

  const unsigned  dim = msh->GetDimension(); // get the domain dimension of the problem
  unsigned    iproc = msh->processor_id(); // get the process_id (for parallel computation)


  const std::string solname_u = ml_sol->GetSolName_string_vec()[0];

  unsigned soluIndex = ml_sol->GetIndex(solname_u.c_str());    // get the position of "u" in the ml_sol object
  unsigned solFEType_u = ml_sol->GetSolutionType(soluIndex);    // get the finite element type for "u"

  unsigned soluPdeIndex = mlPdeSys->GetSolPdeIndex(solname_u.c_str());    // get the position of "u" in the pdeSys object



  std::vector < adept::adouble >  solu; // local solution


  const std::string solname_sxx = ml_sol->GetSolName_string_vec()[1];

  unsigned solsxxIndex = ml_sol->GetIndex(solname_sxx.c_str());    // get the position of "sxx" in the ml_sol object

  unsigned solFEType_sxx = ml_sol->GetSolutionType(solsxxIndex);    // get the finite element type for "sxx"


  unsigned solsxxPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxx.c_str());    // get the position of "sxx" in the pdeSys object

  std::vector < adept::adouble >  solsxx; // local solution


  // ---------- NEW: sxy, syy ----------
  const std::string solname_sxy = ml_sol->GetSolName_string_vec()[2];
  unsigned solsxyIndex   = ml_sol->GetIndex(solname_sxy.c_str());
  unsigned solFEType_sxy = ml_sol->GetSolutionType(solsxyIndex);



  unsigned solsxyPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxy.c_str());
  std::vector < adept::adouble >  solsxy;

  const std::string solname_syy = ml_sol->GetSolName_string_vec()[3];
  unsigned solsyyIndex   = ml_sol->GetIndex(solname_syy.c_str());
  unsigned solFEType_syy = ml_sol->GetSolutionType(solsyyIndex);



  unsigned solsyyPdeIndex = mlPdeSys->GetSolPdeIndex(solname_syy.c_str());
  std::vector < adept::adouble >  solsyy;
  // -----------------------------------


  std::vector < std::vector < double > > x(dim);    // local coordinates
  unsigned xType = 2; // get the finite element type for "x", it is always 2 (LAGRANGE QUADRATIC)

  std::vector < int > sysDof; // local to global pdeSys dofs
  std::vector <double> phi;  // local test function
  std::vector <double> phi_x; // local test function first order partial derivatives
  std::vector <double> phi_xx; // local test function second order partial derivatives
  double weight; // gauss point weight

  std::vector < double > Res; // local redidual vector
  std::vector < adept::adouble > aResu; // local redidual vector
  std::vector < adept::adouble > aRessxx; // local redidual vector
  // new residuals for sxy,syy
  std::vector < adept::adouble > aRessxy;
  std::vector < adept::adouble > aRessyy;


  // reserve memory for the local standar vectors
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));          // conservative: based on line3, quad9, hex27
  solu.reserve(maxSize);
  solsxx.reserve(maxSize);
  solsxy.reserve(maxSize);
  solsyy.reserve(maxSize);

  for (unsigned i = 0; i < dim; i++){
    x[i].reserve(maxSize);}

  sysDof.reserve(4 * maxSize);
  phi.reserve(maxSize);
  phi_x.reserve(maxSize * dim);
  unsigned dim2 = (6 * (dim - 1) + !(dim - 1));        // dim2 is the number of second order partial derivatives (1,3,6 depending on the dimension)
  phi_xx.reserve(maxSize * dim2);

  Res.reserve(4 * maxSize);
  aResu.reserve(maxSize);
  aRessxx.reserve(maxSize);
  aRessxy.reserve(maxSize);
  aRessyy.reserve(maxSize);

  std::vector < double > Jac; // local Jacobian matrix (ordered by column, adept)
  // reserve enough for a 4x4 block (16 blocks)
  Jac.reserve(256 * maxSize * maxSize);


  KK->zero(); // Set to zero all the entries of the Global Matrix


  for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

    short unsigned ielGeom = msh->GetElementType(iel);

    unsigned nDofs  = msh->GetElementDofNumber(iel, solFEType_u);    // number of solution element dofs


    unsigned nDofs2 = msh->GetElementDofNumber(iel, xType);    // number of coordinate element dofs


    unsigned nDofs_u   = msh->GetElementDofNumber(iel, solFEType_u);
unsigned nDofs_sxx = msh->GetElementDofNumber(iel, solFEType_sxx);
// same for sxy and syy



       std::vector<unsigned> Sol_n_el_dofs_Mat_vol(4, nDofs); // changed to 4 blocks

    // resize local arrays
    sysDof.resize(4 * nDofs);
    solu.resize(nDofs);
    solsxx.resize(nDofs);
    solsxy.resize(nDofs);
    solsyy.resize(nDofs);

    for (int i = 0; i < dim; i++) {
      x[i].resize(nDofs2);
    }

    aResu.assign(nDofs, 0.);    //resize
    aRessxx.assign(nDofs, 0.);    //resize
    aRessxy.assign(nDofs, 0.);   //resize
    aRessyy.assign(nDofs, 0.);   //resize

    // local storage of global mapping and solution
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_u);    // global to global mapping between solution node and solution dof

      solu[i]          = (*sol->_Sol[soluIndex])(solDof);      // global extraction and local storage for the solution
      solsxx[i]          = (*sol->_Sol[solsxxIndex])(solDof);      // global extraction and local storage for the solution

      // sxy and syy use the same mapping of local node to solution dof (assuming they share FE nodes)
      solsxy[i]         = (*sol->_Sol[solsxyIndex])(solDof);
      solsyy[i]         = (*sol->_Sol[solsyyIndex])(solDof);

      sysDof[i]               = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
      sysDof[nDofs + i]       = pdeSys->GetSystemDof(solsxxIndex, solsxxPdeIndex, i, iel);    // global to global mapping between solution node and pdeSys dof
      sysDof[2 * nDofs + i]   = pdeSys->GetSystemDof(solsxyIndex, solsxyPdeIndex, i, iel);  // sxy
      sysDof[3 * nDofs + i]   = pdeSys->GetSystemDof(solsyyIndex, solsyyPdeIndex, i, iel);  // syy
    }

    // local storage of coordinates
    for (unsigned i = 0; i < nDofs2; i++) {
      unsigned xDof  = msh->GetSolutionDof(i, iel, xType); // global to global mapping between coordinates node and coordinate dof

      for (unsigned jdim = 0; jdim < dim; jdim++) {
        x[jdim][i] = (*msh->GetTopology()->_Sol[jdim])(xDof);  // global extraction and local storage for the element coordinates
      }
    }

    // start a new recording of all the operations involving adept::adouble variables
    s.new_recording();

    // *** Gauss point loop ***

    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solFEType_u]->GetGaussPointNumber(); ig++) {
      // *** get gauss point weight, test function and test function partial derivatives ***
      msh->_finiteElement[ielGeom][solFEType_u]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);


      // evaluate the solution, the solution derivatives and the coordinates in the gauss point
      adept::adouble soluGauss = 0;
      std::vector < adept::adouble > soluGauss_x(dim, 0.);

      adept::adouble solsxxGauss = 0;
      std::vector < adept::adouble > solsxxGauss_x(dim, 0.);

      adept::adouble solsxyGauss = 0;
      std::vector < adept::adouble > solsxyGauss_x(dim, 0.);

      adept::adouble solsyyGauss = 0;
      std::vector < adept::adouble > solsyyGauss_x(dim, 0.);

      std::vector < double > xGauss(dim, 0.);

      for (unsigned i = 0; i < nDofs; i++) {
        soluGauss   += phi[i] * solu[i];
        solsxxGauss   += phi[i] * solsxx[i];
        solsxyGauss  += phi[i] * solsxy[i];
        solsyyGauss  += phi[i] * solsyy[i];

        for (unsigned jdim = 0; jdim < dim; jdim++) {
          soluGauss_x[jdim]  += phi_x[i * dim + jdim] * solu[i];
          solsxxGauss_x[jdim]  += phi_x[i * dim + jdim] * solsxx[i];
          solsxyGauss_x[jdim] += phi_x[i * dim + jdim] * solsxy[i];
          solsyyGauss_x[jdim] += phi_x[i * dim + jdim] * solsyy[i];

          xGauss[jdim] += x[jdim][i] * phi[i];
        }
      }
      // *** phi_i loop ***
      for (unsigned i = 0; i < nDofs; i++) {

        adept::adouble Laplace_u = 0.;
        adept::adouble Laplace_sxx = 0.;

        adept::adouble Laplace_sxy = 0.;
        adept::adouble Laplace_syy = 0.;




        for (unsigned jdim = 0; jdim < dim; jdim++) {
          Laplace_u   +=  - phi_x[i * dim + jdim] * soluGauss_x[jdim];
          Laplace_sxx   +=  - phi_x[i * dim + jdim] * solsxxGauss_x[jdim];

          Laplace_sxy  +=  - phi_x[i * dim + jdim] * solsxyGauss_x[jdim];
          Laplace_syy  +=  - phi_x[i * dim + jdim] * solsyyGauss_x[jdim];
        }



    adept::adouble A_Laplace_u = 0.0;
    adept::adouble A_Laplace_sxx = 0.0;
    adept::adouble A_Laplace_sxy = 0.0;
    adept::adouble A_Laplace_syy = 0.0;

    adept::adouble B_sxx = 0.0;
    adept::adouble B_sxy = 0.0;
    adept::adouble B_syy = 0.0;
    adept::adouble B_u_sxx= 0.0;
    adept::adouble B_u_sxy= 0.0;
    adept::adouble B_u_syy= 0.0;



    for (unsigned j = 0; j < nDofs; ++j) {
        adept::adouble dphi_i_dx = phi_x[i * dim + 0];
        adept::adouble dphi_j_dx = phi_x[j * dim + 0];
        adept::adouble dphi_i_dy = phi_x[i * dim + 1];
        adept::adouble dphi_j_dy = phi_x[j * dim + 1];

        A_Laplace_u += - (dphi_i_dx * dphi_j_dx + dphi_i_dy * dphi_j_dy) * solu[j];
        A_Laplace_sxx += - (dphi_i_dx * dphi_j_dx + dphi_i_dy * dphi_j_dy) * solsxx[j];
        A_Laplace_sxy += - (dphi_i_dx * dphi_j_dx + dphi_i_dy * dphi_j_dy) * solsxy[j];
        A_Laplace_syy += - (dphi_i_dx * dphi_j_dx + dphi_i_dy * dphi_j_dy) * solsyy[j];


        B_sxx +=  dphi_i_dx * dphi_j_dx * solsxx[j];
        B_sxy +=   ( dphi_i_dy * dphi_j_dx  + dphi_i_dx * dphi_j_dy ) * solsxy[j];
// // //         B_sxy +=  2. * ( dphi_i_dy * dphi_j_dx ) * solsxy[j];

        B_syy +=  dphi_i_dy * dphi_j_dy * solsyy[j];


        B_u_sxx +=  dphi_i_dx * dphi_j_dx * solu[j];
        B_u_sxy +=  ( dphi_i_dy * dphi_j_dx  + dphi_i_dx * dphi_j_dy ) * solu[j];
// // //         B_u_sxy += 2. *  ( dphi_i_dy * dphi_j_dx  ) * solu[j];

        B_u_syy +=  dphi_i_dy * dphi_j_dy * solu[j];


    }
    /*
    adept::adouble B_sxx = phi_x[i * dim + 0] * solsxxGauss_x[0];
    adept::adouble B_syy = phi_x[i * dim + 1] * solsyyGauss_x[1];
    adept::adouble B_sxy = phi_x[i * dim + 0] * solsxyGauss_x[0]
                    + phi_x[i * dim + 1] * solsxyGauss_x[1];

    adept::adouble B_u_sxx = phi_x[i * dim + 0] * soluGauss_x[0];
    adept::adouble B_u_syy = phi_x[i * dim + 1] * soluGauss_x[1];
    adept::adouble B_u_sxy = phi_x[i * dim + 0] * soluGauss_x[0] + phi_x[i * dim + 1] * soluGauss_x[1];
*/
        adept::adouble F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->laplacian(xGauss) * phi[i];

        aResu[i] += ( B_sxx + B_sxy + B_syy + F_term) * weight;
        aRessxx[i] += (B_u_sxx + solsxxGauss * phi[i] ) * weight;
        aRessxy[i] += (B_u_sxy + 2 * solsxyGauss * phi[i] ) * weight;
        aRessyy[i] += ( B_u_syy + solsyyGauss * phi[i] ) * weight;


/*
        aResu[i] += (F_term - A_Laplace_sxx) * weight;
        aRessxx[i] += (solsxxGauss * phi[i] -  A_Laplace_u ) * weight;
        aRessxy[i] += (F_term - A_Laplace_syy ) * weight;
        aRessyy[i] += ( solsyyGauss * phi[i] -  A_Laplace_sxy ) * weight;
*/

      } // end phi_i loop
    } // end gauss point loop


    // Add the local Matrix/Vector into the global Matrix/Vector

    //copy the value of the adept::adoube aRes in double Res and store
    Res.resize(4 * nDofs);

    for (int i = 0; i < nDofs; i++) {
      Res[i]              = -aResu[i].value();
      Res[nDofs + i]      = -aRessxx[i].value();
      Res[2 * nDofs + i]  = -aRessxy[i].value();
      Res[3 * nDofs + i]  = -aRessyy[i].value();
    }

    RES->add_vector_blocked(Res, sysDof);

    // resize jacobian for 4x4 blocked element: 16 blocks
    Jac.resize(16 * nDofs * nDofs);

        // define the independent variables (order matters — will define jacobian column ordering)
    s.independent(&solu[0],  nDofs);
    s.independent(&solsxx[0],  nDofs);
    s.independent(&solsxy[0], nDofs);
    s.independent(&solsyy[0], nDofs);


    // define the dependent variables (order matters — must match residual ordering)
    s.dependent(&aResu[0],  nDofs);
    s.dependent(&aRessxx[0],  nDofs);
    s.dependent(&aRessxy[0], nDofs);
    s.dependent(&aRessyy[0], nDofs);



    // get the jacobian matrix (ordered by column)
    s.jacobian(&Jac[0], true);

    KK->add_matrix_blocked(Jac, sysDof, sysDof);

             constexpr bool print_algebra_local = false;
     if (print_algebra_local) {

         assemble_jacobian<double,double>::print_element_jacobian(iel, Jac, Sol_n_el_dofs_Mat_vol, 10, 5);
         assemble_jacobian<double,double>::print_element_residual(iel, Res, Sol_n_el_dofs_Mat_vol, 10, 5);

     }

    s.clear_independents();
    s.clear_dependents();

  } //end element loop for each process

  RES->close();
  KK->close();


}


  };


}





#endif
