#ifndef __femus_poisson_hpp__
#define __femus_poisson_hpp__

// Standard FEMUS headers
#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"

// Use the femus namespace to avoid redundant typing
using namespace femus;

namespace karthik {

  /**
   * @class poisson_equation
   * @brief This class contains methods for assembling the system of equations for the Poisson problem.
   */
  class poisson_equation {

  public:

/**
 * @brief Assembles the stiffness matrix (Jacobian) and residual vector for the Poisson problem.
 *
 * This function implements the manual assembly of the finite element system, avoiding automatic
 * differentiation. It iterates through each element of the mesh, calculates local contributions,
 * and then assembles them into the global sparse matrix and residual vector.
 *
 * **Problem Statement:**
 * The code solves the Poisson equation:
 * -Δu = f
 * with appropriate boundary conditions, where u is the unknown solution and f is a given source term.
 *
 * **Weak Formulation:**
 * To solve this using the Finite Element Method (FEM), we use the weak formulation.
 * We seek a solution u in a function space such that for all test functions v in the same space:
 * ∫_Ω (∇u ⋅ ∇v) dΩ = ∫_Ω (f * v) dΩ
 *
 * We approximate the solution u with a linear combination of shape functions φ_j:
 * u_h = Σ_j u_j * φ_j
 *
 * Substituting u_h for u and using test functions v_i = φ_i, we get the discrete system:
 * Σ_j (∫_Ω (∇φ_j ⋅ ∇φ_i) dΩ) * u_j = ∫_Ω (f * φ_i) dΩ
 *
 * This can be written in matrix form as:
 * Ku = F
 *
 * Where K is the stiffness matrix with entries K_ij = ∫_Ω (∇φ_j ⋅ ∇φ_i) dΩ,
 * and F is the load vector with entries F_i = ∫_Ω (f * φ_i) dΩ.
 *
 * The residual R_i for a nonlinear solver is then:
 * R_i = ∫_Ω (∇u_h ⋅ ∇φ_i - f * φ_i) dΩ
 *
 * The Jacobian J_ij is the derivative of the residual R_i with respect to the unknown u_j:
 * J_ij = ∂R_i / ∂u_j = ∫_Ω (∇φ_j ⋅ ∇φ_i) dΩ
 *
 * @param ml_prob The MultiLevelProblem object containing all problem data.
 */
static void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob) {

  // --- Step 1: Extract Pointers to FEMUS Objects ---
  // These objects manage the mesh, solution, and linear system for a specific level.
  NonLinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<NonLinearImplicitSystem> (ml_prob.get_app_specs_pointer()->_system_name);
  const unsigned level = mlPdeSys->GetLevelToAssemble();
  Mesh* msh          = ml_prob._ml_msh->GetLevel(level);
  elem* el         = msh->GetMeshElements();
  MultiLevelSolution* ml_sol        = ml_prob._ml_sol;
  Solution* sol        = ml_prob._ml_sol->GetSolutionLevel(level);
  LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level];
  SparseMatrix* KK         = pdeSys->_KK;  // Global Stiffness Matrix (Jacobian)
  NumericVector* RES          = pdeSys->_RES; // Global Residual Vector

  // Get problem metadata
  const unsigned  dim = msh->GetDimension();
  unsigned    iproc = msh->processor_id(); // for parallel processing
  const std::string solname_u = ml_sol->GetSolName_string_vec()[0];
  unsigned soluIndex = ml_sol->GetIndex(solname_u.c_str());
  unsigned solFEType_u = ml_sol->GetSolutionType(soluIndex);
  unsigned soluPdeIndex = mlPdeSys->GetSolPdeIndex(solname_u.c_str());
  unsigned xType = 2; // LAGRANGE QUADRATIC for coordinate mapping

  // --- Step 2: Local Data Structures for Element Assembly ---
  // These vectors will store data for a single element during assembly.
  std::vector < double >  solu_eldofs; // local solution degrees of freedom
  std::vector < std::vector < double > > x(dim);    // local coordinates of nodes
  std::vector < int > sysDof; // mapping from local to global DoFs
  std::vector <double> phi;  // shape function values at a Gauss point
  std::vector <double> phi_x; // shape function first derivatives at a Gauss point
  std::vector <double> phi_xx; // shape function second derivatives at a Gauss point
  double weight; // Gauss point weight
  std::vector < double > Res_el; // local residual vector
  std::vector < double > Jac_el; // local Jacobian matrix

  // Reserve memory to prevent reallocations inside the loops.
  const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));
  solu_eldofs.reserve(maxSize);
  for (unsigned i = 0; i < dim; i++){ x[i].reserve(maxSize);}
  sysDof.reserve(1 * maxSize);
  phi.reserve(maxSize);
  phi_x.reserve(maxSize * dim);
  unsigned dim2 = (2 * (dim - 1) + !(dim - 1));
  phi_xx.reserve(maxSize * dim2);
  Res_el.reserve(1 * maxSize);
  Jac_el.reserve(4 * maxSize * maxSize);

  // --- Step 3: Global System Initialization ---
  KK->zero(); // Zero out the global stiffness matrix
  RES->zero(); // Zero out the global residual vector

  // --- Step 4: Element Loop (The heart of FEM assembly) ---
  for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

    short unsigned ielGeom = msh->GetElementType(iel);
    unsigned nDofs  = msh->GetElementDofNumber(iel, solFEType_u);
    unsigned nDofs_coords = msh->GetElementDofNumber(iel, xType);

    std::vector<unsigned> Sol_n_el_dofs_Mat_vol(1, nDofs);

    // Resize local vectors to the correct number of DoFs for this element
    sysDof.resize(1 * nDofs);
    solu_eldofs.resize(nDofs);
    for (int i = 0; i < dim; i++) { x[i].resize(nDofs_coords); }
    Res_el.assign(nDofs, 0.);

    // --- Local Data Extraction ---
    // Get the current solution values and the global DoF indices for this element.
    for (unsigned i = 0; i < nDofs; i++) {
      unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_u);
      solu_eldofs[i] = (*sol->_Sol[soluIndex])(solDof);
      sysDof[i] = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);
    }

    // Get the physical coordinates of the element's nodes.
    for (unsigned i = 0; i < nDofs_coords; i++) {
      unsigned xDof  = msh->GetSolutionDof(i, iel, xType);
      for (unsigned jdim = 0; jdim < dim; jdim++) {
        x[jdim][i] = (*msh->GetTopology()->_Sol[jdim])(xDof);
      }
    }

    // Reset local Jacobian for the current element.
    Jac_el.resize(nDofs * nDofs);
    std::fill(Jac_el.begin(), Jac_el.end(), 0.0);

    // --- Step 5: Gauss Point Loop ---
    // Iterate over each quadrature point to perform numerical integration.
    for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solFEType_u]->GetGaussPointNumber(); ig++) {
      // Get the shape functions (phi) and their derivatives (phi_x, phi_xx) at the current Gauss point.
      // The `Jacobian` function handles the mapping from reference to physical coordinates.
      msh->_finiteElement[ielGeom][solFEType_u]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);

      // --- Evaluate Solution and Gradient at Gauss Point ---
      // This is the value of the approximated solution u_h and its gradient ∇u_h at the quadrature point.
      double solu_qp = 0;
      std::vector < double > solu_grad_qp(dim, 0.);
      std::vector < double > xGauss(dim, 0.);

      for (unsigned i = 0; i < nDofs; i++) {
        solu_qp += phi[i] * solu_eldofs[i];
        for (unsigned jdim = 0; jdim < dim; jdim++) {
          solu_grad_qp[jdim]  += phi_x[i * dim + jdim] * solu_eldofs[i];
          xGauss[jdim] += x[jdim][i] * phi[i]; // Physical coordinates of the Gauss point
        }
      }

      // --- Step 6: Shape Function Loop (Manual Residual and Jacobian Calculation) ---
      // Iterate over each local shape function φ_i to compute the local residual and Jacobian.
      for (unsigned i = 0; i < nDofs; i++) {
        // --- Residual Calculation ---
        // Calculate the contribution of the Laplace term to the residual.
        // This corresponds to the term ∫(∇u ⋅ ∇φ_i) dΩ.
        double Laplace_u = 0.;
        for (unsigned jdim = 0; jdim < dim; jdim++) {
          Laplace_u   +=  - phi_x[i * dim + jdim] * solu_grad_qp[jdim];
        }

        // Calculate the contribution of the source term f.
        // This corresponds to the term ∫(f * φ_i) dΩ.
        double F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->value(xGauss) * phi[i];

        // Add the combined contributions to the local residual vector `Res_el`.
        // The `weight` accounts for the numerical integration.
        Res_el[i] += (F_term - Laplace_u) * weight;

        // --- Jacobian Calculation ---
        // Iterate over each local shape function φ_j to compute the Jacobian matrix J_ij.
        // J_ij = ∂R_i / ∂u_j = ∫(∇φ_j ⋅ ∇φ_i) dΩ
        for (unsigned j = 0; j < nDofs; j++) {
          double jac_term = 0.;
          // Sum the dot product of the gradients of the test function φ_i and trial function φ_j.
          for (unsigned jdim = 0; jdim < dim; jdim++) {
            jac_term += phi_x[i * dim + jdim] * phi_x[j * dim + jdim];
          }
          // Add the contribution to the local Jacobian matrix `Jac_el`.
          Jac_el[i * nDofs + j] += weight * jac_term;
        }
      } // end phi_i loop
    } // end gauss point loop


    // --- Step 7: Global Assembly ---
    // Transfer the local element contributions to the global system.
    // Note: The sign of the residual is flipped here to match the solver convention.
    for (int i = 0; i < nDofs; i++) {
      Res_el[i] = -Res_el[i];
    }
    // Add local residual vector `Res_el` to global `RES`.
    RES->add_vector_blocked(Res_el, sysDof);
    // Add local Jacobian matrix `Jac_el` to global `KK`.
    KK->add_matrix_blocked(Jac_el, sysDof, sysDof);

    // Optional: print local algebraic quantities for debugging.
    constexpr bool print_algebra_local = false;
     if (print_algebra_local) {
         assemble_jacobian<double,double>::print_element_jacobian(iel, Jac_el, Sol_n_el_dofs_Mat_vol, 10, 5);
         assemble_jacobian<double,double>::print_element_residual(iel, Res_el, Sol_n_el_dofs_Mat_vol, 10, 5);
     }
  } //end element loop

  // --- Step 8: Finalize the Global System ---
  // This step finalizes the global matrix and vector for parallel communication and assembly.
  RES->close();
  KK->close();
}


  };


}

#endif
