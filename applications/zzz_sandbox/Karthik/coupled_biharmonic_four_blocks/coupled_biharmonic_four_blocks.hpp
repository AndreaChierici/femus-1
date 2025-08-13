#ifndef __femus_biharmonic_coupled_hpp__
#define __femus_biharmonic_coupled_hpp__

#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"

using namespace femus;

namespace karthik {

  class biharmonic_coupled_equation {

  public:

/**
 * @brief Assembles the stiffness matrix and residual vector for a 4-equation biharmonic system.
 *
 * This function manually assembles the finite element system for the coupled biharmonic problem.
 * The system consists of four unknowns: u, sxx, sxy, and syy.
 *
 * **Weak Formulation and Residuals (Corrected):**
 * The residuals (R_u, R_sxx, R_sxy, R_syy) are derived from the weak form of the equations
 * implied by the original `adept` implementation.
 *
 * - R_u   = ∫(∇sxx ⋅ ∇v_u + f * v_u) dΩ
 * - R_sxx = ∫(∇u ⋅ ∇v_sxx + sxx * v_sxx) dΩ
 * - R_sxy = ∫(∇syy ⋅ ∇v_sxy + f * v_sxy) dΩ
 * - R_syy = ∫(∇sxy ⋅ ∇v_syy + syy * v_syy) dΩ
 *
 * **Jacobian Matrix (J_ij = ∂R_i / ∂u_j):**
 * The Jacobian is a 4x4 block matrix, with each block representing the partial derivative
 * of one residual with respect to one unknown.
 * - J_usxx   = ∂R_u/∂sxx_j   = ∫(∇φ_j ⋅ ∇φ_i) dΩ
 * - J_sxxu   = ∂R_sxx/∂u_j   = ∫(∇φ_j ⋅ ∇φ_i) dΩ
 * - J_sxxsxx = ∂R_sxx/∂sxx_j = ∫(φ_j ⋅ φ_i) dΩ  <-- Corrected sign
 * - J_sxy_syy = ∂R_sxy/∂syy_j = ∫(∇φ_j ⋅ ∇φ_i) dΩ
 * - J_syy_sxy = ∂R_syy/∂sxy_j = ∫(∇φ_j ⋅ ∇φ_i) dΩ
 * - J_syy_syy = ∂R_syy/∂syy_j = ∫(φ_j ⋅ φ_i) dΩ  <-- Corrected sign
 *
 * @param ml_prob The MultiLevelProblem object.
 */
static void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob) {

    // --- Step 1: Extract Pointers to FEMUS Objects ---
    NonLinearImplicitSystem* mlPdeSys = &ml_prob.get_system<NonLinearImplicitSystem>(ml_prob.get_app_specs_pointer()->_system_name);
    const unsigned level = mlPdeSys->GetLevelToAssemble();
    Mesh* msh = ml_prob._ml_msh->GetLevel(level);
    MultiLevelSolution* ml_sol = ml_prob._ml_sol;
    Solution* sol = ml_prob._ml_sol->GetSolutionLevel(level);
    LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
    SparseMatrix* KK = pdeSys->_KK;
    NumericVector* RES = pdeSys->_RES;
    const unsigned dim = msh->GetDimension();
    unsigned iproc = msh->processor_id();

    // Solution and PDE indices for all four unknowns
    const std::string solname_u = ml_sol->GetSolName_string_vec()[0];
    unsigned soluIndex = ml_sol->GetIndex(solname_u.c_str());
    unsigned solFEType_u = ml_sol->GetSolutionType(soluIndex);
    unsigned soluPdeIndex = mlPdeSys->GetSolPdeIndex(solname_u.c_str());

    const std::string solname_sxx = ml_sol->GetSolName_string_vec()[1];
    unsigned solsxxIndex = ml_sol->GetIndex(solname_sxx.c_str());
    unsigned solFEType_sxx = ml_sol->GetSolutionType(solsxxIndex);
    unsigned solsxxPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxx.c_str());

    const std::string solname_sxy = ml_sol->GetSolName_string_vec()[2];
    unsigned solsxyIndex = ml_sol->GetIndex(solname_sxy.c_str());
    unsigned solFEType_sxy = ml_sol->GetSolutionType(solsxyIndex);
    unsigned solsxyPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxy.c_str());

    const std::string solname_syy = ml_sol->GetSolName_string_vec()[3];
    unsigned solsyyIndex = ml_sol->GetIndex(solname_syy.c_str());
    unsigned solFEType_syy = ml_sol->GetSolutionType(solsyyIndex);
    unsigned solsyyPdeIndex = mlPdeSys->GetSolPdeIndex(solname_syy.c_str());

    // --- Step 2: Local Data Structures for Element Assembly ---
    std::vector<double> solu;
    std::vector<double> solsxx;
    std::vector<double> solsxy;
    std::vector<double> solsyy;
    std::vector<std::vector<double>> x(dim);
    unsigned xType = 2; // LAGRANGE QUADRATIC for coordinates
    std::vector<int> sysDof;
    std::vector<double> phi;
    std::vector<double> phi_x;
    std::vector<double> phi_xx;
    double weight;
    std::vector<double> Res_el_u;
    std::vector<double> Res_el_sxx;
    std::vector<double> Res_el_sxy;
    std::vector<double> Res_el_syy;
    std::vector<double> Jac_el;

    // Reserve memory to prevent reallocations
    const unsigned maxSize = static_cast<unsigned>(ceil(pow(3, dim)));
    solu.reserve(maxSize);
    solsxx.reserve(maxSize);
    solsxy.reserve(maxSize);
    solsyy.reserve(maxSize);
    for (unsigned i = 0; i < dim; i++) { x[i].reserve(maxSize); }
    sysDof.reserve(4 * maxSize);
    phi.reserve(maxSize);
    phi_x.reserve(maxSize * dim);
    unsigned dim2 = (6 * (dim - 1) + !(dim - 1));
    phi_xx.reserve(maxSize * dim2);
    Res_el_u.reserve(maxSize);
    Res_el_sxx.reserve(maxSize);
    Res_el_sxy.reserve(maxSize);
    Res_el_syy.reserve(maxSize);
    Jac_el.reserve(16 * maxSize * maxSize);

    // --- Step 3: Global System Initialization ---
    KK->zero();
    RES->zero();

    // --- Step 4: Element Loop ---
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

        short unsigned ielGeom = msh->GetElementType(iel);
        unsigned nDofs = msh->GetElementDofNumber(iel, solFEType_u);
        unsigned nDofs_coords = msh->GetElementDofNumber(iel, xType);

        std::vector<unsigned> Sol_n_el_dofs_Mat_vol(4, nDofs);

        // Resize local vectors
        sysDof.resize(4 * nDofs);
        solu.resize(nDofs);
        solsxx.resize(nDofs);
        solsxy.resize(nDofs);
        solsyy.resize(nDofs);
        for (int i = 0; i < dim; i++) { x[i].resize(nDofs_coords); }
        Res_el_u.assign(nDofs, 0.0);
        Res_el_sxx.assign(nDofs, 0.0);
        Res_el_sxy.assign(nDofs, 0.0);
        Res_el_syy.assign(nDofs, 0.0);
        Jac_el.assign((4 * nDofs) * (4 * nDofs), 0.0);

        // Local storage of solutions and global mapping
        for (unsigned i = 0; i < nDofs; i++) {
            unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_u);
            solu[i] = (*sol->_Sol[soluIndex])(solDof);
            solsxx[i] = (*sol->_Sol[solsxxIndex])(solDof);
            solsxy[i] = (*sol->_Sol[solsxyIndex])(solDof);
            solsyy[i] = (*sol->_Sol[solsyyIndex])(solDof);

            sysDof[i] = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);
            sysDof[nDofs + i] = pdeSys->GetSystemDof(solsxxIndex, solsxxPdeIndex, i, iel);
            sysDof[2 * nDofs + i] = pdeSys->GetSystemDof(solsxyIndex, solsxyPdeIndex, i, iel);
            sysDof[3 * nDofs + i] = pdeSys->GetSystemDof(solsyyIndex, solsyyPdeIndex, i, iel);
        }

        // Local storage of coordinates
        for (unsigned i = 0; i < nDofs_coords; i++) {
            unsigned xDof = msh->GetSolutionDof(i, iel, xType);
            for (unsigned jdim = 0; jdim < dim; jdim++) {
                x[jdim][i] = (*msh->GetTopology()->_Sol[jdim])(xDof);
            }
        }

        // --- Step 5: Gauss Point Loop (Numerical Integration) ---
        for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solFEType_u]->GetGaussPointNumber(); ig++) {
            msh->_finiteElement[ielGeom][solFEType_u]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);

            // Evaluate solutions and gradients at the Gauss point
            double soluGauss = 0;
            std::vector<double> soluGauss_x(dim, 0.);
            double solsxxGauss = 0;
            std::vector<double> solsxxGauss_x(dim, 0.);
            double solsxyGauss = 0;
            std::vector<double> solsxyGauss_x(dim, 0.);
            double solsyyGauss = 0;
            std::vector<double> solsyyGauss_x(dim, 0.);
            std::vector<double> xGauss(dim, 0.);

            for (unsigned i = 0; i < nDofs; i++) {
                soluGauss += phi[i] * solu[i];
                solsxxGauss += phi[i] * solsxx[i];
                solsxyGauss += phi[i] * solsxy[i];
                solsyyGauss += phi[i] * solsyy[i];
                for (unsigned jdim = 0; jdim < dim; jdim++) {
                    soluGauss_x[jdim] += phi_x[i * dim + jdim] * solu[i];
                    solsxxGauss_x[jdim] += phi_x[i * dim + jdim] * solsxx[i];
                    solsxyGauss_x[jdim] += phi_x[i * dim + jdim] * solsxy[i];
                    solsyyGauss_x[jdim] += phi_x[i * dim + jdim] * solsyy[i];
                    xGauss[jdim] += x[jdim][i] * phi[i];
                }
            }

            // --- Step 6: Shape Function Loop (Residual and Jacobian Calculation) ---
            for (unsigned i = 0; i < nDofs; i++) {

                // Common terms
                double Laplace_sxx_term = 0.;
                for (unsigned jdim = 0; jdim < dim; jdim++) { Laplace_sxx_term += phi_x[i * dim + jdim] * solsxxGauss_x[jdim]; }

                double Laplace_u_term = 0.;
                for (unsigned jdim = 0; jdim < dim; jdim++) { Laplace_u_term += phi_x[i * dim + jdim] * soluGauss_x[jdim]; }

                double Laplace_syy_term = 0.;
                for (unsigned jdim = 0; jdim < dim; jdim++) { Laplace_syy_term += phi_x[i * dim + jdim] * solsyyGauss_x[jdim]; }

                double Laplace_sxy_term = 0.;
                for (unsigned jdim = 0; jdim < dim; jdim++) { Laplace_sxy_term += phi_x[i * dim + jdim] * solsxyGauss_x[jdim]; }

                double F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->laplacian(xGauss) * phi[i];

                // Residual 1: R_u = ∫(∇sxx ⋅ ∇v_u + f * v_u) dΩ
                Res_el_u[i] += (Laplace_sxx_term + F_term) * weight;

                // Residual 2: R_sxx = ∫(∇u ⋅ ∇v_sxx + sxx * v_sxx) dΩ
                Res_el_sxx[i] += (Laplace_u_term + solsxxGauss * phi[i]) * weight;

                // Residual 3: R_sxy = ∫(∇syy ⋅ ∇v_sxy + f * v_sxy) dΩ
                Res_el_sxy[i] += (Laplace_syy_term + F_term) * weight;

                // Residual 4: R_syy = ∫(∇sxy ⋅ ∇v_syy + syy * v_syy) dΩ
                Res_el_syy[i] += (Laplace_sxy_term + solsyyGauss * phi[i]) * weight;

                // Manual Jacobian calculation loop (all 16 blocks)
                for (unsigned j = 0; j < nDofs; j++) {

                    // Common terms
                    double jac_laplace_term = 0.;
                    for (unsigned jdim = 0; jdim < dim; jdim++) {
                        jac_laplace_term += phi_x[i * dim + jdim] * phi_x[j * dim + jdim];
                    }
                    double jac_mass_term = phi[i] * phi[j];

                    // J_usxx: ∂R_u/∂sxx_j
                    Jac_el[i * (4 * nDofs) + (nDofs + j)] += weight * jac_laplace_term;

                    // J_sxxu: ∂R_sxx/∂u_j
                    Jac_el[(nDofs + i) * (4 * nDofs) + j] += weight * jac_laplace_term;

                    // J_sxxsxx: ∂R_sxx/∂sxx_j
                    Jac_el[(nDofs + i) * (4 * nDofs) + (nDofs + j)] += weight * jac_mass_term;

                    // J_sxy_syy: ∂R_sxy/∂syy_j
                    Jac_el[(2 * nDofs + i) * (4 * nDofs) + (3 * nDofs + j)] += weight * jac_laplace_term;

                    // J_syy_sxy: ∂R_syy/∂sxy_j
                    Jac_el[(3 * nDofs + i) * (4 * nDofs) + (2 * nDofs + j)] += weight * jac_laplace_term;

                    // J_syy_syy: ∂R_syy/∂syy_j
                    Jac_el[(3 * nDofs + i) * (4 * nDofs) + (3 * nDofs + j)] += weight * jac_mass_term;

                    // The other Jacobian blocks are zero
                }
            } // end phi_i loop
        } // end gauss point loop

        // --- Step 7: Global Assembly ---
        std::vector<double> Res_total;
        Res_total.reserve(4 * nDofs);
        for (int i = 0; i < nDofs; i++) { Res_total.push_back(-Res_el_u[i]); }
        for (int i = 0; i < nDofs; i++) { Res_total.push_back(-Res_el_sxx[i]); }
        for (int i = 0; i < nDofs; i++) { Res_total.push_back(-Res_el_sxy[i]); }
        for (int i = 0; i < nDofs; i++) { Res_total.push_back(-Res_el_syy[i]); }

        RES->add_vector_blocked(Res_total, sysDof);
        KK->add_matrix_blocked(Jac_el, sysDof, sysDof);

        constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            assemble_jacobian<double,double>::print_element_jacobian(iel, Jac_el, Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, Res_total, Sol_n_el_dofs_Mat_vol, 10, 5);
        }
    } // end element loop

    RES->close();
    KK->close();
}


  };


}

#endif
