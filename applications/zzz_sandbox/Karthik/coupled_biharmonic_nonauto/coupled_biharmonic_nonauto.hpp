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
 * @brief Assembles the stiffness matrix (Jacobian) and residual vector for a coupled biharmonic problem.
 *
 * This function implements the manual assembly for the coupled biharmonic problem,
 * avoiding the use of automatic differentiation.
 * The biharmonic equation Δ²u = -f is split into a system of two second-order PDEs:
 *
 * Equation 1:  Δsxx = -f
 * Equation 2:  Δu = -sxx
 *
 * where sxx = Δu is an auxiliary variable.
 *
 * **Weak Formulation and Residuals:**
 * By multiplying each equation by a test function and integrating by parts, we get the weak form.
 *
 * 1.  For Equation 1 (Δsxx = -f) with test function v_u:
 * ∫_Ω (∇sxx ⋅ ∇v_u) dΩ - ∫_Ω (f * v_u) dΩ = 0
 * This gives the Residual R_u = ∫_Ω (∇sxx ⋅ ∇v_u - f * v_u) dΩ
 *
 * 2.  For Equation 2 (Δu = -sxx) with test function v_sxx:
 * ∫_Ω (∇u ⋅ ∇v_sxx) dΩ - ∫_Ω (sxx * v_sxx) dΩ = 0
 * This gives the Residual R_sxx = ∫_Ω (∇u ⋅ ∇v_sxx - sxx * v_sxx) dΩ
 *
 * **Jacobian Matrix:**
 * The Jacobian matrix J is a 2x2 block matrix of elemental contributions, where J_ij = ∂R_i / ∂u_j.
 * The unknowns are the coefficients of u and sxx.
 *
 * J_uu = ∂R_u / ∂u_j = 0
 * J_usxx = ∂R_u / ∂sxx_j = ∫_Ω (∇φ_j ⋅ ∇φ_i) dΩ
 * J_sxxu = ∂R_sxx / ∂u_j = ∫_Ω (∇φ_j ⋅ ∇φ_i) dΩ
 * J_sxxsxx = ∂R_sxx / ∂sxx_j = ∫_Ω (-φ_j ⋅ φ_i) dΩ
 *
 * @param ml_prob The MultiLevelProblem object.
 */
static void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob) {

    // --- Step 1: Extract Pointers to FEMUS Objects ---
    NonLinearImplicitSystem* mlPdeSys   = &ml_prob.get_system<NonLinearImplicitSystem> (ml_prob.get_app_specs_pointer()->_system_name);
    const unsigned level = mlPdeSys->GetLevelToAssemble();
    Mesh* msh          = ml_prob._ml_msh->GetLevel(level);
    elem* el         = msh->GetMeshElements();
    MultiLevelSolution* ml_sol        = ml_prob._ml_sol;
    Solution* sol        = ml_prob._ml_sol->GetSolutionLevel(level);
    LinearEquationSolver* pdeSys        = mlPdeSys->_LinSolver[level];
    SparseMatrix* KK         = pdeSys->_KK;
    NumericVector* RES          = pdeSys->_RES;
    const unsigned dim = msh->GetDimension();
    unsigned iproc = msh->processor_id();

    const std::string solname_u = ml_sol->GetSolName_string_vec()[0];
    unsigned soluIndex = ml_sol->GetIndex(solname_u.c_str());
    unsigned solFEType_u = ml_sol->GetSolutionType(soluIndex);
    unsigned soluPdeIndex = mlPdeSys->GetSolPdeIndex(solname_u.c_str());

    const std::string solname_sxx = ml_sol->GetSolName_string_vec()[1];
    unsigned solsxxIndex = ml_sol->GetIndex(solname_sxx.c_str());
    unsigned solFEType_sxx = ml_sol->GetSolutionType(solsxxIndex);
    unsigned solsxxPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxx.c_str());

    // --- Step 2: Local Data Structures for Element Assembly ---
    std::vector < double >  solu;
    std::vector < double >  solsxx;
    std::vector < std::vector < double > > x(dim);
    unsigned xType = 2; // LAGRANGE QUADRATIC
    std::vector < int > sysDof;
    std::vector <double> phi;
    std::vector <double> phi_x;
    std::vector <double> phi_xx;
    double weight;
    std::vector < double > Res_el_u;
    std::vector < double > Res_el_sxx;
    std::vector < double > Jac_el;

    const unsigned maxSize = static_cast< unsigned >(ceil(pow(3, dim)));
    solu.reserve(maxSize);
    solsxx.reserve(maxSize);
    for (unsigned i = 0; i < dim; i++) { x[i].reserve(maxSize); }
    sysDof.reserve(2 * maxSize);
    phi.reserve(maxSize);
    phi_x.reserve(maxSize * dim);
    unsigned dim2 = (3 * (dim - 1) + !(dim - 1));
    phi_xx.reserve(maxSize * dim2);
    Res_el_u.reserve(maxSize);
    Res_el_sxx.reserve(maxSize);
    Jac_el.reserve(4 * maxSize * maxSize);

    // --- Step 3: Global System Initialization ---
    KK->zero();
    RES->zero();

    // --- Step 4: Element Loop ---
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

        short unsigned ielGeom = msh->GetElementType(iel);
        unsigned nDofs_u = msh->GetElementDofNumber(iel, solFEType_u);
        unsigned nDofs_sxx = msh->GetElementDofNumber(iel, solFEType_sxx);
        unsigned nDofs_coords = msh->GetElementDofNumber(iel, xType);

        std::vector<unsigned> Sol_n_el_dofs_Mat_vol(2, nDofs_u);

        sysDof.resize(nDofs_u + nDofs_sxx);
        solu.resize(nDofs_u);
        solsxx.resize(nDofs_sxx);
        for (int i = 0; i < dim; i++) { x[i].resize(nDofs_coords); }
        Res_el_u.assign(nDofs_u, 0.0);
        Res_el_sxx.assign(nDofs_sxx, 0.0);
        Jac_el.assign( (nDofs_u + nDofs_sxx) * (nDofs_u + nDofs_sxx), 0.0);

        for (unsigned i = 0; i < nDofs_u; i++) {
            unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_u);
            solu[i] = (*sol->_Sol[soluIndex])(solDof);
            sysDof[i] = pdeSys->GetSystemDof(soluIndex, soluPdeIndex, i, iel);
        }
        for (unsigned i = 0; i < nDofs_sxx; i++) {
            unsigned solsxxDof = msh->GetSolutionDof(i, iel, solFEType_sxx);
            solsxx[i] = (*sol->_Sol[solsxxIndex])(solsxxDof);
            sysDof[nDofs_u + i] = pdeSys->GetSystemDof(solsxxIndex, solsxxPdeIndex, i, iel);
        }
        for (unsigned i = 0; i < nDofs_coords; i++) {
            unsigned xDof  = msh->GetSolutionDof(i, iel, xType);
            for (unsigned jdim = 0; jdim < dim; jdim++) {
                x[jdim][i] = (*msh->GetTopology()->_Sol[jdim])(xDof);
            }
        }

        // --- Step 5: Gauss Point Loop ---
        for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solFEType_u]->GetGaussPointNumber(); ig++) {
            msh->_finiteElement[ielGeom][solFEType_u]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);

            // Evaluate solutions and gradients at the Gauss point
            double soluGauss = 0;
            std::vector < double > soluGauss_x(dim, 0.);
            double solsxxGauss = 0;
            std::vector < double > solsxxGauss_x(dim, 0.);
            std::vector < double > xGauss(dim, 0.);

            for (unsigned i = 0; i < nDofs_u; i++) {
                soluGauss += phi[i] * solu[i];
                solsxxGauss += phi[i] * solsxx[i];

                for (unsigned jdim = 0; jdim < dim; jdim++) {
                    soluGauss_x[jdim] += phi_x[i * dim + jdim] * solu[i];
                    solsxxGauss_x[jdim] += phi_x[i * dim + jdim] * solsxx[i];
                    xGauss[jdim] += x[jdim][i] * phi[i];
                }
            }

            // --- Step 6: Shape Function Loop (Manual Residual and Jacobian Calculation) ---
            for (unsigned i = 0; i < nDofs_u; i++) {

                // Calculate residual for 'u' (Res_u: Δsxx = -f)
                double Laplace_sxx_term = 0.;
                for (unsigned jdim = 0; jdim < dim; jdim++) {
                    Laplace_sxx_term += phi_x[i * dim + jdim] * solsxxGauss_x[jdim];
                }
                double F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->laplacian(xGauss) * phi[i];
                Res_el_u[i] += (Laplace_sxx_term - F_term) * weight;

                // Calculate residual for 'sxx' (Res_sxx: Δu = -sxx)
                double Laplace_u_term = 0.;
                for (unsigned jdim = 0; jdim < dim; jdim++) {
                    Laplace_u_term += phi_x[i * dim + jdim] * soluGauss_x[jdim];
                }
                Res_el_sxx[i] += (Laplace_u_term - solsxxGauss * phi[i]) * weight;

                // Manual Jacobian calculation loop
                for (unsigned j = 0; j < nDofs_u; j++) {
                    // J_usxx: ∂R_u/∂sxx_j = ∫(∇φ_j ⋅ ∇φ_i) dΩ
                    double Jac_usxx = 0.;
                    for (unsigned jdim = 0; jdim < dim; jdim++) {
                        Jac_usxx += phi_x[i * dim + jdim] * phi_x[j * dim + jdim];
                    }
                    Jac_el[i * (nDofs_u + nDofs_sxx) + (nDofs_u + j)] += weight * Jac_usxx;

                    // J_sxxu: ∂R_sxx/∂u_j = ∫(∇φ_j ⋅ ∇φ_i) dΩ
                    double Jac_sxxu = 0.;
                    for (unsigned jdim = 0; jdim < dim; jdim++) {
                        Jac_sxxu += phi_x[i * dim + jdim] * phi_x[j * dim + jdim];
                    }
                    Jac_el[(nDofs_u + i) * (nDofs_u + nDofs_sxx) + j] += weight * Jac_sxxu;

                    // J_sxxsxx: ∂R_sxx/∂sxx_j = ∫(-φ_j ⋅ φ_i) dΩ
                    double Jac_sxxsxx = -phi[i] * phi[j];
                    Jac_el[(nDofs_u + i) * (nDofs_u + nDofs_sxx) + (nDofs_u + j)] += weight * Jac_sxxsxx;

                    // J_uu: ∂R_u/∂u_j = 0, so this block is left as zero.
                }
            } // end phi_i loop
        } // end gauss point loop

        // --- Step 7: Global Assembly ---
        std::vector<double> Res_total;
        Res_total.reserve(nDofs_u + nDofs_sxx);
        // The final residual is -1 times the calculated residual.
        for (int i = 0; i < nDofs_u; i++) { Res_total.push_back(-Res_el_u[i]); }
        for (int i = 0; i < nDofs_sxx; i++) { Res_total.push_back(-Res_el_sxx[i]); }

        RES->add_vector_blocked(Res_total, sysDof);
        KK->add_matrix_blocked(Jac_el, sysDof, sysDof);

        constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            assemble_jacobian<double,double>::print_element_jacobian(iel, Jac_el, Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, Res_total, Sol_n_el_dofs_Mat_vol, 10, 5);
        }
    } //end element loop

    RES->close();
    KK->close();
}


  };


}

#endif
