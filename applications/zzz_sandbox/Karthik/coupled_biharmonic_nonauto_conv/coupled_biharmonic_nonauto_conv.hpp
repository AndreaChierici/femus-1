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
#include "Assemble_unknown_jacres.hpp" // Required for ElementJacRes
#include "CurrentElem.hpp"            // Required for CurrentElem

using namespace femus;

namespace karthik {

  class biharmonic_coupled_equation {

  public:

/**
 * @brief Assembles the stiffness matrix (Jacobian) and residual vector for a coupled biharmonic problem.
 *
 * This function implements the manual assembly for the coupled biharmonic problem,
 * avoiding the use of automatic differentiation. It follows the structure of
 * `System_assemble_flexible_Laplacian_With_Manufactured_Sol` for consistency within FEMUS.
 *
 * The biharmonic equation $\Delta^2 u = -f$ is split into a system of two second-order PDEs:
 *
 * Equation 1: $\Delta s_{xx} = -f$
 * Equation 2: $\Delta u = -s_{xx}$
 *
 * where $s_{xx} = \Delta u$ is an auxiliary variable.
 *
 * **Weak Formulation and Residuals:**
 * By multiplying each equation by a test function and integrating by parts, we get the weak form.
 *
 * 1. For Equation 1 ($\Delta s_{xx} = -f$) with test function $v_u$:
 * $\int_\Omega (\nabla s_{xx} \cdot \nabla v_u) d\Omega - \int_\Omega (f \cdot v_u) d\Omega = 0$
 * This gives the Residual $R_u = \int_\Omega (\nabla s_{xx} \cdot \nabla v_u - f \cdot v_u) d\Omega$
 *
 * 2. For Equation 2 ($\Delta u = -s_{xx}$) with test function $v_{s_{xx}}$:
 * $\int_\Omega (\nabla u \cdot \nabla v_{s_{xx}}) d\Omega - \int_\Omega (s_{xx} \cdot v_{s_{xx}}) d\Omega = 0$
 * This gives the Residual $R_{s_{xx}} = \int_\Omega (\nabla u \cdot \nabla v_{s_{xx}} - s_{xx} \cdot v_{s_{xx}}) d\Omega$
 *
 * **Jacobian Matrix:**
 * The Jacobian matrix J is a 2x2 block matrix of elemental contributions, where $J_{ij} = \partial R_i / \partial \text{coeff}_j$.
 * The unknowns are the coefficients of u and sxx.
 *
 * $J_{uu} = \partial R_u / \partial u_j = 0$
 * $J_{us_{xx}} = \partial R_u / \partial s_{xx_j} = \int_\Omega (\nabla \phi_j \cdot \nabla \phi_i) d\Omega$
 * $J_{s_{xx}u} = \partial R_{s_{xx}} / \partial u_j = \int_\Omega (\nabla \phi_j \cdot \nabla \phi_i) d\Omega$
 * $J_{s_{xx}s_{xx}} = \partial R_{s_{xx}} / \partial s_{xx_j} = \int_\Omega (-\phi_j \cdot \phi_i) d\Omega$
 *
 * @tparam system_type Type of the PDE system (e.g., NonLinearImplicitSystem).
 * @tparam real_num Numeric type for solution variables (e.g., double).
 * @tparam real_num_mov Numeric type for moving domain variables (e.g., double).
 * @param elem_all Vector of element types for shape function evaluation.
 * @param elem_all_for_domain Vector of element types for domain geometry evaluation.
 * @param quad_rules Quadrature rules for integration.
 * @param mlPdeSys MultiLevel PDE system.
 * @param ml_mesh_in MultiLevel mesh.
 * @param ml_sol_in MultiLevel solution.
 * @param unknowns Vector of Unknown objects defining the problem's unknowns.
 * @param source_functions Vector of Math::Function objects for the source term.
 */
template < class system_type, class real_num, class real_num_mov >
static void AssembleBilaplaceProblem(
                            const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > & elem_all,
                            const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > & elem_all_for_domain,
                            const std::vector<Gauss> & quad_rules,
                            system_type * mlPdeSys,
                            MultiLevelMesh * ml_mesh_in,
                            MultiLevelSolution * ml_sol_in,
                            const std::vector< Unknown > &  unknowns,
                            const std::vector< Math::Function< double > * > & source_functions) {

    // level is the level of the PDE system to be assembled
    const unsigned level = mlPdeSys->GetLevelToAssemble();
    const bool assembleMatrix = mlPdeSys->GetAssembleMatrix();

    Mesh* msh = ml_mesh_in->GetLevel(level);

    MultiLevelSolution* ml_sol = ml_sol_in;
    Solution* sol = ml_sol->GetSolutionLevel(level);

    LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
    SparseMatrix* KK = pdeSys->_KK;
    NumericVector* RES = pdeSys->_RES;

    const unsigned dim = msh->GetDimension();
    const unsigned iproc = msh->processor_id(); // get the process_id (for parallel computation)

    RES->zero();
    if (assembleMatrix) KK->zero();

    // The ADEPT stack is typically used for automatic differentiation, but here we do manual assembly.
    // However, the interface might still require it if using `assemble_jacobian` calls.
    adept::Stack & stack = FemusInit::_adeptStack;

    // This is typically used with AD or for a single-unknown problem.
    // For coupled manual assembly, we will directly compute the Jacobian terms.
    // const assemble_jacobian< real_num, double > * unk_assemble_jac;


    constexpr unsigned int space_dim = 3; // Assuming max spatial dimension 3 for general compatibility
    const unsigned int dim_offset_grad = dim; // Use actual problem dimension for gradients

    std::vector < std::vector < /*double*/ real_num_mov > > JacI_qp(space_dim);
    std::vector < std::vector < /*double*/ real_num_mov > > Jac_qp(dim);
    for (unsigned d = 0; d < dim; d++) { Jac_qp[d].resize(space_dim); }
    for (unsigned d = 0; d < space_dim; d++) { JacI_qp[d].resize(dim); }

    real_num_mov detJac_qp;

    //=============== Integration ========================================
    real_num_mov weight_qp;

    //=============== Geometry - BEGIN ========================================
    unsigned xType = CONTINUOUS_BIQUADRATIC; // Or appropriate geometric element type

    CurrentElem < real_num_mov > geom_element(dim, msh);
    Phi < real_num_mov > geom_element_phi_dof_qp(dim_offset_grad);
    //=============== Geometry - END ========================================

    //=============== Unknowns - BEGIN ========================================
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size(); // Should be 2 (u, sxx)

    std::vector < UnknownLocal < real_num > > unknowns_local(n_unknowns);    //-- at dofs
    // unknowns_phi_dof_qp will store shape function values and derivatives at quadrature points
    std::vector < Phi < real_num > > unknowns_phi_dof_qp(n_unknowns, Phi< real_num >(dim_offset_grad)); //-- at dofs and quadrature points ---------------

    for(int u = 0; u < n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
        // assert(u == unknowns_local[u].pde_index()); // This assertion might fail if pde_index doesn't match the vector index
    }

    //=============== Unknowns, Elem matrix and Rhs - BEGIN ========================================
    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);
    //=============== Unknowns, Elem matrix and Rhs - END ========================================

    std::vector < unsigned int > unk_num_elem_dofs_interface(n_unknowns); //to avoid recomputing offsets inside quadrature
    //=============== Unknowns - END ========================================

    // No exact solution is used directly in this assembly for manufactured solution comparison within the loop,
    // but the source function `f` is derived from it.
    // UnknownLocal < double > sol_exact; // Not directly used in the assembly anymore


    // element loop: each process loops only on the elements that owns
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

        //=============== Geometry - BEGIN ========================================
        geom_element.set_coords_at_dofs_and_geom_type(iel, xType);
        geom_element.set_elem_center_3d(iel, xType);
        const short unsigned ielGeom = geom_element.geom_type();
        //=============== Geometry - END ========================================

        //=============== Unknowns - BEGIN ========================================
        for (unsigned u = 0; u < n_unknowns; u++) {
            unknowns_local[u].set_elem_dofs(iel, msh, sol);
        }
        //=============== Unknowns - END ========================================

        // No direct use of sol_exact's set_elem_dofs in this biharmonic assembly,
        // as the source term 'f' is directly obtained from `source_functions[0]->laplacian(x_gss)`.
        // sol_exact.set_elem_dofs(unknowns_local[0].num_elem_dofs(), geom_element, * source_functions[0] );

        //=============== Unknowns, Elem matrix and Rhs - BEGIN ========================================
        unk_element_jac_res.set_loc_to_glob_map(iel, msh, pdeSys);
        // Resize local residual and Jacobian for current element
        unk_element_jac_res.res().assign(unk_element_jac_res.dof_map().size(), 0.0);
        unk_element_jac_res.jac().assign(unk_element_jac_res.dof_map().size() * unk_element_jac_res.dof_map().size(), 0.0);
        //=============== Unknowns, Elem matrix and Rhs - END ========================================


        // unk_assemble_jac->prepare_before_integration_loop(stack); // Not directly used for manual assembly

        //interface to avoid computation inside quadrature - BEGIN
        unsigned sum_unk_num_elem_dofs_interface = 0;
        for (unsigned u = 0; u < n_unknowns; u++) {
            unk_num_elem_dofs_interface[u] = unknowns_local[u].num_elem_dofs();
            sum_unk_num_elem_dofs_interface += unk_num_elem_dofs_interface[u];
        }
        //interface to avoid computation inside quadrature - END

        // *** Gauss point loop ***
        for (unsigned ig = 0; ig < quad_rules[ielGeom].GetGaussPointsNumber(); ig++) {

            // Compute Jacobian of transformation, its inverse, determinant, and quadrature weight
            elem_all_for_domain[ielGeom][xType]->JacJacInv(geom_element.get_coords_at_dofs_3d(), ig, Jac_qp, JacI_qp, detJac_qp, space_dim);
            weight_qp = detJac_qp * quad_rules[ielGeom].GetGaussWeightsPointer()[ig];

            // *** Get shape functions and their partial derivatives at the Gauss point ***
            for (unsigned u = 0; u < n_unknowns; u++) {
                // The `fe_type()` of the unknown defines which shape functions to use.
                elem_all[ielGeom][unknowns_local[u].fe_type()]->shape_funcs_current_elem(ig, JacI_qp, unknowns_phi_dof_qp[u].phi(), unknowns_phi_dof_qp[u].phi_grad(), unknowns_phi_dof_qp[u].phi_hess(), space_dim);
            }
            // Shape functions for geometry mapping
            elem_all_for_domain[ielGeom][xType]->shape_funcs_current_elem(ig, JacI_qp, geom_element_phi_dof_qp.phi(), geom_element_phi_dof_qp.phi_grad(), geom_element_phi_dof_qp.phi_hess(), space_dim);


            // Evaluate the solution, its derivatives, and coordinates in the gauss point
            // For 'u' (unknowns_local[0])
            real_num solu_u_gss = 0.;
            std::vector < real_num > gradSolu_u_gss(dim_offset_grad, 0.);
            for (unsigned i = 0; i < unknowns_local[0].num_elem_dofs(); i++) {
                solu_u_gss += unknowns_phi_dof_qp[0].phi(i) * unknowns_local[0].elem_dofs()[i];
                for (unsigned jdim = 0; jdim < dim_offset_grad; jdim++) {
                    gradSolu_u_gss[jdim] += unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + jdim) * unknowns_local[0].elem_dofs()[i];
                }
            }

            // For 'sxx' (unknowns_local[1])
            real_num solu_sxx_gss = 0.;
            std::vector < real_num > gradSolu_sxx_gss(dim_offset_grad, 0.);
            for (unsigned i = 0; i < unknowns_local[1].num_elem_dofs(); i++) {
                solu_sxx_gss += unknowns_phi_dof_qp[1].phi(i) * unknowns_local[1].elem_dofs()[i];
                for (unsigned jdim = 0; jdim < dim_offset_grad; jdim++) {
                    gradSolu_sxx_gss[jdim] += unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + jdim) * unknowns_local[1].elem_dofs()[i];
                }
            }

            std::vector < double > x_gss(dim, 0.); // Physical coordinates at Gauss point
            for (unsigned i = 0; i < geom_element.get_coords_at_dofs()[0].size(); i++) {
                for (unsigned jdim = 0; jdim < x_gss.size(); jdim++) {
                    x_gss[jdim] += geom_element.get_coords_at_dofs(jdim,i) * geom_element_phi_dof_qp.phi(i);
                }
            }

            // --- Manual Residual and Jacobian Calculation ---

            const unsigned nDofs_u = unknowns_local[0].num_elem_dofs();
            const unsigned nDofs_sxx = unknowns_local[1].num_elem_dofs();
            const unsigned total_elem_dofs = nDofs_u + nDofs_sxx;

            // Iterate over test functions (i for rows of element matrix/vector)
            // Note: Assuming test functions for u and sxx have the same number of DOFs and types
            for (unsigned i = 0; i < nDofs_u; i++) { // For test function associated with 'u' (first equation)

                // Residual for 'u' (R_u): from $\int_\Omega (\nabla s_{xx} \cdot \nabla v_u - f \cdot v_u) d\Omega$
                real_num laplace_sxx_term_res = 0.;
                for (unsigned jdim = 0; jdim < dim_offset_grad; jdim++) {
                    laplace_sxx_term_res += unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + jdim) * gradSolu_sxx_gss[jdim];
                }
                double f_source_term = source_functions[0]->laplacian(x_gss); // The 'f' term
                unk_element_jac_res.res()[i] += (laplace_sxx_term_res - f_source_term * unknowns_phi_dof_qp[0].phi(i)) * weight_qp;


                // Iterate over basis functions (j for columns of element matrix)
                for (unsigned j = 0; j < nDofs_u; j++) { // Loop for basis functions of u and sxx

                    // J_usxx: $\partial R_u / \partial s_{xx_j} = \int_\Omega (\nabla \phi_j \cdot \nabla \phi_i) d\Omega$
                    // Row for R_u (i), Column for sxx_j (nDofs_u + j)
                    real_num jac_usxx = 0.;
                    for (unsigned kdim = 0; kdim < dim_offset_grad; kdim++) {
                        jac_usxx += unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + kdim) * unknowns_phi_dof_qp[1].phi_grad(j * dim_offset_grad + kdim);
                    }
                    unk_element_jac_res.jac()[i * total_elem_dofs + (nDofs_u + j)] += jac_usxx * weight_qp;

                    // J_uu: $\partial R_u / \partial u_j = 0$ (already initialized to zero)
                }
            } // end loop for test functions for u (R_u)

            // Iterate over test functions (i for rows of element matrix/vector)
            for (unsigned i = 0; i < nDofs_sxx; i++) { // For test function associated with 'sxx' (second equation)

                // Residual for 'sxx' (R_sxx): from $\int_\Omega (\nabla u \cdot \nabla v_{s_{xx}} - s_{xx} \cdot v_{s_{xx}}) d\Omega$
                real_num laplace_u_term_res = 0.;
                for (unsigned jdim = 0; jdim < dim_offset_grad; jdim++) {
                    laplace_u_term_res += unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + jdim) * gradSolu_u_gss[jdim];
                }
                unk_element_jac_res.res()[nDofs_u + i] += (laplace_u_term_res - solu_sxx_gss * unknowns_phi_dof_qp[1].phi(i)) * weight_qp;

                // Iterate over basis functions (j for columns of element matrix)
                for (unsigned j = 0; j < nDofs_u; j++) { // Loop for basis functions of u and sxx

                    // J_sxxu: $\partial R_{s_{xx}} / \partial u_j = \int_\Omega (\nabla \phi_j \cdot \nabla \phi_i) d\Omega$
                    // Row for R_sxx (nDofs_u + i), Column for u_j (j)
                    real_num jac_sxxu = 0.;
                    for (unsigned kdim = 0; kdim < dim_offset_grad; kdim++) {
                        jac_sxxu += unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + kdim) * unknowns_phi_dof_qp[0].phi_grad(j * dim_offset_grad + kdim);
                    }
                    unk_element_jac_res.jac()[(nDofs_u + i) * total_elem_dofs + j] += jac_sxxu * weight_qp;

                    // J_sxxsxx: $\partial R_{s_{xx}} / \partial s_{xx_j} = \int_\Omega (-\phi_j \cdot \phi_i) d\Omega$
                    // Row for R_sxx (nDofs_u + i), Column for sxx_j (nDofs_u + j)
                    real_num jac_sxxsxx = -unknowns_phi_dof_qp[1].phi(i) * unknowns_phi_dof_qp[1].phi(j);
                    unk_element_jac_res.jac()[(nDofs_u + i) * total_elem_dofs + (nDofs_u + j)] += jac_sxxsxx * weight_qp;
                }
            } // end loop for test functions for sxx (R_sxx)


        } // end gauss point loop

        // No need for unk_assemble_jac->compute_jacobian_outside_integration_loop as it's for AD.
        // Direct global assembly:
        // The final residual for FEMUS is typically -1 times the calculated residual.
        std::vector<double> Res_total(unk_element_jac_res.res().size());
        for (size_t k = 0; k < unk_element_jac_res.res().size(); ++k) {
            Res_total[k] = -unk_element_jac_res.res()[k];
        }

        RES->add_vector_blocked(Res_total, unk_element_jac_res.dof_map());
        KK->add_matrix_blocked(unk_element_jac_res.jac(), unk_element_jac_res.dof_map(), unk_element_jac_res.dof_map());

        // Optional: Print elemental algebra for debugging
        constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            // These variables are now correctly declared within this scope
            const unsigned nDofs_u_local = unknowns_local[0].num_elem_dofs();
            const unsigned nDofs_sxx_local = unknowns_local[1].num_elem_dofs();
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = {nDofs_u_local, nDofs_sxx_local}; // Helper for printing
            // Ensure adept::Stack is in scope if assemble_jacobian methods need it, though not directly used for AD here.
            // adept::Stack & stack_for_print = FemusInit::_adeptStack;
            assemble_jacobian<double,double>::print_element_jacobian(iel, unk_element_jac_res.jac(), Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, Res_total, Sol_n_el_dofs_Mat_vol, 10, 5);
        }
    } //end element loop for each process

    RES->close();
    KK->close();

    // ***************** END ASSEMBLY *******************
}

  }; // end class biharmonic_coupled_equation

} //end namespace karthik

#endif
