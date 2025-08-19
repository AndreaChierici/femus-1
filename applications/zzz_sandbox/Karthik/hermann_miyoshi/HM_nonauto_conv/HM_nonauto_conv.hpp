#ifndef __femus_biharmonic_HM_nonauto_conv_D_hpp__
#define __femus_biharmonic_HM_nonauto_conv_D_hpp__

#include "FemusInit.hpp" //for the adept stack
#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"
#include "Assemble_unknown_jacres.hpp" // <-- ADDED THIS LINE
// // #include "ElementJacRes.hpp"

/**
 * AssembleHermannMiyoshiProblem
 *
 * Assembles the 4x4 block Hermann--Miyoshi system:
 *
 *   |  0      B_xx^T   B_xy^T   B_yy^T |   | u     |   = | -f |
 *   |  B_xx   A_xx_xx   0        0    |   | sxx   |     | 0  |
 *   |  B_xy    0      2 A_xy_xy   0    | * | sxy   |  =  | 0  |
 *   |  B_yy    0        0     A_yy_yy |   | syy   |     | 0  |
 *
 * where:
 *  - B_xx(i,j) = ∫ (∂_x phi_j) (∂_x psi_i) dΩ   (we store B^T in u-row assembly)
 *  - B_yy(i,j) = ∫ (∂_y phi_j) (∂_y psi_i) dΩ
 *  - B_xy(i,j) = ∫ (∂_y phi_j ∂_x psi_i + ∂_x phi_j ∂_y psi_i ) dΩ
 *  - A blocks are mass-like: ∫ phi_k^α phi_l^β dΩ  (factor 2 on A_xy_xy)
 *
 *  2) B_xy uses the symmetric mixed-gradient formula exactly as in your discrete form
 *  3) The sxy-sxy block uses the factor 2 multiplier.
 *
 */

using namespace femus;

namespace karthik {

class biharmonic_HM_nonauto_conv {

public:

//========= BOUNDARY_IMPLEMENTATION_U - BEGIN ==================

static void natural_loop_1dU(const MultiLevelProblem * ml_prob,
                             const Mesh * msh,
                             const MultiLevelSolution * ml_sol,
                             const unsigned iel,
                             CurrentElem < double > & geom_element,
                             const unsigned xType,
                             const std::string solname_u,
                             const unsigned solFEType_u,
                             std::vector< double > & Res) {

    double grad_u_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);
       geom_element.set_elem_center_bdry_3d();

       std::vector < double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet = ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_u.c_str(), grad_u_dot_n, face, 0.);

         if ( !(is_dirichlet) && (grad_u_dot_n != 0.) ) { //dirichlet == false and nonhomogeneous Neumann

             unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_u);
             for (unsigned i = 0; i < n_dofs_face; i++) {
                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);
                 Res[i_vol] += grad_u_dot_n;
             }
         }
       }
    }
}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dU(const MultiLevelProblem * ml_prob,
                               const Mesh * msh,
                               const MultiLevelSolution * ml_sol,
                               const unsigned iel,
                               CurrentElem < double > & geom_element,
                               const unsigned solType_coords,
                               const std::string solname_u,
                               const unsigned solFEType_u,
                               std::vector< double > & Res,
                               //-----------
                               std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > elem_all,
                               const unsigned dim,
                               const unsigned space_dim,
                               const unsigned max_size) {

    std::vector < std::vector < double > > JacI_iqp_bdry(space_dim);
    std::vector < std::vector < double > > Jac_iqp_bdry(dim-1);
    for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) { Jac_iqp_bdry[d].resize(space_dim); }
    for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }

    double detJac_iqp_bdry;
    double weight_iqp_bdry = 0.;
    std::vector <double> phi_u_bdry;
    std::vector <double> phi_u_x_bdry;
    phi_u_bdry.reserve(max_size);
    phi_u_x_bdry.reserve(max_size * space_dim);

    std::vector <double> phi_coords_bdry;
    std::vector <double> phi_coords_x_bdry;
    phi_coords_bdry.reserve(max_size);
    phi_coords_x_bdry.reserve(max_size * space_dim);

    double grad_u_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);
       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);
       std::vector < double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();
       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary
         unsigned int face = - (boundary_index + 1);
         bool is_dirichlet = ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_u.c_str(), grad_u_dot_n, face, 0.);

         if ( !(is_dirichlet) /* && (grad_u_dot_n != 0.)*/ ) { //dirichlet == false and nonhomogeneous Neumann
           unsigned n_dofs_face_u = msh->GetElementFaceDofNumber(iel, jface, solFEType_u);
           std::vector< double > grad_u_dot_n_at_dofs(n_dofs_face_u);
           for (unsigned i_bdry = 0; i_bdry < grad_u_dot_n_at_dofs.size(); i_bdry++) {
               std::vector<double> x_at_node(dim, 0.);
               for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];
               double grad_u_dot_n_at_dofs_temp = 0.;
               ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_u.c_str(), grad_u_dot_n_at_dofs_temp, face, 0.);
               grad_u_dot_n_at_dofs[i_bdry] = grad_u_dot_n_at_dofs_temp;
           }

           const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();
           for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {
               elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);
               weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];
               elem_all[ielGeom_bdry][solFEType_u ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_u_bdry, phi_u_x_bdry, boost::none, space_dim);
               elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry, boost::none, space_dim);

               double grad_u_dot_n_qp = 0.;
               for (unsigned i_bdry = 0; i_bdry < phi_u_bdry.size(); i_bdry ++) {
                   grad_u_dot_n_qp += grad_u_dot_n_at_dofs[i_bdry] * phi_u_bdry[i_bdry];
               }

               for (unsigned i_bdry = 0; i_bdry < n_dofs_face_u; i_bdry++) {
                   unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);
                   Res[i_vol] += weight_iqp_bdry * grad_u_dot_n_qp * phi_u_bdry[i_bdry];
               }
           }
         }
       }
    }
}

//========= BOUNDARY_IMPLEMENTATION_U - END ==================


//========= BOUNDARY_IMPLEMENTATION_Sxx - BEGIN ==================

static void natural_loop_1dV(const MultiLevelProblem * ml_prob,
                             const Mesh * msh,
                             const MultiLevelSolution * ml_sol,
                             const unsigned iel,
                             CurrentElem < double > & geom_element,
                             const unsigned xType,
                             const std::string solname_sxx,
                             const unsigned solFEType_sxx,
                             std::vector< double > & Res) {

    double grad_sxx_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);
       geom_element.set_elem_center_bdry_3d();

       std::vector < double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary
         unsigned int face = - (boundary_index + 1);
         bool is_dirichlet = ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxx.c_str(), grad_sxx_dot_n, face, 0.);

         if ( !(is_dirichlet) && (grad_sxx_dot_n != 0.) ) { //dirichlet == false and nonhomogeneous Neumann
           unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxx);
           for (unsigned i = 0; i < n_dofs_face; i++) {
               unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);
               Res[i_vol] += grad_sxx_dot_n;
           }
         }
       }
    }
}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dV(const MultiLevelProblem * ml_prob,
                               const Mesh * msh,
                               const MultiLevelSolution * ml_sol,
                               const unsigned iel,
                               CurrentElem < double > & geom_element,
                               const unsigned solType_coords,
                               const std::string solname_sxx,
                               const unsigned solFEType_sxx,
                               std::vector< double > & Res,
                               //-----------
                               std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > elem_all,
                               const unsigned dim,
                               const unsigned space_dim,
                               const unsigned max_size) {

    std::vector < std::vector < double > > JacI_iqp_bdry(space_dim);
    std::vector < std::vector < double > > Jac_iqp_bdry(dim-1);
    for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) { Jac_iqp_bdry[d].resize(space_dim); }
    for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }

    double detJac_iqp_bdry;
    double weight_iqp_bdry = 0.;
    std::vector <double> phi_sxx_bdry;
    std::vector <double> phi_sxx_x_bdry;
    phi_sxx_bdry.reserve(max_size);
    phi_sxx_x_bdry.reserve(max_size * space_dim);

    std::vector <double> phi_coords_bdry;
    std::vector <double> phi_coords_x_bdry;
    phi_coords_bdry.reserve(max_size);
    phi_coords_x_bdry.reserve(max_size * space_dim);

    double grad_sxx_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {
       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);
       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);
       std::vector < double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();
       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary
         unsigned int face = - (boundary_index + 1);
         bool is_dirichlet = ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxx.c_str(), grad_sxx_dot_n, face, 0.);

         if ( !(is_dirichlet) /* && (grad_u_dot_n != 0.)*/ ) { //dirichlet == false and nonhomogeneous Neumann
           unsigned n_dofs_face_sxx = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxx);
           std::vector< double > grad_sxx_dot_n_at_dofs(n_dofs_face_sxx);
           for (unsigned i_bdry = 0; i_bdry < grad_sxx_dot_n_at_dofs.size(); i_bdry++) {
               std::vector<double> x_at_node(dim, 0.);
               for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];
               double grad_sxx_dot_n_at_dofs_temp = 0.;
               ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_sxx.c_str(), grad_sxx_dot_n_at_dofs_temp, face, 0.);
               grad_sxx_dot_n_at_dofs[i_bdry] = grad_sxx_dot_n_at_dofs_temp;
           }

           const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();
           for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {
               elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);
               weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];
               elem_all[ielGeom_bdry][solFEType_sxx ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_sxx_bdry, phi_sxx_x_bdry, boost::none, space_dim);
               elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry, boost::none, space_dim);

               double grad_sxx_dot_n_qp = 0.;
               for (unsigned i_bdry = 0; i_bdry < phi_sxx_bdry.size(); i_bdry ++) {
                   grad_sxx_dot_n_qp += grad_sxx_dot_n_at_dofs[i_bdry] * phi_sxx_bdry[i_bdry];
               }

               for (unsigned i_bdry = 0; i_bdry < n_dofs_face_sxx; i_bdry++) {
                   unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);
                   Res[i_vol] += weight_iqp_bdry * grad_sxx_dot_n_qp * phi_sxx_bdry[i_bdry];
               }
           }
         }
       }
    }
}


//========= BOUNDARY_IMPLEMENTATION_V - END ==================
/*
template < class system_type, class real_num, class real_num_mov >
static void AssembleBilaplaceProblem(
    const std::vector < std::vector <  elem_type_templ_base<real_num, real_num_mov> * > > & elem_all,
    const std::vector < std::vector <  elem_type_templ_base<real_num_mov, real_num_mov> * > > & elem_all_for_domain,
    const std::vector<Gauss> & quad_rules,
    system_type * mlPdeSys,
    MultiLevelMesh * ml_mesh_in,
    MultiLevelSolution * ml_sol_in,
    const std::vector< Unknown > & unknowns,
    const std::vector< Math::Function< double > * > & source_functions) {

    const unsigned level = mlPdeSys->GetLevelToAssemble();
    const bool assembleMatrix = mlPdeSys->GetAssembleMatrix();

    Mesh* msh = ml_mesh_in->GetLevel(level);

    MultiLevelSolution* ml_sol = ml_sol_in;
    Solution* sol = ml_sol->GetSolutionLevel(level);

    LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
    SparseMatrix* KK = pdeSys->_KK;
    NumericVector* RES = pdeSys->_RES;

    const unsigned dim = msh->GetDimension();
    const unsigned iproc = msh->processor_id();

    RES->zero();
    if (assembleMatrix) KK->zero();

    constexpr unsigned int space_dim = 3;
    const unsigned int dim_offset_grad = dim;

    std::vector < std::vector < real_num_mov > > JacI_qp(space_dim);
    std::vector < std::vector < real_num_mov > > Jac_qp(dim);
    for (unsigned d = 0; d < dim; d++) { Jac_qp[d].resize(space_dim); }
    for (unsigned d = 0; d < space_dim; d++) { JacI_qp[d].resize(dim); }

    real_num_mov detJac_qp;
    real_num_mov weight_qp;

    unsigned xType = CONTINUOUS_BIQUADRATIC;

    CurrentElem < real_num_mov > geom_element(dim, msh);
    Phi < real_num_mov > geom_element_phi_dof_qp(dim_offset_grad);

    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();

    std::vector < UnknownLocal < real_num > > unknowns_local(n_unknowns);
    std::vector < Phi < real_num > > unknowns_phi_dof_qp(n_unknowns, Phi< real_num >(dim_offset_grad));

    for(int u = 0; u < n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
    }

    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);

    std::vector < unsigned int > unk_num_elem_dofs_interface(n_unknowns);

    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {

        geom_element.set_coords_at_dofs_and_geom_type(iel, xType);
        geom_element.set_elem_center_3d(iel, xType);
        const short unsigned ielGeom = geom_element.geom_type();

        for (unsigned u = 0; u < n_unknowns; u++) {
            unknowns_local[u].set_elem_dofs(iel, msh, sol);
        }

        unk_element_jac_res.set_loc_to_glob_map(iel, msh, pdeSys);
        unk_element_jac_res.res().assign(unk_element_jac_res.dof_map().size(), 0.0);
        unk_element_jac_res.jac().assign(unk_element_jac_res.dof_map().size() * unk_element_jac_res.dof_map().size(), 0.0);

        unsigned sum_unk_num_elem_dofs_interface = 0;
        for (unsigned u = 0; u < n_unknowns; u++) {
            unk_num_elem_dofs_interface[u] = unknowns_local[u].num_elem_dofs();
            sum_unk_num_elem_dofs_interface += unk_num_elem_dofs_interface[u];
        }

        for (unsigned ig = 0; ig < quad_rules[ielGeom].GetGaussPointsNumber(); ig++) {

            elem_all_for_domain[ielGeom][xType]->JacJacInv(geom_element.get_coords_at_dofs_3d(), ig, Jac_qp, JacI_qp, detJac_qp, space_dim);
            weight_qp = detJac_qp * quad_rules[ielGeom].GetGaussWeightsPointer()[ig];

            for (unsigned u = 0; u < n_unknowns; u++) {
                elem_all[ielGeom][unknowns_local[u].fe_type()]->shape_funcs_current_elem(ig, JacI_qp, unknowns_phi_dof_qp[u].phi(), unknowns_phi_dof_qp[u].phi_grad(), unknowns_phi_dof_qp[u].phi_hess(), space_dim);
            }
            elem_all_for_domain[ielGeom][xType]->shape_funcs_current_elem(ig, JacI_qp, geom_element_phi_dof_qp.phi(), geom_element_phi_dof_qp.phi_grad(), geom_element_phi_dof_qp.phi_hess(), space_dim);

            real_num solu_u_gss = 0.;
            std::vector < real_num > gradSolu_u_gss(dim_offset_grad, 0.);
            for (unsigned i = 0; i < unknowns_local[0].num_elem_dofs(); i++) {
                solu_u_gss += unknowns_phi_dof_qp[0].phi(i) * unknowns_local[0].elem_dofs()[i];
                for (unsigned jdim = 0; jdim < dim_offset_grad; jdim++) {
                    gradSolu_u_gss[jdim] += unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + jdim) * unknowns_local[0].elem_dofs()[i];
                }
            }

            real_num solu_sxx_gss = 0.;
            std::vector < real_num > gradSolu_sxx_gss(dim_offset_grad, 0.);
            for (unsigned i = 0; i < unknowns_local[1].num_elem_dofs(); i++) {
                solu_sxx_gss += unknowns_phi_dof_qp[1].phi(i) * unknowns_local[1].elem_dofs()[i];
                for (unsigned jdim = 0; jdim < dim_offset_grad; jdim++) {
                    gradSolu_sxx_gss[jdim] += unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + jdim) * unknowns_local[1].elem_dofs()[i];
                }
            }

            std::vector < double > x_gss(dim, 0.);
            for (unsigned i = 0; i < geom_element.get_coords_at_dofs()[0].size(); i++) {
                for (unsigned jdim = 0; jdim < x_gss.size(); jdim++) {
                    x_gss[jdim] += geom_element.get_coords_at_dofs(jdim,i) * geom_element_phi_dof_qp.phi(i);
                }
            }

            const unsigned nDofs_u = unknowns_local[0].num_elem_dofs();
            const unsigned nDofs_sxx = unknowns_local[1].num_elem_dofs();
            const unsigned total_elem_dofs = nDofs_u + nDofs_sxx;

            for (unsigned i = 0; i < nDofs_u; i++) {
                real_num laplace_sxx_term_res = 0.;
                for (unsigned jdim = 0; jdim < dim_offset_grad; jdim++) {
                    laplace_sxx_term_res += unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + jdim) * gradSolu_sxx_gss[jdim];
                }
                double f_source_term = source_functions[0]->laplacian(x_gss);
                unk_element_jac_res.res()[i] += (laplace_sxx_term_res - f_source_term * unknowns_phi_dof_qp[0].phi(i)) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        // J_usxx: ∂Ru / ∂sxx_j = ∫Ω (∇ϕj · ∇ϕi) dΩ
                        real_num jac_usxx = 0.;
                        for (unsigned kdim = 0; kdim < dim_offset_grad; kdim++) {
                            jac_usxx += unknowns_phi_dof_qp[0].phi_grad(i * dim_offset_grad + kdim) * unknowns_phi_dof_qp[1].phi_grad(j * dim_offset_grad + kdim);
                        }
                        unk_element_jac_res.jac()[i * total_elem_dofs + (nDofs_u + j)] += jac_usxx * weight_qp;
                    }
                }
            }

            for (unsigned i = 0; i < nDofs_sxx; i++) {
                real_num laplace_u_term_res = 0.;
                for (unsigned jdim = 0; jdim < dim_offset_grad; jdim++) {
                    laplace_u_term_res += unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + jdim) * gradSolu_u_gss[jdim];
                }
                unk_element_jac_res.res()[nDofs_u + i] += (laplace_u_term_res - solu_sxx_gss * unknowns_phi_dof_qp[1].phi(i)) * weight_qp;

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_u; j++) {
                        // J_sxxu: ∂Rsxx / ∂u_j = ∫Ω (∇ϕj · ∇ϕi) dΩ
                        real_num jac_sxxu = 0.;
                        for (unsigned kdim = 0; kdim < dim_offset_grad; kdim++) {
                            jac_sxxu += unknowns_phi_dof_qp[1].phi_grad(i * dim_offset_grad + kdim) * unknowns_phi_dof_qp[0].phi_grad(j * dim_offset_grad + kdim);
                        }
                        unk_element_jac_res.jac()[(nDofs_u + i) * total_elem_dofs + j] += jac_sxxu * weight_qp;

                        // J_sxxsxx: ∂Rsxx / ∂sxx_j = ∫Ω (-ϕj · ϕi) dΩ
                        real_num jac_sxxsxx = -unknowns_phi_dof_qp[1].phi(i) * unknowns_phi_dof_qp[1].phi(j);
                        unk_element_jac_res.jac()[(nDofs_u + i) * total_elem_dofs + (nDofs_u + j)] += jac_sxxsxx * weight_qp;
                    }
                }
            }
        }

        std::vector<double> Res_total(unk_element_jac_res.res().size());
        for (size_t k = 0; k < unk_element_jac_res.res().size(); ++k) {
            Res_total[k] = -unk_element_jac_res.res()[k];
        }

        RES->add_vector_blocked(Res_total, unk_element_jac_res.dof_map());
        if (assembleMatrix) {
            KK->add_matrix_blocked(unk_element_jac_res.jac(), unk_element_jac_res.dof_map(), unk_element_jac_res.dof_map());
        }

        constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            const unsigned nDofs_u_local = unknowns_local[0].num_elem_dofs();
            const unsigned nDofs_sxx_local = unknowns_local[1].num_elem_dofs();
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = {nDofs_u_local, nDofs_sxx_local};
            assemble_jacobian<double,double>::print_element_jacobian(iel, unk_element_jac_res.jac(), Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, Res_total, Sol_n_el_dofs_Mat_vol, 10, 5);
        }
    }

    RES->close();
    KK->close();
}*/

/*
template < class system_type, class real_num, class real_num_mov >
static void AssembleHermannMiyoshiProblem(
    const std::vector < std::vector <  elem_type_templ_base<real_num, real_num_mov> * > > & elem_all,
    const std::vector < std::vector <  elem_type_templ_base<real_num_mov, real_num_mov> * > > & elem_all_for_domain,
    const std::vector<Gauss> & quad_rules,
    system_type * mlPdeSys,
    MultiLevelMesh * ml_mesh_in,
    MultiLevelSolution * ml_sol_in,
    const std::vector< Unknown > & unknowns,
    const std::vector< Math::Function< double > * > & source_functions)  // source_functions[0] provides f(x)
{
    // --- Step 0: basic handles and checks ---
    const unsigned level = mlPdeSys->GetLevelToAssemble();
    const bool assembleMatrix = mlPdeSys->GetAssembleMatrix();

    Mesh* msh = ml_mesh_in->GetLevel(level);
    MultiLevelSolution* ml_sol = ml_sol_in;
    Solution* sol = ml_sol->GetSolutionLevel(level);

    LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
    SparseMatrix* KK = pdeSys->_KK;
    NumericVector* RES = pdeSys->_RES;

    const unsigned dim = msh->GetDimension();
    const unsigned iproc = msh->processor_id();

    RES->zero();
    if (assembleMatrix) KK->zero();

    // --- geometry / quadrature containers ---
    constexpr unsigned int space_dim = 2;                 // working in 2D
    const unsigned int dim_offset_grad = dim;             // gradient components

    std::vector < std::vector < real_num_mov > > JacI_qp(space_dim);
    std::vector < std::vector < real_num_mov > > Jac_qp(dim);
    for (unsigned d = 0; d < dim; d++) Jac_qp[d].resize(space_dim);
    for (unsigned d = 0; d < space_dim; d++) JacI_qp[d].resize(dim);
    real_num_mov detJac_qp = (real_num_mov)0.0;
    real_num_mov weight_qp = (real_num_mov)0.0;

    unsigned xType = CONTINUOUS_BIQUADRATIC; // geometry FE for coordinates
    CurrentElem < real_num_mov > geom_element(dim, msh);
    Phi < real_num_mov > geom_element_phi_dof_qp(dim_offset_grad);

    // --- unknowns: expect 4 unknowns (u, sxx, sxy, syy) ---
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();
    if (n_unknowns < 4u) {
        std::cerr << "AssembleHermannMiyoshiProblem: expected at least 4 unknowns (u,sxx,sxy,syy) but found " << n_unknowns << "\n";
        return;
    }

    // Map indices by name (robust if ordering changes)
    int idx_u = -1, idx_sxx = -1, idx_sxy = -1, idx_syy = -1;
    for (unsigned k = 0; k < unknowns.size(); ++k) {
        if (unknowns[k]._name == "u")   idx_u = (int)k;
        if (unknowns[k]._name == "sxx") idx_sxx = (int)k;
        if (unknowns[k]._name == "sxy") idx_sxy = (int)k;
        if (unknowns[k]._name == "syy") idx_syy = (int)k;
    }
    if (idx_u < 0 || idx_sxx < 0 || idx_sxy < 0 || idx_syy < 0) {
        std::cerr << "AssembleHermannMiyoshiProblem: unknown names must contain 'u','sxx','sxy','syy'\n";
        return;
    }

    // --- UnknownLocal + Phi containers ---
    std::vector < UnknownLocal < real_num > > unknowns_local(n_unknowns);
    std::vector < Phi < real_num > > unknowns_phi_dof_qp(n_unknowns, Phi< real_num >(dim_offset_grad));
    for (int u = 0; u < (int)n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
    }

    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);

    // --- element loop ---
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); ++iel) {

        // geometry for current element
        geom_element.set_coords_at_dofs_and_geom_type(iel, xType);
        geom_element.set_elem_center_3d(iel, xType);
        const short unsigned ielGeom = geom_element.geom_type();

        // set local dofs for all unknowns
        for (unsigned u = 0; u < n_unknowns; ++u) {
            unknowns_local[u].set_elem_dofs(iel, msh, sol);
        }

        // prepare local to global mapping and reset locals
        unk_element_jac_res.set_loc_to_glob_map(iel, msh, pdeSys);
        const unsigned total_local_dofs = unk_element_jac_res.dof_map().size();
        unk_element_jac_res.res().assign(total_local_dofs, (real_num)0.0);
        unk_element_jac_res.jac().assign(total_local_dofs * total_local_dofs, (real_num)0.0);

        // per-unknown local DOF counts
        std::vector<unsigned> unk_num_elem_dofs(n_unknowns);
        unsigned sum_unk_num_elem_dofs = 0;
        for (unsigned u = 0; u < n_unknowns; ++u) {
            unk_num_elem_dofs[u] = unknowns_local[u].num_elem_dofs();
            sum_unk_num_elem_dofs += unk_num_elem_dofs[u];
        }
        // local names for convenience
        const unsigned nDofs_u   = unk_num_elem_dofs[idx_u];
        const unsigned nDofs_sxx = unk_num_elem_dofs[idx_sxx];
        const unsigned nDofs_sxy = unk_num_elem_dofs[idx_sxy];
        const unsigned nDofs_syy = unk_num_elem_dofs[idx_syy];

        // --- Gauss loop ---
        const unsigned nGauss = quad_rules[ielGeom].GetGaussPointsNumber();
        for (unsigned ig = 0; ig < nGauss; ++ig) {

            // jacobian and weight
            elem_all_for_domain[ielGeom][xType]->JacJacInv(geom_element.get_coords_at_dofs_3d(), ig, Jac_qp, JacI_qp, detJac_qp, space_dim);
            weight_qp = detJac_qp * quad_rules[ielGeom].GetGaussWeightsPointer()[ig];

            // evaluate shape functions for every unknown at this qp
            for (unsigned u = 0; u < n_unknowns; ++u) {
                elem_all[ielGeom][unknowns_local[u].fe_type()]->shape_funcs_current_elem(
                    ig, JacI_qp,
                    unknowns_phi_dof_qp[u].phi(),
                    unknowns_phi_dof_qp[u].phi_grad(),
                    unknowns_phi_dof_qp[u].phi_hess(),
                    space_dim
                );
            }
            // geometry phi
            elem_all_for_domain[ielGeom][xType]->shape_funcs_current_elem(
                ig, JacI_qp,
                geom_element_phi_dof_qp.phi(),
                geom_element_phi_dof_qp.phi_grad(),
                geom_element_phi_dof_qp.phi_hess(),
                space_dim
            );

            // --- local references to shape arrays (for clarity) ---
            auto & phi_u       = unknowns_phi_dof_qp[idx_u].phi();
            auto & gradphi_u   = unknowns_phi_dof_qp[idx_u].phi_grad();
            auto & phi_sxx     = unknowns_phi_dof_qp[idx_sxx].phi();
            auto & gradphi_sxx = unknowns_phi_dof_qp[idx_sxx].phi_grad();
            auto & phi_sxy     = unknowns_phi_dof_qp[idx_sxy].phi();
            auto & gradphi_sxy = unknowns_phi_dof_qp[idx_sxy].phi_grad();
            auto & phi_syy     = unknowns_phi_dof_qp[idx_syy].phi();
            auto & gradphi_syy = unknowns_phi_dof_qp[idx_syy].phi_grad();

            // --- interpolate unknown values and their gradients at qp ---
            // u and grad u
            real_num_mov u_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_u_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_u; ++a) {
                const real_num phi_a = (real_num) phi_u[a];
                const real_num val_a = (real_num) unknowns_local[idx_u].elem_dofs()[a];
                u_val_g += (real_num_mov) phi_a * (real_num_mov) val_a;
                for (unsigned d = 0; d < dim_offset_grad; ++d)
                    grad_u_g[d] += (real_num_mov) gradphi_u[a * dim_offset_grad + d] * (real_num_mov) val_a;
            }
            // sxx
            real_num_mov sxx_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_sxx_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_sxx; ++a) {
                const real_num phi_a = (real_num) phi_sxx[a];
                const real_num val_a = (real_num) unknowns_local[idx_sxx].elem_dofs()[a];
                sxx_val_g += (real_num_mov) phi_a * (real_num_mov) val_a;
                for (unsigned d = 0; d < dim_offset_grad; ++d)
                    grad_sxx_g[d] += (real_num_mov) gradphi_sxx[a * dim_offset_grad + d] * (real_num_mov) val_a;
            }
            // sxy
            real_num_mov sxy_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_sxy_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_sxy; ++a) {
                const real_num phi_a = (real_num) phi_sxy[a];
                const real_num val_a = (real_num) unknowns_local[idx_sxy].elem_dofs()[a];
                sxy_val_g += (real_num_mov) phi_a * (real_num_mov) val_a;
                for (unsigned d = 0; d < dim_offset_grad; ++d)
                    grad_sxy_g[d] += (real_num_mov) gradphi_sxy[a * dim_offset_grad + d] * (real_num_mov) val_a;
            }
            // syy
            real_num_mov syy_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_syy_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_syy; ++a) {
                const real_num phi_a = (real_num) phi_syy[a];
                const real_num val_a = (real_num) unknowns_local[idx_syy].elem_dofs()[a];
                syy_val_g += (real_num_mov) phi_a * (real_num_mov) val_a;
                for (unsigned d = 0; d < dim_offset_grad; ++d)
                    grad_syy_g[d] += (real_num_mov) gradphi_syy[a * dim_offset_grad + d] * (real_num_mov) val_a;
            }

            // compute physical coordinates at qp for f(x)
            std::vector< real_num_mov > x_gss(dim, (real_num_mov)0.0);
            auto & coords = geom_element.get_coords_at_dofs();
            const unsigned nGeomDofs = coords[0].size();
            for (unsigned a = 0; a < nGeomDofs; ++a) {
                const real_num_mov geom_phi = (real_num_mov) geom_element_phi_dof_qp.phi()[a];
                for (unsigned d = 0; d < dim; ++d)
                    x_gss[d] += (real_num_mov) coords[d][a] * geom_phi;
            }
            const real_num_mov f_val = (real_num_mov) source_functions[0]->laplacian(x_gss);

            // ---------- compute div sigma at qp ----------
            // div σ = (∂x sxx + ∂y sxy,  ∂x sxy + ∂y syy)
            const real_num_mov divS_x = grad_sxx_g[0] + grad_sxy_g[1];
            const real_num_mov divS_y = grad_sxy_g[0] + grad_syy_g[1];

            // ----------------- Residuals and Jacobian contributions -----------------
            // (A) Tau-equations (σ-test) for sxx: index offset row = offset_sxx
            const unsigned offset_u   = 0;
            const unsigned offset_sxx = offset_u + nDofs_u;
            const unsigned offset_sxy = offset_sxx + nDofs_sxx;
            const unsigned offset_syy = offset_sxy + nDofs_sxy;

            // 1) sxx-test: (σ, τ) + (∇u, div τ) = 0
            for (unsigned i = 0; i < nDofs_sxx; ++i) {
                const real_num phi_i = (real_num) phi_sxx[i];
                const real_num phix_i = (real_num) gradphi_sxx[i * dim_offset_grad + 0];
                // div τ for τ with only xx-component = (∂x phi, 0)
                real_num_mov R = (real_num_mov) sxx_val_g * (real_num_mov) phi_i
                                 + grad_u_g[0] * (real_num_mov) phix_i;
                unk_element_jac_res.res()[ offset_sxx + i ] += (real_num) ( R * weight_qp );

                if (assembleMatrix) {
                    // derivative wrt sxx_j: (σ,τ) => ∫ φ_j * φ_i
                    for (unsigned j = 0; j < nDofs_sxx; ++j) {
                        const real_num val = (real_num) ( (real_num) phi_sxx[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_sxx + i) * total_local_dofs + (offset_sxx + j) ] += (real_num) ( val * weight_qp );
                    }
                    // derivative wrt u_j: (∇u, div τ) => ∫ ∇φ_j^u · div τ
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_u[j * dim_offset_grad + 0] * phix_i );
                        unk_element_jac_res.jac()[ (offset_sxx + i) * total_local_dofs + (offset_u + j) ] += (real_num) ( val * weight_qp );
                    }
                }
            }

            // 2) sxy-test: τ = [[0, phi],[phi,0]] -> div τ = (∂y phi, ∂x phi)
            for (unsigned i = 0; i < nDofs_sxy; ++i) {
                const real_num phi_i = (real_num) phi_sxy[i];
                const real_num phix_i = (real_num) gradphi_sxy[i * dim_offset_grad + 0];
                const real_num phiy_i = (real_num) gradphi_sxy[i * dim_offset_grad + 1];
                // (σ,τ) term uses sxy component, plus ∇u·divτ = ux*∂y phi + uy*∂x phi
                real_num_mov R = (real_num_mov) sxy_val_g * (real_num_mov) phi_i
                                 + grad_u_g[0] * (real_num_mov) phiy_i
                                 + grad_u_g[1] * (real_num_mov) phix_i;
                unk_element_jac_res.res()[ offset_sxy + i ] += (real_num) ( R * weight_qp );

                if (assembleMatrix) {
                    // wrt sxy_j
                    for (unsigned j = 0; j < nDofs_sxy; ++j) {
                        const real_num val = (real_num) ( (real_num) phi_sxy[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_sxy + i) * total_local_dofs + (offset_sxy + j) ] += (real_num) ( val * weight_qp );
                    }
                    // wrt u_j: derivative of ∇u·divτ
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_u[j * dim_offset_grad + 0] * phiy_i
                                                        + (real_num) gradphi_u[j * dim_offset_grad + 1] * phix_i );
                        unk_element_jac_res.jac()[ (offset_sxy + i) * total_local_dofs + (offset_u + j) ] += (real_num) ( val * weight_qp );
                    }
                }
            }

            // 3) syy-test: τ with only yy-component -> div τ = (0, ∂y phi)
            for (unsigned i = 0; i < nDofs_syy; ++i) {
                const real_num phi_i = (real_num) phi_syy[i];
                const real_num phiy_i = (real_num) gradphi_syy[i * dim_offset_grad + 1];
                real_num_mov R = (real_num_mov) syy_val_g * (real_num_mov) phi_i
                                 + grad_u_g[1] * (real_num_mov) phiy_i;
                unk_element_jac_res.res()[ offset_syy + i ] += (real_num) ( R * weight_qp );

                if (assembleMatrix) {
                    for (unsigned j = 0; j < nDofs_syy; ++j) {
                        const real_num val = (real_num) ( (real_num) phi_syy[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_syy + i) * total_local_dofs + (offset_syy + j) ] += (real_num) ( val * weight_qp );
                    }
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_u[j * dim_offset_grad + 1] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_syy + i) * total_local_dofs + (offset_u + j) ] += (real_num) ( val * weight_qp );
                    }
                }
            }

            // (B) u-equation: (div σ, ∇v) - (f, v) = 0
            for (unsigned i = 0; i < nDofs_u; ++i) {
                const real_num phi_i = (real_num) phi_u[i];
                const real_num phix_i = (real_num) gradphi_u[i * dim_offset_grad + 0];
                const real_num phiy_i = (real_num) gradphi_u[i * dim_offset_grad + 1];

                real_num_mov R = divS_x * (real_num_mov) phix_i + divS_y * (real_num_mov) phiy_i
                                 - (real_num_mov) f_val * (real_num_mov) phi_i;
                unk_element_jac_res.res()[ offset_u + i ] += (real_num) ( R * weight_qp );

                if (assembleMatrix) {
                    // derivative wrt sxx_j: contribution ∂x sxx * ∂x v -> ∫ ∂x φ_j^{sxx} * ∂x φ_i^{u}
                    for (unsigned j = 0; j < nDofs_sxx; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_sxx[j * dim_offset_grad + 0] * phix_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_sxx + j) ] += (real_num) ( val * weight_qp );
                    }
                    // wrt sxy_j: ∂y sxy * ∂x v  + ∂x sxy * ∂y v
                    for (unsigned j = 0; j < nDofs_sxy; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_sxy[j * dim_offset_grad + 1] * phix_i
                                                        + (real_num) gradphi_sxy[j * dim_offset_grad + 0] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_sxy + j) ] += (real_num) ( val * weight_qp );
                    }
                    // wrt syy_j: ∂y syy * ∂y v
                    for (unsigned j = 0; j < nDofs_syy; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_syy[j * dim_offset_grad + 1] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_syy + j) ] += (real_num) ( val * weight_qp );
                    }
                    // wrt u_j: no contribution (u appears only in τ-equations)
                }
            }

        } // end gauss loop

        // --- finalize local residual (FEMUS convention: negate) and assemble globally ---
        std::vector<double> Res_total( unk_element_jac_res.res().size() );
        for (size_t kk = 0; kk < unk_element_jac_res.res().size(); ++kk) {
            Res_total[kk] = - ( double ) ( unk_element_jac_res.res()[kk] );
        }
        RES->add_vector_blocked(Res_total, unk_element_jac_res.dof_map());

        if (assembleMatrix) {
            KK->add_matrix_blocked( unk_element_jac_res.jac(), unk_element_jac_res.dof_map(), unk_element_jac_res.dof_map() );
        }

        // optional local print (disabled by default)
        constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = { nDofs_u, nDofs_sxx, nDofs_sxy, nDofs_syy };
            assemble_jacobian<double,double>::print_element_jacobian(iel, unk_element_jac_res.jac(), Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, Res_total, Sol_n_el_dofs_Mat_vol, 10, 5);
        }
    } // end element loop

    RES->close();
    if (assembleMatrix) KK->close();
} // end AssembleHermannMiyoshiProblem
*/


template < class system_type, class real_num, class real_num_mov >
static void AssembleHermannMiyoshiProblem(
    const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> * > > & elem_all,
    const std::vector < std::vector < /*const*/ elem_type_templ_base<real_num_mov, real_num_mov> * > > & elem_all_for_domain,
    const std::vector<Gauss> & quad_rules,
    system_type * mlPdeSys,
    MultiLevelMesh * ml_mesh_in,
    MultiLevelSolution * ml_sol_in,
    const std::vector< Unknown > & unknowns,
    const std::vector< Math::Function< double > * > & source_functions)
{
    // --- basic handles ---
    const unsigned level = mlPdeSys->GetLevelToAssemble();
    const bool assembleMatrix = mlPdeSys->GetAssembleMatrix();

    Mesh* msh = ml_mesh_in->GetLevel(level);
    MultiLevelSolution* ml_sol = ml_sol_in;
    Solution* sol = ml_sol->GetSolutionLevel(level);

    LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
    SparseMatrix* KK = pdeSys->_KK;
    NumericVector* RES = pdeSys->_RES;

    const unsigned dim = msh->GetDimension();
    const unsigned iproc = msh->processor_id();

    RES->zero();
    if (assembleMatrix) KK->zero();

    // --- geometry and quadrature containers ---
    constexpr unsigned int space_dim = 2;
    const unsigned int dim_offset_grad = dim;

    std::vector < std::vector < real_num_mov > > JacI_qp(space_dim);
    std::vector < std::vector < real_num_mov > > Jac_qp(dim);
    for (unsigned d = 0; d < dim; d++) Jac_qp[d].resize(space_dim);
    for (unsigned d = 0; d < space_dim; d++) JacI_qp[d].resize(dim);
    real_num_mov detJac_qp = (real_num_mov)0.0;
    real_num_mov weight_qp = (real_num_mov)0.0;

    unsigned xType = CONTINUOUS_BIQUADRATIC;
    CurrentElem < real_num_mov > geom_element(dim, msh);
    Phi < real_num_mov > geom_element_phi_dof_qp(dim_offset_grad);

    // --- unknowns (expect at least u,sxx,sxy,syy) ---
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();
    if (n_unknowns < 4u) {
        std::cerr << "AssembleHermannMiyoshiProblem: expected at least 4 unknowns but found " << n_unknowns << "\n";
        return;
    }

    // map indices by name for robustness
    int idx_u = -1, idx_sxx = -1, idx_sxy = -1, idx_syy = -1;
    for (unsigned k = 0; k < unknowns.size(); ++k) {
        if (unknowns[k]._name == "u")   idx_u = (int)k;
        if (unknowns[k]._name == "sxx") idx_sxx = (int)k;
        if (unknowns[k]._name == "sxy") idx_sxy = (int)k;
        if (unknowns[k]._name == "syy") idx_syy = (int)k;
    }
    if (idx_u < 0 || idx_sxx < 0 || idx_sxy < 0 || idx_syy < 0) {
        std::cerr << "AssembleHermannMiyoshiProblem: unknown names must contain 'u','sxx','sxy','syy'\n";
        return;
    }

    // --- UnknownLocal + Phi ---
    std::vector < UnknownLocal < real_num > > unknowns_local(n_unknowns);
    std::vector < Phi < real_num > > unknowns_phi_dof_qp(n_unknowns, Phi< real_num >(dim_offset_grad));
    for (int u = 0; u < (int)n_unknowns; u++) {
        unknowns_local[u].initialize(dim_offset_grad, unknowns[u], ml_sol, mlPdeSys);
    }

    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);

    // --- element loop ---
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); ++iel) {

        geom_element.set_coords_at_dofs_and_geom_type(iel, xType);
        geom_element.set_elem_center_3d(iel, xType);
        const short unsigned ielGeom = geom_element.geom_type();

        // set local dofs for all unknowns
        for (unsigned u = 0; u < n_unknowns; ++u) {
            unknowns_local[u].set_elem_dofs(iel, msh, sol);
        }

        // prepare local to global mapping and reset local arrays
        unk_element_jac_res.set_loc_to_glob_map(iel, msh, pdeSys);
        const unsigned total_local_dofs = unk_element_jac_res.dof_map().size();
        unk_element_jac_res.res().assign(total_local_dofs, (real_num)0.0);
        unk_element_jac_res.jac().assign(total_local_dofs * total_local_dofs, (real_num)0.0);

        // per-unknown local DOF counts
        std::vector<unsigned> unk_num_elem_dofs(n_unknowns);
        unsigned sum_unk_num_elem_dofs = 0;
        for (unsigned u = 0; u < n_unknowns; ++u) {
            unk_num_elem_dofs[u] = unknowns_local[u].num_elem_dofs();
            sum_unk_num_elem_dofs += unk_num_elem_dofs[u];
        }

        // local sizes
        const unsigned nDofs_u   = unk_num_elem_dofs[idx_u];
        const unsigned nDofs_sxx = unk_num_elem_dofs[idx_sxx];
        const unsigned nDofs_sxy = unk_num_elem_dofs[idx_sxy];
        const unsigned nDofs_syy = unk_num_elem_dofs[idx_syy];

        // offsets in the local vector (unknown ordering u, sxx, sxy, syy)
        const unsigned offset_u   = 0;
        const unsigned offset_sxx = offset_u + nDofs_u;
        const unsigned offset_sxy = offset_sxx + nDofs_sxx;
        const unsigned offset_syy = offset_sxy + nDofs_sxy;

        // Gauss loop
        const unsigned nGauss = quad_rules[ielGeom].GetGaussPointsNumber();
        for (unsigned ig = 0; ig < nGauss; ++ig) {

            elem_all_for_domain[ielGeom][xType]->JacJacInv(
                geom_element.get_coords_at_dofs_3d(), ig, Jac_qp, JacI_qp, detJac_qp, space_dim);
            weight_qp = detJac_qp * quad_rules[ielGeom].GetGaussWeightsPointer()[ig];

            // shape functions for each unknown at this qp
            for (unsigned u = 0; u < n_unknowns; ++u) {
                elem_all[ielGeom][unknowns_local[u].fe_type()]->shape_funcs_current_elem(
                    ig, JacI_qp,
                    unknowns_phi_dof_qp[u].phi(),
                    unknowns_phi_dof_qp[u].phi_grad(),
                    unknowns_phi_dof_qp[u].phi_hess(),
                    space_dim
                );
            }
            // geometry phi
            elem_all_for_domain[ielGeom][xType]->shape_funcs_current_elem(
                ig, JacI_qp,
                geom_element_phi_dof_qp.phi(),
                geom_element_phi_dof_qp.phi_grad(),
                geom_element_phi_dof_qp.phi_hess(),
                space_dim
            );

            // local references
            auto & phi_u       = unknowns_phi_dof_qp[idx_u].phi();
            auto & gradphi_u   = unknowns_phi_dof_qp[idx_u].phi_grad();
            auto & phi_sxx     = unknowns_phi_dof_qp[idx_sxx].phi();
            auto & gradphi_sxx = unknowns_phi_dof_qp[idx_sxx].phi_grad();
            auto & phi_sxy     = unknowns_phi_dof_qp[idx_sxy].phi();
            auto & gradphi_sxy = unknowns_phi_dof_qp[idx_sxy].phi_grad();
            auto & phi_syy     = unknowns_phi_dof_qp[idx_syy].phi();
            auto & gradphi_syy = unknowns_phi_dof_qp[idx_syy].phi_grad();

            // interpolate values & gradients at qp
            real_num_mov u_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_u_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_u; ++a) {
                u_val_g += (real_num_mov) phi_u[a] * (real_num_mov) unknowns_local[idx_u].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d)
                    grad_u_g[d] += (real_num_mov) gradphi_u[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[idx_u].elem_dofs()[a];
            }

            real_num_mov sxx_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_sxx_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_sxx; ++a) {
                sxx_val_g += (real_num_mov) phi_sxx[a] * (real_num_mov) unknowns_local[idx_sxx].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d)
                    grad_sxx_g[d] += (real_num_mov) gradphi_sxx[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[idx_sxx].elem_dofs()[a];
            }

            real_num_mov sxy_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_sxy_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_sxy; ++a) {
                sxy_val_g += (real_num_mov) phi_sxy[a] * (real_num_mov) unknowns_local[idx_sxy].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d)
                    grad_sxy_g[d] += (real_num_mov) gradphi_sxy[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[idx_sxy].elem_dofs()[a];
            }

            real_num_mov syy_val_g = (real_num_mov)0.0;
            std::vector< real_num_mov > grad_syy_g(dim_offset_grad, (real_num_mov)0.0);
            for (unsigned a = 0; a < nDofs_syy; ++a) {
                syy_val_g += (real_num_mov) phi_syy[a] * (real_num_mov) unknowns_local[idx_syy].elem_dofs()[a];
                for (unsigned d = 0; d < dim_offset_grad; ++d)
                    grad_syy_g[d] += (real_num_mov) gradphi_syy[a * dim_offset_grad + d] * (real_num_mov) unknowns_local[idx_syy].elem_dofs()[a];
            }

            // physical coords at qp and f(x)
            std::vector< real_num_mov > x_gss(dim, (real_num_mov)0.0);
            auto & coords = geom_element.get_coords_at_dofs();
            const unsigned nGeomDofs = coords[0].size();
            for (unsigned a = 0; a < nGeomDofs; ++a) {
                const real_num_mov geom_phi = (real_num_mov) geom_element_phi_dof_qp.phi()[a];
                for (unsigned d = 0; d < dim; ++d)
                    x_gss[d] += (real_num_mov) coords[d][a] * geom_phi;
            }
            const real_num_mov f_val = (real_num_mov) source_functions[0]->value(x_gss);

            // compute div sigma at qp: divS = (∂x sxx + ∂y sxy,  ∂x sxy + ∂y syy)
            const real_num_mov divS_x = grad_sxx_g[0] + grad_sxy_g[1];
            const real_num_mov divS_y = grad_sxy_g[0] + grad_syy_g[1];

            // ---------------- assemble tau-equations (sxx, sxy, syy) ----------------
            // sxx-test: (sxx, τ) + (∇u, div τ) = 0  -> residual for sxx rows
            for (unsigned i = 0; i < nDofs_sxx; ++i) {
                const real_num phi_i = (real_num) phi_sxx[i];
                const real_num phix_i = (real_num) gradphi_sxx[i * dim_offset_grad + 0];
                real_num_mov R = (real_num_mov) sxx_val_g * (real_num_mov) phi_i
                                 + grad_u_g[0] * (real_num_mov) phix_i;
                unk_element_jac_res.res()[ offset_sxx + i ] += (real_num) ( R * weight_qp );

                if (assembleMatrix) {
                    // A_{xx,xx} mass block
                    for (unsigned j = 0; j < nDofs_sxx; ++j) {
                        const real_num val = (real_num) ( (real_num) phi_sxx[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_sxx + i) * total_local_dofs + (offset_sxx + j) ] += (real_num) ( val * weight_qp );
                    }
                    // B_{xx} block coupling to u (derivative of grad_u · phix_i)
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_u[j * dim_offset_grad + 0] * phix_i );
                        unk_element_jac_res.jac()[ (offset_sxx + i) * total_local_dofs + (offset_u + j) ] += (real_num) ( val * weight_qp );
                    }
                }
            }

            // sxy-test: (sxy, τ) + (∇u, div τ) = 0  -> residual for sxy rows
            for (unsigned i = 0; i < nDofs_sxy; ++i) {
                const real_num phi_i = (real_num) phi_sxy[i];
                const real_num phix_i = (real_num) gradphi_sxy[i * dim_offset_grad + 0];
                const real_num phiy_i = (real_num) gradphi_sxy[i * dim_offset_grad + 1];
                real_num_mov R = (real_num_mov) sxy_val_g * (real_num_mov) phi_i
                                 + grad_u_g[0] * (real_num_mov) phiy_i
                                 + grad_u_g[1] * (real_num_mov) phix_i;
                unk_element_jac_res.res()[ offset_sxy + i ] += (real_num) ( R * weight_qp );

                if (assembleMatrix) {
                    // 2 * A_{xy,xy} block (factor 2)
                    for (unsigned j = 0; j < nDofs_sxy; ++j) {
                        const real_num val = (real_num) ( 2.0 * (real_num) phi_sxy[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_sxy + i) * total_local_dofs + (offset_sxy + j) ] += (real_num) ( val * weight_qp );
                    }
                    // B_{xy} coupling to u
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_u[j * dim_offset_grad + 0] * phiy_i
                                                        + (real_num) gradphi_u[j * dim_offset_grad + 1] * phix_i );
                        unk_element_jac_res.jac()[ (offset_sxy + i) * total_local_dofs + (offset_u + j) ] += (real_num) ( val * weight_qp );
                    }
                }
            }

            // syy-test: (syy, τ) + (∇u, div τ) = 0  -> residual for syy rows
            for (unsigned i = 0; i < nDofs_syy; ++i) {
                const real_num phi_i = (real_num) phi_syy[i];
                const real_num phiy_i = (real_num) gradphi_syy[i * dim_offset_grad + 1];
                real_num_mov R = (real_num_mov) syy_val_g * (real_num_mov) phi_i
                                 + grad_u_g[1] * (real_num_mov) phiy_i;
                unk_element_jac_res.res()[ offset_syy + i ] += (real_num) ( R * weight_qp );

                if (assembleMatrix) {
                    // A_{yy,yy} block
                    for (unsigned j = 0; j < nDofs_syy; ++j) {
                        const real_num val = (real_num) ( (real_num) phi_syy[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_syy + i) * total_local_dofs + (offset_syy + j) ] += (real_num) ( val * weight_qp );
                    }
                    // B_{yy} coupling to u
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_u[j * dim_offset_grad + 1] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_syy + i) * total_local_dofs + (offset_u + j) ] += (real_num) ( val * weight_qp );
                    }
                }
            }

            // ---------------- assemble u-equation (top row) ----------------
            // u-test: (div σ, ∇v) - (f, v) = 0  -> residual for u rows (R_u)
            for (unsigned i = 0; i < nDofs_u; ++i) {
                const real_num phi_i = (real_num) phi_u[i];
                const real_num phix_i = (real_num) gradphi_u[i * dim_offset_grad + 0];
                const real_num phiy_i = (real_num) gradphi_u[i * dim_offset_grad + 1];

                real_num_mov R = divS_x * (real_num_mov) phix_i + divS_y * (real_num_mov) phiy_i - (real_num_mov) f_val * (real_num_mov) phi_i;
                unk_element_jac_res.res()[ offset_u + i ] += (real_num) ( R * weight_qp );

                if (assembleMatrix) {
                    // derivative wrt sxx_j: B_{xx}^T contribution (∂x phi_j * ∂x phi_i)
                    for (unsigned j = 0; j < nDofs_sxx; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_sxx[j * dim_offset_grad + 0] * phix_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_sxx + j) ] += (real_num) ( val * weight_qp );
                    }
                    // derivative wrt sxy_j: B_{xy}^T contribution (∂y phi_j * ∂x phi_i + ∂x phi_j * ∂y phi_i)
                    for (unsigned j = 0; j < nDofs_sxy; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_sxy[j * dim_offset_grad + 1] * phix_i
                                                        + (real_num) gradphi_sxy[j * dim_offset_grad + 0] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_sxy + j) ] += (real_num) ( val * weight_qp );
                    }
                    // derivative wrt syy_j: B_{yy}^T contribution (∂y phi_j * ∂y phi_i)
                    for (unsigned j = 0; j < nDofs_syy; ++j) {
                        const real_num val = (real_num) ( (real_num) gradphi_syy[j * dim_offset_grad + 1] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_syy + j) ] += (real_num) ( val * weight_qp );
                    }
                }
            }

        } // end gauss loop

        // --- finalize local residual (FEMUS convention: negate) and assemble to global ---
        std::vector<double> Res_total( unk_element_jac_res.res().size() );
        for (size_t kk = 0; kk < unk_element_jac_res.res().size(); ++kk)
            Res_total[kk] = - ( double ) ( unk_element_jac_res.res()[kk] );

        RES->add_vector_blocked(Res_total, unk_element_jac_res.dof_map());
        if (assembleMatrix) {
            KK->add_matrix_blocked( unk_element_jac_res.jac(), unk_element_jac_res.dof_map(), unk_element_jac_res.dof_map() );
        }


         constexpr bool print_algebra_local = true;
        if (print_algebra_local) {
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = { nDofs_u, nDofs_sxx, nDofs_sxy, nDofs_syy };
            assemble_jacobian<double,double>::print_element_jacobian(iel, unk_element_jac_res.jac(), Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double,double>::print_element_residual(iel, Res_total, Sol_n_el_dofs_Mat_vol, 10, 5);
        }


    } // end element loop

    RES->close();
    if (assembleMatrix) KK->close();
} // end AssembleHermannMiyoshiProblem






};






}

#endif
