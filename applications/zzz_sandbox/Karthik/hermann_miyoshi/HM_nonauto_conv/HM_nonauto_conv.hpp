#ifndef __femus_biharmonic_HM_nonauto_conv_D_hpp__
#define __femus_biharmonic_HM_nonauto_conv_D_hpp__
#include <cassert>  // FIX: for runtime checks

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
 *  2) B_xy uses the symmetric mixed-gradient formula exactly as in the discrete form
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

    std::vector < std::vector < real_num_mov > > JacI_qp(space_dim);
    std::vector < std::vector < real_num_mov > > Jac_qp(dim);
    for (unsigned d = 0; d < dim; d++) Jac_qp[d].resize(space_dim);
    for (unsigned d = 0; d < space_dim; d++) JacI_qp[d].resize(dim);
    real_num_mov detJac_qp = 0.0;
    real_num_mov weight_qp = 0.0;

    unsigned xType = CONTINUOUS_BIQUADRATIC;
    CurrentElem < real_num_mov > geom_element(dim, msh);
    Phi < real_num_mov > geom_element_phi_dof_qp(dim);

    // --- unknowns (expect at least u,sxx,sxy,syy) ---
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();
    if (n_unknowns < 4) {
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
    std::vector < Phi < real_num > > unknowns_phi_dof_qp(n_unknowns, Phi< real_num >(dim));
    for (int u = 0; u < (int)n_unknowns; u++) {
        unknowns_local[u].initialize(dim, unknowns[u], ml_sol, mlPdeSys);
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
        unk_element_jac_res.res().assign(total_local_dofs, 0.0);
        unk_element_jac_res.jac().assign(total_local_dofs * total_local_dofs, 0.0);

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


        std::fill(unk_element_jac_res.res().begin(), unk_element_jac_res.res().end(), 0.0);
        std::fill(unk_element_jac_res.jac().begin(), unk_element_jac_res.jac().end(), 0.0);



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


    std::vector<real_num>& phi_u = unknowns_phi_dof_qp[idx_u].phi();
    std::vector<real_num>& gradphi_u = unknowns_phi_dof_qp[idx_u].phi_grad();
    std::vector<real_num>& phi_sxx = unknowns_phi_dof_qp[idx_sxx].phi();
    std::vector<real_num>& gradphi_sxx = unknowns_phi_dof_qp[idx_sxx].phi_grad();
    std::vector<real_num>& phi_sxy = unknowns_phi_dof_qp[idx_sxy].phi();
    std::vector<real_num>& gradphi_sxy = unknowns_phi_dof_qp[idx_sxy].phi_grad();
    std::vector<real_num>& phi_syy = unknowns_phi_dof_qp[idx_syy].phi();
    std::vector<real_num>& gradphi_syy = unknowns_phi_dof_qp[idx_syy].phi_grad();

            // interpolate values & gradients at qp
            real_num_mov u_val_g = 0.0;
            std::vector< real_num_mov > grad_u_g(dim, 0.0);
            for (unsigned a = 0; a < nDofs_u; ++a) {
                u_val_g +=  phi_u[a] *  unknowns_local[idx_u].elem_dofs()[a];
                for (unsigned d = 0; d < dim; ++d)
                    grad_u_g[d] +=  gradphi_u[a * dim + d] *  unknowns_local[idx_u].elem_dofs()[a];
            }

            real_num_mov sxx_val_g = 0.0;
            std::vector< real_num_mov > grad_sxx_g(dim, 0.0);
            for (unsigned a = 0; a < nDofs_sxx; ++a) {
                sxx_val_g +=  phi_sxx[a] *  unknowns_local[idx_sxx].elem_dofs()[a];
                for (unsigned d = 0; d < dim; ++d)
                    grad_sxx_g[d] +=  gradphi_sxx[a * dim + d] *  unknowns_local[idx_sxx].elem_dofs()[a];
            }

            real_num_mov sxy_val_g = 0.0;
            std::vector< real_num_mov > grad_sxy_g(dim, 0.0);
            for (unsigned a = 0; a < nDofs_sxy; ++a) {
                sxy_val_g +=  phi_sxy[a] *  unknowns_local[idx_sxy].elem_dofs()[a];
                for (unsigned d = 0; d < dim; ++d)
                    grad_sxy_g[d] +=  gradphi_sxy[a * dim + d] *  unknowns_local[idx_sxy].elem_dofs()[a];
            }

            real_num_mov syy_val_g = 0.0;
            std::vector< real_num_mov > grad_syy_g(dim, 0.0);
            for (unsigned a = 0; a < nDofs_syy; ++a) {
                syy_val_g +=  phi_syy[a] *  unknowns_local[idx_syy].elem_dofs()[a];
                for (unsigned d = 0; d < dim; ++d)
                    grad_syy_g[d] +=  gradphi_syy[a * dim + d] *  unknowns_local[idx_syy].elem_dofs()[a];
            }

            // physical coords at qp and f(x)
            std::vector< real_num_mov > x_gss(dim, 0.0);
            std::vector< std::vector< real_num_mov > >   & coords = geom_element.get_coords_at_dofs();
            const unsigned nGeomDofs = coords[0].size();
            for (unsigned a = 0; a < nGeomDofs; ++a) {
                const real_num_mov geom_phi =  geom_element_phi_dof_qp.phi()[a];
                for (unsigned d = 0; d < dim; ++d)
                    x_gss[d] +=  coords[d][a] * geom_phi;
            }
            const real_num_mov f_val =  source_functions[0]->value(x_gss);

            // compute div sigma at qp: divS = (∂x sxx + ∂y sxy,  ∂x sxy + ∂y syy)
            const real_num_mov divS_x = grad_sxx_g[0] + grad_sxy_g[1];
            const real_num_mov divS_y = grad_sxy_g[0] + grad_syy_g[1];

            // ---------------- assemble tau-equations (sxx, sxy, syy) ----------------
            // sxx-test: (sxx, τ) + (∇u, div τ) = 0  -> residual for sxx rows
            for (unsigned i = 0; i < nDofs_sxx; ++i) {
                const real_num phi_i =  phi_sxx[i];
                const real_num phix_i =  gradphi_sxx[i * dim + 0];
                real_num_mov R =  sxx_val_g *  phi_i
                                 + grad_u_g[0] *  phix_i;
                unk_element_jac_res.res()[ offset_sxx + i ] +=  ( R * weight_qp );

                if (assembleMatrix) {
                    // A_{xx,xx} mass block
                    for (unsigned j = 0; j < nDofs_sxx; ++j) {
                        const real_num val =  (  phi_sxx[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_sxx + i) * total_local_dofs + (offset_sxx + j) ] +=  ( val * weight_qp );
                    }
                    // B_{xx} block coupling to u (derivative of grad_u · phix_i)
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val =  (  gradphi_u[j * dim + 0] * phix_i );
                        unk_element_jac_res.jac()[ (offset_sxx + i) * total_local_dofs + (offset_u + j) ] +=  ( val * weight_qp );
                    }
                }
            }

            // sxy-test: (sxy, τ) + (∇u, div τ) = 0  -> residual for sxy rows
            for (unsigned i = 0; i < nDofs_sxy; ++i) {
                const real_num phi_i =  phi_sxy[i];
                const real_num phix_i =  gradphi_sxy[i * dim + 0];
                const real_num phiy_i =  gradphi_sxy[i * dim + 1];
                real_num_mov R = 2.0 * sxy_val_g *  phi_i
                                 + grad_u_g[0] *  phiy_i
                                 + grad_u_g[1] *  phix_i;
                unk_element_jac_res.res()[ offset_sxy + i ] +=  ( R * weight_qp );

                if (assembleMatrix) {
                    // 2 * A_{xy,xy} block (factor 2)
                    for (unsigned j = 0; j < nDofs_sxy; ++j) {
                        const real_num val =  ( 2.0 *  phi_sxy[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_sxy + i) * total_local_dofs + (offset_sxy + j) ] +=  ( val * weight_qp );
                    }
                    // B_{xy} coupling to u
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val =  (  gradphi_u[j * dim + 0] * phiy_i
                                                        +  gradphi_u[j * dim + 1] * phix_i );
                        unk_element_jac_res.jac()[ (offset_sxy + i) * total_local_dofs + (offset_u + j) ] +=  ( val * weight_qp );
                    }
                }
            }

            // syy-test: (syy, τ) + (∇u, div τ) = 0  -> residual for syy rows
            for (unsigned i = 0; i < nDofs_syy; ++i) {
                const real_num phi_i =  phi_syy[i];
                const real_num phiy_i =  gradphi_syy[i * dim + 1];
                real_num_mov R =  syy_val_g *  phi_i
                                 + grad_u_g[1] *  phiy_i;
                unk_element_jac_res.res()[ offset_syy + i ] +=  ( R * weight_qp );

                if (assembleMatrix) {
                    // A_{yy,yy} block
                    for (unsigned j = 0; j < nDofs_syy; ++j) {
                        const real_num val =  (  phi_syy[j] * phi_i );
                        unk_element_jac_res.jac()[ (offset_syy + i) * total_local_dofs + (offset_syy + j) ] +=  ( val * weight_qp );
                    }
                    // B_{yy} coupling to u
                    for (unsigned j = 0; j < nDofs_u; ++j) {
                        const real_num val =  (  gradphi_u[j * dim + 1] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_syy + i) * total_local_dofs + (offset_u + j) ] +=  ( val * weight_qp );
                    }
                }
            }

            // ---------------- assemble u-equation (top row) ----------------
            // u-test: (div σ, ∇v) - (f, v) = 0  -> residual for u rows (R_u)
            for (unsigned i = 0; i < nDofs_u; ++i) {
                const real_num phi_i =  phi_u[i];
                const real_num phix_i =  gradphi_u[i * dim + 0];
                const real_num phiy_i =  gradphi_u[i * dim + 1];

                real_num_mov R = divS_x *  phix_i + divS_y *  phiy_i +  f_val *  phi_i;
                unk_element_jac_res.res()[ offset_u + i ] +=  ( R *    weight_qp );

                if (assembleMatrix) {
                    // derivative wrt sxx_j: B_{xx}^T contribution (∂x phi_j * ∂x phi_i)
                    for (unsigned j = 0; j < nDofs_sxx; ++j) {
                        const real_num val =  (  gradphi_sxx[j * dim + 0] * phix_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_sxx + j) ] +=  ( val * weight_qp );
                    }
                    // derivative wrt sxy_j: B_{xy}^T contribution (∂y phi_j * ∂x phi_i + ∂x phi_j * ∂y phi_i)
                    for (unsigned j = 0; j < nDofs_sxy; ++j) {
                        const real_num val =  (  gradphi_sxy[j * dim + 1] * phix_i
                                                        +  gradphi_sxy[j * dim + 0] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_sxy + j) ] +=  ( val * weight_qp );
                    }
                    // derivative wrt syy_j: B_{yy}^T contribution (∂y phi_j * ∂y phi_i)
                    for (unsigned j = 0; j < nDofs_syy; ++j) {
                        const real_num val =  (  gradphi_syy[j * dim + 1] * phiy_i );
                        unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_syy + j) ] +=  ( val * weight_qp );
                    }
                }
            }

        } // end gauss loop

        // --- finalize local residual (FEMUS convention: negate) and assemble to global ---
        std::vector<double> Res_total( unk_element_jac_res.res().size() );
        for (size_t kk = 0; kk < unk_element_jac_res.res().size(); ++kk)
            Res_total[kk] =   (- unk_element_jac_res.res()[kk] );

        RES->add_vector_blocked(Res_total, unk_element_jac_res.dof_map());
        if (assembleMatrix) {
            KK->add_matrix_blocked( unk_element_jac_res.jac(), unk_element_jac_res.dof_map(), unk_element_jac_res.dof_map() );
        }


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






};






}

#endif
