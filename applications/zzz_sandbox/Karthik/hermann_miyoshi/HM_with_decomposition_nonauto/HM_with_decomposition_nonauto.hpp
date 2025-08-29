#ifndef __femus_biharmonic_HM_D_hpp__
#define __femus_biharmonic_HM_D_hpp__
 
#include "FemusInit.hpp"  //for the adept stack

#include "MultiLevelProblem.hpp"
#include "MultiLevelMesh.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "LinearEquationSolver.hpp"
#include "NumericVector.hpp"
#include "SparseMatrix.hpp"
#include "Assemble_jacobian.hpp"
#include "Assemble_unknown_jacres.hpp" // <-- ADDED THIS LINE

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
 * The system is assembled according to the matrix formulation:
 * [ M   B^T    0     0  ] [W]   [   0   ]
 * [ B    0    ν1C1  ν1C2] [U] = [-ν2F   ]
 * [ 0   C1^T   M     0  ] [S1]  [   0   ]
 * [ 0   C2^T   0     M  ] [S2]  [   0   ]
 *
 * using automatic differentiation
 **/

using namespace femus;


namespace karthik {
  
  class biharmonic_HM_with_decomposition_nonauto {
    
  public:




//========= BOUNDARY_IMPLEMENTATION_U - BEGIN ==================

static void natural_loop_1dU(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_u,
                     const unsigned solFEType_u,
                     std::vector< double > & Res
                    ) {

     double grad_u_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_u.c_str(), grad_u_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_u_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann

                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_u);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_u_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dU(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_u,
                       const unsigned solFEType_u,
                       std::vector< double > & Res,
                       //-----------
                       std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
                       const unsigned dim,
                       const unsigned space_dim,
                       const unsigned max_size
                    ) {


    /// @todo - should put these outside the iel loop --
    std::vector < std::vector < double > >  JacI_iqp_bdry(space_dim);
     std::vector < std::vector < double > >  Jac_iqp_bdry(dim-1);
    for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) {   Jac_iqp_bdry[d].resize(space_dim); }
    for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }




  double detJac_iqp_bdry;
  double weight_iqp_bdry = 0.;
// ---
  //boundary state shape functions
  std::vector <double> phi_u_bdry;
  std::vector <double> phi_u_x_bdry;

  phi_u_bdry.reserve(max_size);
  phi_u_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_u_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_u.c_str(), grad_u_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_u = msh->GetElementFaceDofNumber(iel, jface, solFEType_u);

// dof-based - BEGIN
     std::vector< double > grad_u_dot_n_at_dofs(n_dofs_face_u);


    for (unsigned i_bdry = 0; i_bdry < grad_u_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_u_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_u.c_str(), grad_u_dot_n_at_dofs_temp, face, 0.);
     grad_u_dot_n_at_dofs[i_bdry] = grad_u_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);
//      elem_all[ielGeom_bdry][solType_coords]->compute_normal(Jac_iqp_bdry, normal);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_u ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_u_bdry, phi_u_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_u_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_u_bdry.size(); i_bdry ++) {
           grad_u_dot_n_qp +=  grad_u_dot_n_at_dofs[i_bdry] * phi_u_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_u; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_u_dot_n_qp /*grad_u_dot_n*/  * phi_u_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_U - END ==================


//========= BOUNDARY_IMPLEMENTATION_V - BEGIN ==================

static void natural_loop_1dV(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_v,
                     const unsigned solFEType_v,
                     std::vector< double > & Res
                    ) {

     double grad_v_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_v.c_str(), grad_v_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_v_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_v);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_v_dot_n /* * phi[node] = 1. */;

                         }

                    }

              }

    }

}


template < class real_num, class real_num_mov >
static void natural_loop_2d3dV(const MultiLevelProblem *    ml_prob,
                       const Mesh *                    msh,
                       const MultiLevelSolution *    ml_sol,
                       const unsigned iel,
                       CurrentElem < double > & geom_element,
                       const unsigned solType_coords,
                       const std::string solname_v,
                       const unsigned solFEType_v,
                       std::vector< double > & Res,
                       //-----------
                       std::vector < std::vector < /*const*/ elem_type_templ_base<real_num, real_num_mov> *  > >  elem_all,
                       const unsigned dim,
                       const unsigned space_dim,
                       const unsigned max_size
                    ) {


    /// @todo - should put these outside the iel loop --
    std::vector < std::vector < double > >  JacI_iqp_bdry(space_dim);
     std::vector < std::vector < double > >  Jac_iqp_bdry(dim-1);
    for (unsigned d = 0; d < Jac_iqp_bdry.size(); d++) {   Jac_iqp_bdry[d].resize(space_dim); }
    for (unsigned d = 0; d < JacI_iqp_bdry.size(); d++) { JacI_iqp_bdry[d].resize(dim-1); }




  double detJac_iqp_bdry;
  double weight_iqp_bdry = 0.;
// ---
  //boundary state shape functions
  std::vector <double> phi_v_bdry;
  std::vector <double> phi_v_x_bdry;

  phi_v_bdry.reserve(max_size);
  phi_v_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_v_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_v.c_str(), grad_v_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_v = msh->GetElementFaceDofNumber(iel, jface, solFEType_v);

// dof-based - BEGIN
     std::vector< double > grad_v_dot_n_at_dofs(n_dofs_face_v);


    for (unsigned i_bdry = 0; i_bdry < grad_v_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_v_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_v.c_str(), grad_v_dot_n_at_dofs_temp, face, 0.);
     grad_v_dot_n_at_dofs[i_bdry] = grad_v_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_v ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_v_bdry, phi_v_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_v_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_v_bdry.size(); i_bdry ++) {
           grad_v_dot_n_qp +=  grad_v_dot_n_at_dofs[i_bdry] * phi_v_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_v; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_v_dot_n_qp /*grad_u_dot_n*/  * phi_v_bdry[i_bdry];

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

    // physical/geometry containers
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

    // --- unknowns (expect at least u,v,s1,s2) ---
    const unsigned int n_unknowns = mlPdeSys->GetSolPdeIndex().size();
    if (n_unknowns < 4) {
        std::cerr << "AssembleHermannMiyoshiProblem: expected at least 4 unknowns but found " << n_unknowns << "\n";
        return;
    }

    int idx_u = -1, idx_v = -1, idx_s1 = -1, idx_s2 = -1;
    for (unsigned k = 0; k < unknowns.size(); ++k) {
        if (unknowns[k]._name == "u")   idx_u = (int)k;
        if (unknowns[k]._name == "v")   idx_v = (int)k;
        if (unknowns[k]._name == "s1")  idx_s1 = (int)k;
        if (unknowns[k]._name == "s2")  idx_s2 = (int)k;
    }
    if (idx_u < 0 || idx_v < 0 || idx_s1 < 0 || idx_s2 < 0) {
        std::cerr << "AssembleHermannMiyoshiProblem: unknown names must contain 'u','v','s1','s2'\n";
        return;
    }

    // --- UnknownLocal + Phi ---
    std::vector < UnknownLocal < real_num > > unknowns_local(n_unknowns);
    std::vector < Phi < real_num > > unknowns_phi_dof_qp(n_unknowns, Phi< real_num >(dim));
    for (int u = 0; u < (int)n_unknowns; u++) {
        unknowns_local[u].initialize(dim, unknowns[u], ml_sol, mlPdeSys);
    }

    ElementJacRes < real_num > unk_element_jac_res(dim, unknowns_local);

    // constitutive / params (same as your AD code)
    double nu = 0.4;
    double nu1 = (4.0 * (1.0 - nu)) / (1.0 + nu);
    double nu2 = 2.0 / (1.0 + nu);

    // --- element loop ---
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); ++iel) {

        geom_element.set_coords_at_dofs_and_geom_type(iel, xType);
        geom_element.set_elem_center_3d(iel, xType);
        const short unsigned ielGeom = geom_element.geom_type();

        // set local dofs for all unknowns
        for (unsigned u = 0; u < n_unknowns; ++u) {
            unknowns_local[u].set_elem_dofs(iel, msh, sol);
        }

        // local-to-global mapping & reset
        unk_element_jac_res.set_loc_to_glob_map(iel, msh, pdeSys);
        const unsigned total_local_dofs = unk_element_jac_res.dof_map().size();
        unk_element_jac_res.res().assign(total_local_dofs, 0.0);
        unk_element_jac_res.jac().assign(total_local_dofs * total_local_dofs, 0.0);

        // counts & offsets
        std::vector<unsigned> unk_num_elem_dofs(n_unknowns);
        unsigned sum_unk_num_elem_dofs = 0;
        for (unsigned u = 0; u < n_unknowns; ++u) {
            unk_num_elem_dofs[u] = unknowns_local[u].num_elem_dofs();
            sum_unk_num_elem_dofs += unk_num_elem_dofs[u];
        }

        const unsigned nDofs_u  = unk_num_elem_dofs[idx_u];
        const unsigned nDofs_v  = unk_num_elem_dofs[idx_v];
        const unsigned nDofs_s1 = unk_num_elem_dofs[idx_s1];
        const unsigned nDofs_s2 = unk_num_elem_dofs[idx_s2];

        const unsigned offset_u  = 0;
        const unsigned offset_v  = offset_u + nDofs_u;
        const unsigned offset_s1 = offset_v + nDofs_v;
        const unsigned offset_s2 = offset_s1 + nDofs_s1;

        // temporary local RHS vector (will accumulate -nu2*F for the v-row)
        std::vector<double> rhs_local(total_local_dofs, 0.0);

        // --- GAUSS: assemble element matrix blocks into unk_element_jac_res.jac() ---
        const unsigned nGauss = quad_rules[ielGeom].GetGaussPointsNumber();
        for (unsigned ig = 0; ig < nGauss; ++ig) {

            // geometry & weight
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

            // references to phis/grads
            std::vector<real_num>& phi_u = unknowns_phi_dof_qp[idx_u].phi();
            std::vector<real_num>& gradphi_u = unknowns_phi_dof_qp[idx_u].phi_grad();
            std::vector<real_num>& phi_v = unknowns_phi_dof_qp[idx_v].phi();
            std::vector<real_num>& gradphi_v = unknowns_phi_dof_qp[idx_v].phi_grad();
            std::vector<real_num>& phi_s1 = unknowns_phi_dof_qp[idx_s1].phi();
            std::vector<real_num>& gradphi_s1 = unknowns_phi_dof_qp[idx_s1].phi_grad();
            std::vector<real_num>& phi_s2 = unknowns_phi_dof_qp[idx_s2].phi();
            std::vector<real_num>& gradphi_s2 = unknowns_phi_dof_qp[idx_s2].phi_grad();

            // compute f at this qp (physical coords)
            std::vector< real_num_mov > x_gss(dim, 0.0);
            std::vector< std::vector< real_num_mov > > & coords = geom_element.get_coords_at_dofs();
            const unsigned nGeomDofs = coords[0].size();
            for (unsigned a = 0; a < nGeomDofs; ++a) {
                const real_num_mov geom_phi = geom_element_phi_dof_qp.phi()[a];
                for (unsigned d = 0; d < dim; ++d)
                    x_gss[d] += coords[d][a] * geom_phi;
            }
            const real_num_mov f_val = source_functions[0]->laplacian(x_gss);

            // ---- assemble block integrals at this quadrature point ----
            // 1) M_uu (u-u mass)
            for (unsigned i = 0; i < nDofs_u; ++i) {
                const real_num phi_i = phi_u[i];
                for (unsigned j = 0; j < nDofs_u; ++j) {
                    const real_num val = ( phi_u[j] * phi_i) * weight_qp;
                    unk_element_jac_res.jac()[ (offset_u + i) * total_local_dofs + (offset_u + j) ] += val;
                }
            }

            // 2) Mass for v, s1, s2 (these were already present in your code) - keep them

            for (unsigned i = 0; i < nDofs_s1; ++i) {
                const real_num phi_i = phi_s1[i];
                for (unsigned j = 0; j < nDofs_s1; ++j) {
                    const real_num val = ( phi_s1[j] * phi_i) * weight_qp; // factor 2 as in your weak form
                    unk_element_jac_res.jac()[ (offset_s1 + i) * total_local_dofs + (offset_s1 + j) ] += val;
                }
            }
            for (unsigned i = 0; i < nDofs_s2; ++i) {
                const real_num phi_i = phi_s2[i];
                for (unsigned j = 0; j < nDofs_s2; ++j) {
                    const real_num val = (phi_s2[j] * phi_i) * weight_qp;
                    unk_element_jac_res.jac()[ (offset_s2 + i) * total_local_dofs + (offset_s2 + j) ] += val;
                }
            }

            // 3) B and B^T blocks (grad-grad coupling between u and v)
            //    B (row v, col u): int grad phi_v_i · grad phi_u_j
            for (unsigned i = 0; i < nDofs_v; ++i) {
                for (unsigned j = 0; j < nDofs_u; ++j) {
                    real_num val = 0.0;
                    // sum over spatial dims: gradphi_v[i*dim + d] * gradphi_u[j*dim + d]
                    for (unsigned d = 0; d < dim; ++d)
                        val += gradphi_v[i * dim + d] * gradphi_u[j * dim + d];
                    val *= weight_qp;
                    unk_element_jac_res.jac()[ (offset_v + i) * total_local_dofs + (offset_u + j) ] += - val; // B
                    // transpose B^T (row u, col v)
                    unk_element_jac_res.jac()[ (offset_u + j) * total_local_dofs + (offset_v + i) ] += - val; // B^T
                }
            }

            // 4) C1, C2 and their transposes (with nu1 scaling)
            //    C1 (row v, col s1): 0.5 * (∂x phi_v_i * ∂x phi_s1_j - ∂y phi_v_i * ∂y phi_s1_j)
            //    C2 (row v, col s2): 0.5 * (∂y phi_v_i * ∂x phi_s2_j + ∂x phi_v_i * ∂y phi_s2_j)
            for (unsigned i = 0; i < nDofs_v; ++i) {
                const double vx_x = gradphi_v[i * dim + 0];
                const double vx_y = gradphi_v[i * dim + 1];
                for (unsigned j = 0; j < nDofs_s1; ++j) {
                    const double s1x = gradphi_s1[j * dim + 0];
                    const double s1y = gradphi_s1[j * dim + 1];
                    double c1 = 0.5 * (vx_x * s1x - vx_y * s1y) * weight_qp;
                    // fill row v, col s1 (scaled by nu1, per your matrix)
                    unk_element_jac_res.jac()[ (offset_v + i) * total_local_dofs + (offset_s1 + j) ] += (nu1 * c1);
                    // transpose C1^T -> row s1, col v
                    unk_element_jac_res.jac()[ (offset_s1 + j) * total_local_dofs + (offset_v + i) ] += ( c1);
                }
                for (unsigned j = 0; j < nDofs_s2; ++j) {
                    const double s2x = gradphi_s2[j * dim + 0];
                    const double s2y = gradphi_s2[j * dim + 1];
                    double c2 = 0.5 * (vx_y * s2x + vx_x * s2y) * weight_qp;
                    // fill row v, col s2 (scaled by nu1)
                    unk_element_jac_res.jac()[ (offset_v + i) * total_local_dofs + (offset_s2 + j) ] += (nu1 * c2);
                    // transpose C2^T -> row s2, col v
                    unk_element_jac_res.jac()[ (offset_s2 + j) * total_local_dofs + (offset_v + i) ] += ( c2);
                }
            }

            // 5) Coupling of tau-equations to u already included above (via gradphi_u in tau loops),
            //    and coupling of u-row wrt v/s1/s2 (B^T and C^T) assembled above.

            // 6) Build local RHS: only second block (v-row) is non-zero: -nu2 * \int f * phi_v
            for (unsigned i = 0; i < nDofs_v; ++i) {
                rhs_local[offset_v + i] += (-nu2) * ( (double)phi_v[i] * (double)f_val ) * (double)weight_qp;
            }

        } // end gauss loop

        // --- compute local residual R_local = A_local * local_dofs - rhs_local ----
        // Build local_dofs vector in same ordering as dof_map
        std::vector<double> local_dofs(total_local_dofs, 0.0);
        unsigned pos = 0;
        for (unsigned uu = 0; uu < n_unknowns; ++uu) {
            const unsigned nd = unk_num_elem_dofs[uu];
            for (unsigned a = 0; a < nd; ++a) {
                local_dofs[pos++] = unknowns_local[uu].elem_dofs()[a];
            }
        }

        // compute A * x
        std::vector<double> Res_local(total_local_dofs, 0.0);
        for (unsigned row = 0; row < total_local_dofs; ++row) {
            double s = 0.0;
            unsigned row_off = row * total_local_dofs;
            for (unsigned col = 0; col < total_local_dofs; ++col) {
                s += unk_element_jac_res.jac()[ row_off + col ] * local_dofs[col];
            }
            Res_local[row] = s - rhs_local[row]; // A*x - rhs
        }

        // store into unk_element_jac_res.res() (FEMUS pattern: they will negate later before adding to global RES)
        for (unsigned ii = 0; ii < total_local_dofs; ++ii) {
            unk_element_jac_res.res()[ii] = Res_local[ii];
        }

        // --- assemble into global vectors/matrices ---
        std::vector<double> Res_total( unk_element_jac_res.res().size() );
        for (size_t kk = 0; kk < unk_element_jac_res.res().size(); ++kk)
            Res_total[kk] =   (- unk_element_jac_res.res()[kk] ); // FEMUS convention: negate
        RES->add_vector_blocked(Res_total, unk_element_jac_res.dof_map());

        if (assembleMatrix) {
            KK->add_matrix_blocked( unk_element_jac_res.jac(), unk_element_jac_res.dof_map(), unk_element_jac_res.dof_map() );
        }

        constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            std::vector<unsigned> Sol_n_el_dofs_Mat_vol = { nDofs_u, nDofs_v, nDofs_s1, nDofs_s2 };
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
