#ifndef __femus_biharmonic_HM_nonauto_D_hpp__
#define __femus_biharmonic_HM_nonauto_D_hpp__

#include "FemusInit.hpp"  //for the adept stack

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

  class biharmonic_HM_nonauto {

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


//========= BOUNDARY_IMPLEMENTATION_Sxx - BEGIN ==================

static void natural_loop_1dV(const MultiLevelProblem *    ml_prob,
                     const Mesh *                    msh,
                     const MultiLevelSolution *    ml_sol,
                     const unsigned iel,
                     CurrentElem < double > & geom_element,
                     const unsigned xType,
                     const std::string solname_sxx,
                     const unsigned solFEType_sxx,
                     std::vector< double > & Res
                    ) {

     double grad_sxx_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, xType);

       geom_element.set_elem_center_bdry_3d();

       std::vector <  double > xx_face_elem_center(3, 0.);
          xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxx.c_str(), grad_sxx_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet)  &&  (grad_sxx_dot_n != 0.) ) {  //dirichlet == false and nonhomogeneous Neumann



                   unsigned n_dofs_face = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxx);

                  for (unsigned i = 0; i < n_dofs_face; i++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i);

                 Res[i_vol] +=  grad_sxx_dot_n /* * phi[node] = 1. */;

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
                       const std::string solname_sxx,
                       const unsigned solFEType_sxx,
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
  std::vector <double> phi_sxx_bdry;
  std::vector <double> phi_sxx_x_bdry;

  phi_sxx_bdry.reserve(max_size);
  phi_sxx_x_bdry.reserve(max_size * space_dim);
// ---

// ---
  std::vector <double> phi_coords_bdry;
  std::vector <double> phi_coords_x_bdry;

  phi_coords_bdry.reserve(max_size);
  phi_coords_x_bdry.reserve(max_size * space_dim);
// ---



     double grad_sxx_dot_n = 0.;

    for (unsigned jface = 0; jface < msh->GetElementFaceNumber(iel); jface++) {

       geom_element.set_coords_at_dofs_bdry_3d(iel, jface, solType_coords);

       geom_element.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = msh->GetElementFaceType(iel, jface);


       std::vector <  double > xx_face_elem_center(3, 0.);
       xx_face_elem_center = geom_element.get_elem_center_bdry_3d();

       const int boundary_index = msh->GetMeshElements()->GetFaceElementIndex(iel, jface);

       if ( boundary_index < 0) { //I am on the boundary

         unsigned int face = - (boundary_index + 1);

         bool is_dirichlet =  ml_sol->GetBdcFunctionMLProb()(ml_prob, xx_face_elem_center, solname_sxx.c_str(), grad_sxx_dot_n, face, 0.);
         //we have to be careful here, because in GenerateBdc those coordinates are passed as NODE coordinates,
         //while here we pass the FACE ELEMENT CENTER coordinates.
         // So, if we use this for enforcing space-dependent Dirichlet or Neumann values, we need to be careful!

             if ( !(is_dirichlet) /* &&  (grad_u_dot_n != 0.)*/ ) {  //dirichlet == false and nonhomogeneous Neumann

    unsigned n_dofs_face_sxx = msh->GetElementFaceDofNumber(iel, jface, solFEType_sxx);

// dof-based - BEGIN
     std::vector< double > grad_sxx_dot_n_at_dofs(n_dofs_face_sxx);


    for (unsigned i_bdry = 0; i_bdry < grad_sxx_dot_n_at_dofs.size(); i_bdry++) {
        std::vector<double> x_at_node(dim, 0.);
        for (unsigned jdim = 0; jdim < x_at_node.size(); jdim++) x_at_node[jdim] = geom_element.get_coords_at_dofs_bdry_3d()[jdim][i_bdry];

      double grad_sxx_dot_n_at_dofs_temp = 0.;
      ml_sol->GetBdcFunctionMLProb()(ml_prob, x_at_node, solname_sxx.c_str(), grad_sxx_dot_n_at_dofs_temp, face, 0.);
     grad_sxx_dot_n_at_dofs[i_bdry] = grad_sxx_dot_n_at_dofs_temp;

    }

// dof-based - END


                        const unsigned n_gauss_bdry = ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussPointsNumber();


		for(unsigned ig_bdry = 0; ig_bdry < n_gauss_bdry; ig_bdry++) {

     elem_all[ielGeom_bdry][solType_coords]->JacJacInv(geom_element.get_coords_at_dofs_bdry_3d(), ig_bdry, Jac_iqp_bdry, JacI_iqp_bdry, detJac_iqp_bdry, space_dim);

    weight_iqp_bdry = detJac_iqp_bdry * ml_prob->GetQuadratureRule(ielGeom_bdry).GetGaussWeightsPointer()[ig_bdry];

    elem_all[ielGeom_bdry][solFEType_sxx ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_sxx_bdry, phi_sxx_x_bdry,  boost::none, space_dim);



//---------------------------------------------------------------------------------------------------------

     elem_all[ielGeom_bdry][solType_coords ]->shape_funcs_current_elem(ig_bdry, JacI_iqp_bdry, phi_coords_bdry, phi_coords_x_bdry,  boost::none, space_dim);

  std::vector<double> x_qp_bdry(dim, 0.);

         for (unsigned i = 0; i < phi_coords_bdry.size(); i++) {
           	for (unsigned d = 0; d < dim; d++) {
 	                                                x_qp_bdry[d]    += geom_element.get_coords_at_dofs_bdry_3d()[d][i] * phi_coords_bdry[i]; // fetch of coordinate points
             }
         }

           double grad_sxx_dot_n_qp = 0.;  ///@todo here we should do a function that provides the gradient at the boundary, and then we do "dot n" with the normal at qp

// dof-based
         for (unsigned i_bdry = 0; i_bdry < phi_sxx_bdry.size(); i_bdry ++) {
           grad_sxx_dot_n_qp +=  grad_sxx_dot_n_at_dofs[i_bdry] * phi_sxx_bdry[i_bdry];
         }

//---------------------------------------------------------------------------------------------------------



                  for (unsigned i_bdry = 0; i_bdry < n_dofs_face_sxx; i_bdry++) {

                 unsigned int i_vol = msh->GetLocalFaceVertexIndex(iel, jface, i_bdry);

                 Res[i_vol] +=  weight_iqp_bdry * grad_sxx_dot_n_qp /*grad_u_dot_n*/  * phi_sxx_bdry[i_bdry];

                           }


                        }


                    }

              }
    }

}


//========= BOUNDARY_IMPLEMENTATION_V - END ==================



// Function signature remains the same, but implementation will be different
static void AssembleBilaplaceProblem_AD(MultiLevelProblem& ml_prob) {
    // This is the manual version, so the Adept stack is not used
    // adept::Stack& s = FemusInit::_adeptStack;

    // extract pointers to the several objects that we are going to use
    NonLinearImplicitSystem* mlPdeSys = &ml_prob.get_system<NonLinearImplicitSystem>(ml_prob.get_app_specs_pointer()->_system_name);
    const unsigned level = mlPdeSys->GetLevelToAssemble();
    Mesh* msh = ml_prob._ml_msh->GetLevel(level);
    Solution* sol = ml_prob._ml_sol->GetSolutionLevel(level);
    LinearEquationSolver* pdeSys = mlPdeSys->_LinSolver[level];
    SparseMatrix* KK = pdeSys->_KK;
    NumericVector* RES = pdeSys->_RES;

    const unsigned dim = msh->GetDimension();
    unsigned iproc = msh->processor_id();

    // Solution variable names and indices
    const std::string solname_u = ml_prob._ml_sol->GetSolName_string_vec()[0];
    unsigned soluIndex = ml_prob._ml_sol->GetIndex(solname_u.c_str());
    unsigned solFEType_u = ml_prob._ml_sol->GetSolutionType(soluIndex);
    unsigned soluPdeIndex = mlPdeSys->GetSolPdeIndex(solname_u.c_str());

    const std::string solname_sxx = ml_prob._ml_sol->GetSolName_string_vec()[1];
    unsigned solsxxIndex = ml_prob._ml_sol->GetIndex(solname_sxx.c_str());
    unsigned solFEType_sxx = ml_prob._ml_sol->GetSolutionType(solsxxIndex);
    unsigned solsxxPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxx.c_str());

    const std::string solname_sxy = ml_prob._ml_sol->GetSolName_string_vec()[2];
    unsigned solsxyIndex = ml_prob._ml_sol->GetIndex(solname_sxy.c_str());
    unsigned solsxyPdeIndex = mlPdeSys->GetSolPdeIndex(solname_sxy.c_str());

    const std::string solname_syy = ml_prob._ml_sol->GetSolName_string_vec()[3];
    unsigned solsyyIndex = ml_prob._ml_sol->GetIndex(solname_syy.c_str());
    unsigned solsyyPdeIndex = mlPdeSys->GetSolPdeIndex(solname_syy.c_str());

    // Local solution vectors, now using `double` instead of `adept::adouble`
    std::vector<double> solu;
    std::vector<double> solsxx;
    std::vector<double> solsxy;
    std::vector<double> solsyy;

    // Other local vectors
    std::vector<std::vector<double>> x(dim);
    unsigned xType = 2; // LAGRANGE QUADRATIC
    std::vector<int> sysDof;
    std::vector<double> phi;
    std::vector<double> phi_x;
    std::vector<double> phi_xx;
    double weight;
    std::vector<double> Res; // Local residual vector
    std::vector<double> Jac; // Local Jacobian matrix

    // Reserve memory
    const unsigned maxSize = static_cast<unsigned>(ceil(pow(3, dim)));
    solu.reserve(maxSize);
    solsxx.reserve(maxSize);
    solsxy.reserve(maxSize);
    solsyy.reserve(maxSize);
    for (unsigned i = 0; i < dim; i++)
        x[i].reserve(maxSize);
    sysDof.reserve(4 * maxSize);
    phi.reserve(maxSize);
    phi_x.reserve(maxSize * dim);
    unsigned dim2 = (6 * (dim - 1) + !(dim - 1));
    phi_xx.reserve(maxSize * dim2);
    Res.reserve(4 * maxSize);
    Jac.reserve(16 * maxSize * maxSize);

    KK->zero();

    double nu = 0.1;
    double nu1 = (4.0 * (1.0 - nu)) / (1.0 + nu);
    double nu2 = 2.0 / (1.0 + nu);

    // Loop through elements
    for (int iel = msh->GetElementOffset(iproc); iel < msh->GetElementOffset(iproc + 1); iel++) {
        short unsigned ielGeom = msh->GetElementType(iel);
        unsigned nDofs = msh->GetElementDofNumber(iel, solFEType_sxx);
        unsigned nDofs2 = msh->GetElementDofNumber(iel, xType);
        std::vector<unsigned> Sol_n_el_dofs_Mat_vol(4, nDofs);

        // Resize local arrays
        sysDof.resize(4 * nDofs);
        solu.resize(nDofs);
        solsxx.resize(nDofs);
        solsxy.resize(nDofs);
        solsyy.resize(nDofs);
        for (int i = 0; i < dim; i++) {
            x[i].resize(nDofs2);
        }
        std::vector<double> local_aResu(nDofs, 0.);
        std::vector<double> local_aRessxx(nDofs, 0.);
        std::vector<double> local_aRessxy(nDofs, 0.);
        std::vector<double> local_aRessyy(nDofs, 0.);

        // Local storage of global mapping and solution
        for (unsigned i = 0; i < nDofs; i++) {
            unsigned solDof = msh->GetSolutionDof(i, iel, solFEType_sxx);
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
        for (unsigned i = 0; i < nDofs2; i++) {
            unsigned xDof = msh->GetSolutionDof(i, iel, xType);
            for (unsigned jdim = 0; jdim < dim; jdim++) {
                x[jdim][i] = (*msh->GetTopology()->_Sol[jdim])(xDof);
            }
        }

        // Residual and Jacobian computation
        Jac.assign(16 * nDofs * nDofs, 0.0);

        // Gauss point loop
        for (unsigned ig = 0; ig < msh->_finiteElement[ielGeom][solFEType_u]->GetGaussPointNumber(); ig++) {
            msh->_finiteElement[ielGeom][solFEType_sxx]->Jacobian(x, ig, weight, phi, phi_x, phi_xx);

            // Evaluate solution and its derivatives at the Gauss point
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
                }
            }

            // phi_i loop for residuals and Jacobian
            for (unsigned i = 0; i < nDofs; i++) {
                // Calculate residual terms at the gauss point
                double Bxxsxx = phi_x[i * dim] * solsxxGauss_x[0];
                double Bxysxy = phi_x[i * dim + 1] * solsxyGauss_x[0] + phi_x[i * dim] * solsxyGauss_x[1];
                double Byysyy = phi_x[i * dim + 1] * solsyyGauss_x[1];

                double Bxxu = phi_x[i * dim] * soluGauss_x[0];
                double Bxyu = phi_x[i * dim + 1] * soluGauss_x[0] + phi_x[i * dim] * soluGauss_x[1];
                double Byyu = phi_x[i * dim + 1] * soluGauss_x[1];

                double Mxxxx_sxx = phi[i] * solsxxGauss;
                double Mxyxy_sxy = 2. * phi[i] * solsxyGauss;
                double Myyyy_syy = phi[i] * solsyyGauss;

                std::vector<double> xGauss(dim, 0.);
                for (unsigned k=0; k<nDofs; k++) {
                    for(unsigned jdim=0; jdim<dim; jdim++) {
                        xGauss[jdim] += x[jdim][k] * phi[k];
                    }
                }

                double F_term = ml_prob.get_app_specs_pointer()->_assemble_function_for_rhs->laplacian(xGauss) * phi[i];

                // System residuals
                local_aResu[i] += (Bxxsxx + Bxysxy + Byysyy + F_term) * weight;
                local_aRessxx[i] += (Bxxu + Mxxxx_sxx) * weight;
                local_aRessxy[i] += (Bxyu + Mxyxy_sxy) * weight;
                local_aRessyy[i] += (Byyu + Myyyy_syy) * weight;

                // Loop for Jacobian calculation
                for (unsigned j = 0; j < nDofs; j++) {
                    const unsigned num_rows_cols = 4 * nDofs;

                    // d(aResu[i]) / d(solu[j]) = 0
                    // d(aResu[i]) / d(solsxx[j])
                    Jac[(nDofs + j) * num_rows_cols + i] += (phi_x[i * dim] * phi_x[j * dim]) * weight;
                    // d(aResu[i]) / d(solsxy[j])
                    Jac[(2 * nDofs + j) * num_rows_cols + i] += (phi_x[i * dim + 1] * phi_x[j * dim] + phi_x[i * dim] * phi_x[j * dim + 1]) * weight;
                    // d(aResu[i]) / d(solsyy[j])
                    Jac[(3 * nDofs + j) * num_rows_cols + i] += (phi_x[i * dim + 1] * phi_x[j * dim + 1]) * weight;

                    // d(aRessxx[i]) / d(solu[j])
                    Jac[j * num_rows_cols + (nDofs + i)] += (phi_x[i * dim] * phi_x[j * dim]) * weight;
                    // d(aRessxx[i]) / d(solsxx[j])
                    Jac[(nDofs + j) * num_rows_cols + (nDofs + i)] += phi[i] * phi[j] * weight;
                    // d(aRessxx[i]) / d(solsxy[j]) = 0
                    // d(aRessxx[i]) / d(solsyy[j]) = 0

                    // d(aRessxy[i]) / d(solu[j])
                    Jac[j * num_rows_cols + (2 * nDofs + i)] += (phi_x[i * dim + 1] * phi_x[j * dim] + phi_x[i * dim] * phi_x[j * dim + 1]) * weight;
                    // d(aRessxy[i]) / d(solsxx[j]) = 0
                    // d(aRessxy[i]) / d(solsxy[j])
                    Jac[(2 * nDofs + j) * num_rows_cols + (2 * nDofs + i)] += 2. * phi[i] * phi[j] * weight;
                    // d(aRessxy[i]) / d(solsyy[j]) = 0

                    // d(aRessyy[i]) / d(solu[j])
                    Jac[j * num_rows_cols + (3 * nDofs + i)] += (phi_x[i * dim + 1] * phi_x[j * dim + 1]) * weight;
                    // d(aRessyy[i]) / d(solsxx[j]) = 0
                    // d(aRessyy[i]) / d(solsxy[j]) = 0
                    // d(aRessyy[i]) / d(solsyy[j])
                    Jac[(3 * nDofs + j) * num_rows_cols + (3 * nDofs + i)] += phi[i] * phi[j] * weight;
                }
            }
        }

        // Copy values from local residuals to the global residual vector
        Res.resize(4 * nDofs);
        for (int i = 0; i < nDofs; i++) {
            Res[i] = -local_aResu[i];
            Res[nDofs + i] = -local_aRessxx[i];
            Res[2 * nDofs + i] = -local_aRessxy[i];
            Res[3 * nDofs + i] = -local_aRessyy[i];
        }

        RES->add_vector_blocked(Res, sysDof);
        KK->add_matrix_blocked(Jac, sysDof, sysDof);

        constexpr bool print_algebra_local = false;
        if (print_algebra_local) {
            assemble_jacobian<double, double>::print_element_jacobian(iel, Jac, Sol_n_el_dofs_Mat_vol, 10, 5);
            assemble_jacobian<double, double>::print_element_residual(iel, Res, Sol_n_el_dofs_Mat_vol, 10, 5);
        }
    }

    RES->close();
    KK->close();
}

  };

}

#endif
