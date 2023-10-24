/**
 * Solve 
 *     - \Delta u = 1
*/


#include "FemusInit.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"

#include "app_specifics.hpp"

#include "00_poisson_eqn_with_dirichlet_or_neumann_bc.hpp"

#include "Solution_functions_over_domains_or_mesh_files.hpp"

#include "FE_convergence.hpp"


using namespace femus;


// ======= Domain-related and Mesh-related stuff - BEGIN =========================


// 1D - BEGIN ===============================

namespace segment_0x1 {
  
 
// This depends on: 
// mesh file (for the face flags)
// underlying exact solution (if provided)
// equation (name of the unknowns; also, if it is Laplace, biharmonic, Stokes, etc)
// 
// 
bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob,
                      const std::vector < double >& x,
                      const char name[], 
                      double& value,
                      const int face_name,
                      const double time) {

  bool dirichlet = false;
  value = 0.;
  
  
 if (ml_prob->GetMLMesh()->GetDimension() != 1 )  abort();
  
  if (face_name == 1) {
      dirichlet = true;
        value = 0.;
    }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
    }

 
  return dirichlet;
  
 }
 
  }
  
// 1D - END ===============================



// 2D - BEGIN ===============================


// SQUARE - BEGIN

namespace square_01_by_01 {

  
bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

  if (ml_prob->GetMLMesh()->GetDimension() != 2 )  abort();
  
   bool dirichlet = false;
  value = 0.;
    
     
  if (face_name == 1) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
  }

 else  if (face_name == 3) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 4) {
      dirichlet = true;
        value = 0.;
  }
  
  

   return dirichlet;
   
}



}
// SQUARE - END



// CIRCLE - BEGIN



namespace circle {
  
  
bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

 if (ml_prob->GetMLMesh()->GetDimension() != 2 )  abort();
  


//  return dirichlet;
    
}






}
// CIRCLE - END



// SEMICIRCLE - BEGIN


namespace semicircle {

bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {
    
   if (ml_prob->GetMLMesh()->GetDimension() != 2 )  abort();

  bool dirichlet = false;
  value = 0.;
   
   
     if (face_name == 1) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
  }   
  
       return dirichlet;

    
}


}
// SEMICIRCLE - END



// QUARTER CIRCLE - BEGIN


namespace quarter_circle {


bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {
    
 if (ml_prob->GetMLMesh()->GetDimension() != 2 )  abort();
 
  bool dirichlet = false;
  value = 0.;
  
  const double tolerance = 1.e-5;
  
  
    if (face_name == 1) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 3) {
      dirichlet = true;
        value = 0.;
  }

  
  

  return dirichlet;
  
 }
 
 

 }
// QUARTER CIRCLE - END




// ANNULUS - BEGIN


namespace annulus {
 
 
bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

     if (ml_prob->GetMLMesh()->GetDimension() != 2 )  abort();
     
  bool dirichlet = false;
  value = 0.;
    

    if (face_name == 1) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
  }
   
   return dirichlet;
    
}

 
}
// ANNULUS - END



// SEMIANNULUS - BEGIN

namespace semiannulus {
 
 bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

 if (ml_prob->GetMLMesh()->GetDimension() != 2 )  abort();
 
  bool dirichlet = false;
  value = 0.;
  
     
    if (face_name == 1) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 3) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 4) {
      dirichlet = true;
        value = 0.;
  }
   
 
 
  return dirichlet;
  
 }

 
 

}
// SEMIANNULUS - END


// 2D - END ===============================



// 3D - BEGIN ===============================

// CUBE - BEGIN


namespace cube_01_by_01_by_01 {

bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

  if (ml_prob->GetMLMesh()->GetDimension() != 3 )  abort();
  
   bool dirichlet = false;
  value = 0.;
    
     
  if (face_name == 1) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
  }

 else  if (face_name == 3) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 4) {
      dirichlet = true;
        value = 0.;
  }
 else  if (face_name == 5) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 6) {
      dirichlet = true;
        value = 0.;
  }
  
  

   return dirichlet;
   
}





}
// CUBE - END



// CYLINDER - BEGIN


namespace cylinder_along_z_with_base_centered_at_1_by_1 {
 
bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

 if (ml_prob->GetMLMesh()->GetDimension() != 3 )  abort();

 bool dirichlet = false;
  value = 0.;
  
  
    if (face_name == 1) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 3) {
      dirichlet = true;
        value = 0.;
  }

 
  return dirichlet;
  
 }
 

 
}
// CYLINDER - END


// SEMICYLINDER - BEGIN

namespace semicylinder {
  
  
 bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

    if (ml_prob->GetMLMesh()->GetDimension() != 3 )  abort();

    
    
//       return dirichlet; 
    
}

 
 
}
// SEMICYLINDER - END


// QUARTER CYLINDER - BEGIN
 
 
namespace quarter_cylinder_along_z_with_base_centered_at_0_by_0 {
   
bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

     if (ml_prob->GetMLMesh()->GetDimension() != 3 )  abort();

  bool dirichlet = false;
  value = 0.;
    

     
    if (face_name == 1) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 2) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 3) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 4) {
      dirichlet = true;
        value = 0.;
  }
  else if (face_name == 5) {
      dirichlet = true;
        value = 0.;
  }
 
 
  return dirichlet;
  
}

 
 
}
// QUARTER CYLINDER - END





// PRISM WITH ANNULAR BASE - BEGIN


namespace prism_annular_base_along_z_with_base_centered_at_0_by_0 {
  
  
 bool bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double& value, const int face_name, const double time) {

 if (ml_prob->GetMLMesh()->GetDimension() != 3 )  abort();   
 
 bool dirichlet = false;
    value = 0.;


        if (face_name == 1) {
            dirichlet = true;
            value = 0.;
        }
        else if (face_name == 2) {
            dirichlet = true;
            value = 0.;
        }
        else if (face_name == 3) {
            dirichlet = true;
            value = 0.;
        }
        else if (face_name == 4) {
            dirichlet = true;
            value = 0.;
        }

    
    return dirichlet;

}



}
// PRISM WITH ANNULAR BASE - END


// 3D - END ===============================



// ======= Domain-related and Mesh-related stuff - END =========================




 //Unknown initial condition - BEGIN  ==================
double InitialValueU(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[]) {
    
  return 0.;
  
}
 //Unknown initial condition - END  ==================




 //Unknown definition - BEGIN  ==================
 const std::vector< Unknown >  provide_list_of_unknowns() {
     
     
  std::vector< FEFamily > feFamily;
  std::vector< FEOrder >   feOrder;

                        feFamily.push_back(LAGRANGE);
 
                        feOrder.push_back(/*FIRST*/SECOND);
 

  assert( feFamily.size() == feOrder.size() );
 
 std::vector< Unknown >  unknowns(feFamily.size());

   unknowns[0]._name      = "u";

   unknowns[0]._is_sparse = true;
   
     for (unsigned int u = 0; u < unknowns.size(); u++) {
         
              unknowns[u]._fe_family  = feFamily[u];
              unknowns[u]._fe_order   = feOrder[u];
              unknowns[u]._time_order = 0;
              unknowns[u]._is_pde_unknown = true;
              
     }
 
 
   return unknowns;
     
}
 //Unknown definition - END  ==================





int main(int argc, char** args) {

  // ======= Init ========================
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  // ======= Problem ========================
  MultiLevelProblem ml_prob;
  
  // ======= Files - BEGIN  ========================
  const bool use_output_time_folder = false;
  const bool redirect_cout_to_file = true;
  Files files; 
        files.CheckIODirectories(use_output_time_folder);
        files.RedirectCout(redirect_cout_to_file);

  // ======= Problem, Files ========================
  ml_prob.SetFilesHandler(&files);
  // ======= Files - END  ========================
  
  // ======= Problem, Quad Rule - BEGIN ========================
  std::string fe_quad_rule("seventh");
  
  ml_prob.SetQuadratureRuleAllGeomElems(fe_quad_rule);
  ml_prob.set_all_abstract_fe_AD_or_not();
  // ======= Problem, Quad Rule - END  ========================
  
    // ======= App Specifics - BEGIN  ==================
  std::string system_common_name = "Laplace";
  std::vector< app_specifics >   my_specifics;
  
  app_specifics  app_segment;   //me

  app_specifics  app_square;   //me
//   app_specifics  app_circle;   //Gayani
  app_specifics  app_semicircle;   //Himali
  app_specifics  app_quarter_circle;      //Max
  app_specifics  app_annulus;       //Jon
  app_specifics  app_semiannulus;        //Fahad

  app_specifics  app_cube;   //me
  app_specifics  app_cylinder;            //Aman
//   app_specifics  app_semicylinder;   //Rifat
  app_specifics  app_quarter_cylinder; //Armando   //coarser mesh to be done
  app_specifics  app_prism_annular_base;  //Abu
 
  const std::string relative_path_to_build_directory =  "../../../";
  
  //segment - BEGIN
  app_segment._system_name = system_common_name;
  app_segment._assemble_function = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
  app_segment._mesh_files.push_back("segment_16_dir_neu.med");
  app_segment._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/01_1d/segment/0-1/");
  
  app_segment._boundary_conditions_types_and_values             = segment_0x1::bc_all_dirichlet_homogeneous;

  segment_0x1::Function_Zero_on_boundary_1<>   app_segment_function_zero_on_boundary_1;
  app_segment._assemble_function_for_rhs   = & app_segment_function_zero_on_boundary_1;
  app_segment._true_solution_function      = & app_segment_function_zero_on_boundary_1;
  //segment - END
  
  
 //assignment_square - BEGIN
  app_square._system_name = system_common_name;
  app_square._assemble_function                            = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
  app_square._mesh_files.push_back("square_0-1x0-1_divisions_1x1.med");
  app_square._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/02_2d/square/0-1x0-1/");
  
  app_square._boundary_conditions_types_and_values             = square_01_by_01::bc_all_dirichlet_homogeneous;

  square_01_by_01::Function_Zero_on_boundary_1<>   app_square_function_zero_on_boundary_1;
  app_square._assemble_function_for_rhs        = & app_square_function_zero_on_boundary_1;
  app_square._true_solution_function           = & app_square_function_zero_on_boundary_1;
 //assignment_square - END


  //assignment_semicircle - BEGIN
  app_semicircle._system_name = system_common_name;
  app_semicircle._assemble_function = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
//   app_semicircle._mesh_files.push_back("assignment_semicircle_triangular.med");
  app_semicircle._mesh_files.push_back("assignment_semicircle_quadrilateral.med");
  app_semicircle._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/02_2d/circle_semi/");
  
  app_semicircle._boundary_conditions_types_and_values             = semicircle::bc_all_dirichlet_homogeneous;

  semicircle::Function_Zero_on_boundary_1<>     app_semicircle_function_zero_on_boundary_1;
  app_semicircle._assemble_function_for_rhs = & app_semicircle_function_zero_on_boundary_1;
  app_semicircle._true_solution_function    = & app_semicircle_function_zero_on_boundary_1;
  //assignment_semicircle - END
   
  
  //assignment_quarter_circle - BEGIN
  app_quarter_circle._system_name = system_common_name;
  app_quarter_circle._assemble_function = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
  app_quarter_circle._mesh_files.push_back("assignment_quarter_circle_triangular.med");
  app_quarter_circle._mesh_files.push_back("assignment_quarter_circle_quadrilateral.med");
  app_quarter_circle._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/02_2d/circle_quarter/");
  app_quarter_circle._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/02_2d/circle_quarter/");
  
  app_quarter_circle._boundary_conditions_types_and_values             = quarter_circle::bc_all_dirichlet_homogeneous;

  quarter_circle::Function_Zero_on_boundary_1<>     app_quarter_circle_function_zero_on_boundary_1;
  app_quarter_circle._assemble_function_for_rhs = & app_quarter_circle_function_zero_on_boundary_1;
  app_quarter_circle._true_solution_function    = & app_quarter_circle_function_zero_on_boundary_1;
  //assignment_quarter_circle - END
 

  //assignment_annulus - BEGIN
  app_annulus._system_name = system_common_name;
  app_annulus._assemble_function = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
//   app_annulus._mesh_files.push_back("assignment_annulus_triangular.med");
  app_annulus._mesh_files.push_back("assignment_annulus_quadrilateral.med");
  app_annulus._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/02_2d/annulus/");
  
  app_annulus._boundary_conditions_types_and_values             = annulus::bc_all_dirichlet_homogeneous;

  annulus::Function_Zero_on_boundary_1<>     app_annulus_function_zero_on_boundary_1;
  app_annulus._assemble_function_for_rhs = & app_annulus_function_zero_on_boundary_1;
  app_annulus._true_solution_function    = & app_annulus_function_zero_on_boundary_1;
  //assignment_annulus - END

  
  //assignment_semiannulus - BEGIN
  app_semiannulus._system_name = system_common_name;
  app_semiannulus._assemble_function = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
  app_semiannulus._mesh_files.push_back("assignment_semiannulus_triangular.med");
  app_semiannulus._mesh_files.push_back("assignment_semiannulus_quadrilateral.med");
  app_semiannulus._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/02_2d/annulus_semi/");
  app_semiannulus._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/02_2d/annulus_semi/");
  
  app_semiannulus._boundary_conditions_types_and_values             = semiannulus::bc_all_dirichlet_homogeneous;

  semiannulus::Function_Zero_on_boundary_1<>     app_semiannulus_function_zero_on_boundary_1;
  app_semiannulus._assemble_function_for_rhs = & app_semiannulus_function_zero_on_boundary_1;
  app_semiannulus._true_solution_function    = & app_semiannulus_function_zero_on_boundary_1;
  //assignment_semiannulus - END
 

 //assignment_cube - BEGIN
  app_cube._system_name = system_common_name;
  app_cube._assemble_function                            = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
  app_cube._mesh_files.push_back("assignment_cube_hexahedral.med");
  app_cube._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/03_3d/cube/0-1x0-1x0-1/");
  
  app_cube._boundary_conditions_types_and_values             = cube_01_by_01_by_01::bc_all_dirichlet_homogeneous;
    
  cube_01_by_01_by_01::Function_Zero_on_boundary_1<>   app_cube_function_zero_on_boundary_1;
  app_cube._assemble_function_for_rhs              = & app_cube_function_zero_on_boundary_1;
  app_cube._true_solution_function                 = & app_cube_function_zero_on_boundary_1;
 //assignment_cube - END
  

  
  //assignment_cylinder - BEGIN
  app_cylinder._system_name = system_common_name;
  app_cylinder._assemble_function = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
  app_cylinder._mesh_files.push_back("assignment_cylinder_tetrahedral.med");
  app_cylinder._mesh_files.push_back("assignment_cylinder_hexahedral.med");
  app_cylinder._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/03_3d/cylinder/");
  app_cylinder._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/03_3d/cylinder/");
  
  app_cylinder._boundary_conditions_types_and_values             = cylinder_along_z_with_base_centered_at_1_by_1::bc_all_dirichlet_homogeneous;

  cylinder_along_z_with_base_centered_at_1_by_1::Function_Zero_on_boundary_1<>   app_cylinder_function_zero_on_boundary_1;
  app_cylinder._assemble_function_for_rhs                                    = & app_cylinder_function_zero_on_boundary_1;
  app_cylinder._true_solution_function                                       = & app_cylinder_function_zero_on_boundary_1;
  //assignment_cylinder - END


  //assignment_quarter_cylinder - BEGIN
  app_quarter_cylinder._system_name = system_common_name;
  app_quarter_cylinder._assemble_function = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;

  app_quarter_cylinder._mesh_files.push_back("assignment_quarter_cylinder_tetrahedral.med");
  app_quarter_cylinder._mesh_files.push_back("assignment_quarter_cylinder_hexahedral.med");
//   app_quarter_cylinder._mesh_files.push_back("assignment_quarter_cylinder_hexahedral_0.med");
//   app_quarter_cylinder._mesh_files.push_back("assignment_quarter_cylinder_hexahedral_1.med");
  app_quarter_cylinder._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/03_3d/cylinder_quarter/");
  app_quarter_cylinder._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/03_3d/cylinder_quarter/");
  
  app_quarter_cylinder._boundary_conditions_types_and_values             = quarter_cylinder_along_z_with_base_centered_at_0_by_0::bc_all_dirichlet_homogeneous;
    
  quarter_cylinder_along_z_with_base_centered_at_0_by_0::Function_Zero_on_boundary_1<>   app_quarter_cylinder_function_zero_on_boundary_1;
  app_quarter_cylinder._assemble_function_for_rhs                                    = & app_quarter_cylinder_function_zero_on_boundary_1;
  app_quarter_cylinder._true_solution_function                                       = & app_quarter_cylinder_function_zero_on_boundary_1;
  //assignment_quarter_cylinder - END


  
  
  //assignment_tetra_prism_annular_base - BEGIN
  app_prism_annular_base._system_name = system_common_name;
  app_prism_annular_base._assemble_function = poisson_equation::equation_with_dirichlet_or_neumann_bc<double, double>;
  
  app_prism_annular_base._mesh_files.push_back("assignment_prism_annular_base_tetrahedral.med");
  app_prism_annular_base._mesh_files.push_back("assignment_prism_annular_base_hexahedral.med");
  app_prism_annular_base._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/03_3d/prism_annular_base/");
  app_prism_annular_base._mesh_files_path_relative_to_executable.push_back(relative_path_to_build_directory + DEFAULT_MESH_FILES_PATH + "00_salome/03_3d/prism_annular_base/");
  
  app_prism_annular_base._boundary_conditions_types_and_values             = prism_annular_base_along_z_with_base_centered_at_0_by_0::bc_all_dirichlet_homogeneous;

  prism_annular_base_along_z_with_base_centered_at_0_by_0::Function_Zero_on_boundary_1<>   app_prism_annular_base_function_zero_on_boundary_1;
  app_prism_annular_base._assemble_function_for_rhs                                    = & app_prism_annular_base_function_zero_on_boundary_1;
  app_prism_annular_base._true_solution_function                                       = & app_prism_annular_base_function_zero_on_boundary_1;
  //assignment_tetra_prism_annular_base - END
  
 
  


  
  
  my_specifics.push_back(app_segment);
  my_specifics.push_back(app_square);
  my_specifics.push_back(app_semicircle);
  my_specifics.push_back(app_quarter_circle);
  my_specifics.push_back(app_annulus);
  my_specifics.push_back(app_semiannulus);
  my_specifics.push_back(app_cube);
  my_specifics.push_back(app_cylinder);
  my_specifics.push_back(app_quarter_cylinder);
  my_specifics.push_back(app_prism_annular_base);

    // ======= App Specifics - END  ==================
  
  
 //begin app loop  
  for (unsigned int app = 0; app < my_specifics.size(); app++)  {
        
  // ======= Problem, App (every Problem has 1 App for now) ========================
  ml_prob.set_app_specs_pointer(&my_specifics[app]);
  
  
  // ======= Mesh - BEGIN  ==================
  MultiLevelMesh ml_mesh;
  // ======= Mesh - END  ==================
 

 for (unsigned int m = 0; m < my_specifics[app]._mesh_files.size(); m++)  {
   
  // ======= Mesh, Coarse reading - BEGIN ==================
  double Lref = 1.;
  
  const bool read_groups = true; //with this being false, we don't read any group at all. Therefore, we cannot even read the boundary groups that specify what are the boundary faces, for the boundary conditions
  const bool read_boundary_groups = true;
  

  const std::string mesh_file = my_specifics[app]._mesh_files_path_relative_to_executable[m] + my_specifics[app]._mesh_files[m];
  
  ml_mesh.ReadCoarseMeshFileReadingBeforePartitioning(mesh_file.c_str(), Lref, read_groups, read_boundary_groups);
    
  ml_mesh.GetLevelZero(0)->build_dofmap_all_fe_families_and_elem_and_node_structures();
 

  ml_mesh.BuildFETypesBasedOnExistingCoarseMeshGeomElements();
  
  ml_mesh.PrepareNewLevelsForRefinement();
  // ======= Mesh, Coarse reading - END ==================



 for (unsigned int r = 1; r < 3; r++)  {

  // ======= Mesh: Refinement - BEGIN  ==================
  unsigned numberOfUniformLevels = r;
  unsigned numberOfSelectiveLevels = 0;
  ml_mesh.RefineMesh(numberOfUniformLevels , numberOfUniformLevels + numberOfSelectiveLevels, NULL);
  // ======= Mesh: Refinement - END  ==================
  
  // ======= Mesh: Coarse erasing - BEGIN  ========================
  ml_mesh.EraseCoarseLevels(numberOfUniformLevels + numberOfSelectiveLevels - 1);
  ml_mesh.PrintInfo();
  // ======= Mesh: Coarse erasing - END  ========================
  
  // ======= Solution - BEGIN ==================
  MultiLevelSolution ml_sol(&ml_mesh);

  ml_sol.SetWriter(VTK);
  ml_sol.GetWriter()->SetDebugOutput(true);
  
  // ======= Problem, Mesh and Solution  ==================
  ml_prob.SetMultiLevelMeshAndSolution(& ml_sol);
  // ======= Solution - END ==================
  
  // ======= Solutions that are Unknowns - BEGIN ==================

 // ======= Solution: Add ==================
  std::vector< Unknown > unknowns = provide_list_of_unknowns();   ///@todo probably this should go in the Problem, to be accessed from other places
  
   for (unsigned int u = 0; u < unknowns.size(); u++)  { 
      ml_sol.AddSolution(unknowns[u]._name.c_str(), unknowns[u]._fe_family, unknowns[u]._fe_order, unknowns[u]._time_order, unknowns[u]._is_pde_unknown);
  }
 // ======= Solution: Initial Conditions ==================
   for (unsigned int u = 0; u < unknowns.size(); u++)  { 
      ml_sol.Initialize(unknowns[u]._name.c_str(), InitialValueU, & ml_prob);
  }
  // ======= Solution: Boundary Conditions ==================
  ml_sol.AttachSetBoundaryConditionFunction(my_specifics[app]._boundary_conditions_types_and_values);
   for (unsigned int u = 0; u < unknowns.size(); u++)  { 
      ml_sol.GenerateBdc(unknowns[u]._name.c_str(), (unknowns[u]._time_order == 0) ? "Steady" : "Time_dependent", & ml_prob);
  }

  // ======= Solutions that are Unknowns - END ==================

  

//   std::vector < std::vector < const elem_type_templ_base<double, double> *  > > elem_all = ml_prob.evaluate_all_fe<double, double>();
  
    // ======= Problem, System - BEGIN ========================
  ml_prob.clear_systems();
  
  NonLinearImplicitSystem & system = ml_prob.add_system < NonLinearImplicitSystem > (my_specifics[app]._system_name);
  
  system.SetDebugNonlinear(true);
 
       // ======= System Unknowns ========================
  for (unsigned int u = 0; u < unknowns.size(); u++)  system.AddSolutionToSystemPDE(unknowns[u]._name.c_str());  
 
  // attach the assembling function to system
  system.SetAssembleFunction( my_specifics[app]._assemble_function );

//   system.SetMaxNumberOfLinearIterations(2);
  // initialize and solve the system
  // system.SetMgType(V_CYCLE/*F_CYCLE*//*M_CYCLE*/); //it doesn't matter if I use only 1 level


  system.SetOuterSolver(PREONLY/*GMRES*/);
 
  system.init();
  
  system.MGsolve();
  
  compute_L2_norm_of_errors_of_unknowns_with_analytical_sol<double>(ml_prob, system.GetLevelToAssemble(), system.GetSolPdeIndex().size());
    // ======= Problem, System - END ========================
  
  
    // ======= Print - BEGIN ========================
  const std::string print_order = "biquadratic"; //"linear", "quadratic", "biquadratic"
  std::vector < std::string > variablesToBePrinted;
  variablesToBePrinted.push_back("all");
 
  ml_sol.GetWriter()->Write(my_specifics[app]._system_name + "_" + my_specifics[app]._mesh_files/*mesh_files*/[m], files.GetOutputPath(), print_order.c_str(), variablesToBePrinted);
    // ======= Print - END ========================
  
    }  //end refinement loop
  
  }  //end mesh file loop
 
 
} //end app loop
 
  return 0;
}










