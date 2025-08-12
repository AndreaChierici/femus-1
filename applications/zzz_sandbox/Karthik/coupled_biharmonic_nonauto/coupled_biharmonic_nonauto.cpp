/** tutorial/Ex3
 * This example shows how to set and solve the weak form of the nonlinear problem
 *                     -\Delta^2 u = f(x) \text{ on }\Omega,
 *            u=0 \text{ on } \Gamma,
 *      \Delta u=0 \text{ on } \Gamma,
 * on a box domain $\Omega$ with boundary $\Gamma$,
 * by using a system of second order partial differential equation.
 * all the coarse-level meshes are removed;
 * a multilevel problem and an equation system are initialized;
 * a direct solver is used to solve the problem.
 **/

#include "FemusInit.hpp"
#include "Files.hpp"
#include "MultiLevelProblem.hpp"
#include "MultiLevelSolution.hpp"
#include "NonLinearImplicitSystem.hpp"
#include "LinearEquationSolver.hpp"
#include "VTKWriter.hpp"
#include "NumericVector.hpp"

//#include "biharmonic_coupled.hpp"

#include "FE_convergence.hpp"

#include "Solution_functions_over_domains_or_mesh_files.hpp"

#include "adept.h"
// // // extern Domains::square_m05p05::Function_Zero_on_boundary_4<double> analytical_function;


#define LIBRARY_OR_USER   1 //0: library; 1: user

#if LIBRARY_OR_USER == 0
   #include "01_biharmonic_coupled.hpp"
   #define NAMESPACE_FOR_BIHARMONIC   femus
#elif LIBRARY_OR_USER == 1
   #include "coupled_biharmonic_nonauto.hpp"
   #define NAMESPACE_FOR_BIHARMONIC_COUPLED   karthik
#endif



using namespace femus;

namespace Domains {

namespace  square_m05p05  {


template <class type = double>
class Function_Zero_on_boundary_f_9 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 1.;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 0.;
        solGrad[1] = 0.;
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 0.;
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_9 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2. * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = 2. * pi * sin(2. * pi * x[0]) * cos(2.* pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -8. * pi * pi * sin(2.* pi * x[0]) * sin(2.*pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_9_Laplacian : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -8.*pi*pi * sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -16. * pi * pi * pi * cos(2. * pi*x[0]) * sin(2. * pi*x[1]);
        solGrad[1] = -16. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2.* pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 64. * pi * pi * pi * pi * sin(2. * pi*x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_9_sxx : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -4. * pi * pi * sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * cos(2.* pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * sin(2.* pi * x[0]) * cos(2. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 32. * pi * pi * pi * pi * sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_9_sxy : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return  4. * pi * pi * cos(2. * pi * x[0]) * cos(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * cos(2. * pi * x[0]) * sin( 2. * pi * x[1] );
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -32. * pi * pi * pi * pi * cos(2.*pi*x[0]) * cos(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_9_syy : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -4. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * sin(2. * pi * x[0]) * cos( 2. * pi*x[1] );
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 32. * pi * pi * pi * pi * sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};


}


}
/*
namespace Domains {

namespace square_m05p05  {

// ---- Helper: a = 0.5 for domain [-0.5, 0.5]^2 -----------------------------

template <class type = double>
class Function_Zero_on_boundary_9 : public Math::Function<type> {
public:
    static constexpr type a = static_cast<type>(0.5);

    // u(x,y) = (x^2 - a^2)^2 (y^2 - a^2)^2
    type value(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        return (Ax*Ax) * (Ay*Ay);
    }

    // ∇u = [ 4x(x^2-a^2)(y^2-a^2)^2 , 4y(y^2-a^2)(x^2-a^2)^2 ]
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        g[0] = static_cast<type>(4.) * x[0] * Ax * (Ay*Ay);
        g[1] = static_cast<type>(4.) * x[1] * Ay * (Ax*Ax);
        return g;
    }

    // Δu = 4(3x^2-a^2)(y^2-a^2)^2 + 4(3y^2-a^2)(x^2-a^2)^2
    type laplacian(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        const type u_xx = static_cast<type>(4.) * (static_cast<type>(3.)*x[0]*x[0] - a*a) * (Ay*Ay);
        const type u_yy = static_cast<type>(4.) * (static_cast<type>(3.)*x[1]*x[1] - a*a) * (Ax*Ax);
        return u_xx + u_yy;
    }
};

// This is Δu (for your RHS helper usage)
template <class type = double>
class Function_Zero_on_boundary_9_Laplacian : public Math::Function<type> {
public:
    static constexpr type a = static_cast<type>(0.5);

    type value(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        const type u_xx = static_cast<type>(4.) * (static_cast<type>(3.)*x[0]*x[0] - a*a) * (Ay*Ay);
        const type u_yy = static_cast<type>(4.) * (static_cast<type>(3.)*x[1]*x[1] - a*a) * (Ax*Ax);
        return u_xx + u_yy;
    }

    // ∇(Δu) — rarely needed; provided for completeness
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;

        // ∂/∂x Δu = 24 x (Ay^2) + 16 x (3y^2 - a^2) Ax
        g[0] = static_cast<type>(24.) * x[0] * (Ay*Ay)
             + static_cast<type>(16.) * x[0] * (static_cast<type>(3.)*x[1]*x[1] - a*a) * Ax;

        // ∂/∂y Δu = 24 y (Ax^2) + 16 y (3x^2 - a^2) Ay
        g[1] = static_cast<type>(24.) * x[1] * (Ax*Ax)
             + static_cast<type>(16.) * x[1] * (static_cast<type>(3.)*x[0]*x[0] - a*a) * Ay;

        return g;
    }

    // Δ(Δu) optional; not typically used
    type laplacian(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        // From differentiating the gradient above:
        const type d2x = static_cast<type>(24.) * (Ay*Ay)
                       + static_cast<type>(16.) * (static_cast<type>(3.)*x[1]*x[1] - a*a) * (static_cast<type>(3.)*x[0]*x[0] - a*a);
        const type d2y = static_cast<type>(24.) * (Ax*Ax)
                       + static_cast<type>(16.) * (static_cast<type>(3.)*x[0]*x[0] - a*a) * (static_cast<type>(3.)*x[1]*x[1] - a*a);
        return d2x + d2y;
    }
};

// sxx = u_xx
template <class type = double>
class Function_Zero_on_boundary_9_sxx : public Math::Function<type> {
public:
    static constexpr type a = static_cast<type>(0.5);

    // u_xx = 4(3x^2 - a^2) (y^2 - a^2)^2
    type value(const std::vector<type>& x) const {
        const type Ay = x[1]*x[1] - a*a;
        return static_cast<type>(4.) * (static_cast<type>(3.)*x[0]*x[0] - a*a) * (Ay*Ay);
    }

    // ∇(u_xx) = [ 24 x (Ay^2), 16 y (3x^2 - a^2) Ay ]
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type Ay = x[1]*x[1] - a*a;
        g[0] = static_cast<type>(24.) * x[0] * (Ay*Ay);
        g[1] = static_cast<type>(16.) * x[1] * (static_cast<type>(3.)*x[0]*x[0] - a*a) * Ay;
        return g;
    }

    // Δ(u_xx) = 24 (Ay^2) + 16 (3x^2 - a^2)(3y^2 - a^2)
    type laplacian(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        return static_cast<type>(24.) * (Ay*Ay)
             + static_cast<type>(16.) * (static_cast<type>(3.)*x[0]*x[0] - a*a) * (static_cast<type>(3.)*x[1]*x[1] - a*a);
    }
};

// sxy = u_xy  (use 2*u_xy if your formulation uses engineering shear)
template <class type = double>
class Function_Zero_on_boundary_9_sxy : public Math::Function<type> {
public:
    static constexpr type a = static_cast<type>(0.5);

    // u_xy = 16 x y (x^2 - a^2)(y^2 - a^2)
    type value(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        return static_cast<type>(16.) * x[0] * x[1] * Ax * Ay;
    }

    // ∇(u_xy) = [ 16 y (3x^2 - a^2) Ay , 16 x (3y^2 - a^2) Ax ]
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        g[0] = static_cast<type>(16.) * x[1] * (static_cast<type>(3.)*x[0]*x[0] - a*a) * Ay;
        g[1] = static_cast<type>(16.) * x[0] * (static_cast<type>(3.)*x[1]*x[1] - a*a) * Ax;
        return g;
    }

    // Δ(u_xy) = 96 x y [ (x^2 - a^2) + (y^2 - a^2) ] = 96 x y (x^2 + y^2 - 2a^2)
    type laplacian(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        const type Ay = x[1]*x[1] - a*a;
        return static_cast<type>(96.) * x[0] * x[1] * (Ax + Ay);
    }
};

// syy = u_yy
template <class type = double>
class Function_Zero_on_boundary_9_syy : public Math::Function<type> {
public:
    static constexpr type a = static_cast<type>(0.5);

    // u_yy = 4(3y^2 - a^2) (x^2 - a^2)^2
    type value(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        return static_cast<type>(4.) * (static_cast<type>(3.)*x[1]*x[1] - a*a) * (Ax*Ax);
    }

    // ∇(u_yy) = [ 16 x (3y^2 - a^2) Ax , 24 y (Ax^2) ]
    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> g(x.size(), 0.);
        const type Ax = x[0]*x[0] - a*a;
        g[0] = static_cast<type>(16.) * x[0] * (static_cast<type>(3.)*x[1]*x[1] - a*a) * Ax;
        g[1] = static_cast<type>(24.) * x[1] * (Ax*Ax);
        return g;
    }

    // Δ(u_yy) = 16 (3x^2 - a^2)(3y^2 - a^2) + 24 (Ax^2)
    type laplacian(const std::vector<type>& x) const {
        const type Ax = x[0]*x[0] - a*a;
        return static_cast<type>(16.) * (static_cast<type>(3.)*x[0]*x[0] - a*a) * (static_cast<type>(3.)*x[1]*x[1] - a*a)
             + static_cast<type>(24.) * (Ax*Ax);
    }
};

} // namespace square_m05p05

} // namespace Domains
*/


//====Set boundary condition-BEGIN==============================
bool SetBoundaryCondition_bc_all_dirichlet_homogeneous(const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char SolName[], double& Value, const int facename, const double time) {
  bool dirichlet = true; //dirichlet

  if (!strcmp(SolName, "u")) {
      Math::Function <double> * u = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      // strcmp compares two string in lexiographic sense.
    Value = u -> value(x);
  }
  else if (!strcmp(SolName, "sxx")) {
      Math::Function <double> * sxx = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
    Value = sxx -> value(x);
  }
  else if (!strcmp(SolName, "sxy")) {
      Math::Function <double> * sxy = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      Value = sxy -> value(x);
  }
  else if (!strcmp(SolName, "syy")) {
      Math::Function <double> * syy = ml_prob -> get_ml_solution() -> get_analytical_function(SolName);
      Value = syy -> value(x);
  }
  return dirichlet;
}
//====Set boundary condition-END==============================



int main(int argc, char** args) {

  // init Petsc-MPI communicator
  FemusInit mpinit(argc, args, MPI_COMM_WORLD);

  // ======= Files - BEGIN  ========================
  const bool use_output_time_folder = false; // This allows you to run the code multiple times without overwriting. This will generate an output folder each time you run.
  const bool redirect_cout_to_file = false; // puts the output in a log file instead of the term
  Files files;
        files.CheckIODirectories(use_output_time_folder);
        files.RedirectCout(redirect_cout_to_file);

  // ======= Files - END  ========================


    // ======= System Specifics - BEGIN  ==================
  system_specifics  system_biharmonic_coupled;   //me

  // =========Mesh file - BEGIN ==================
  system_biharmonic_coupled._mesh_files.push_back("square_-0p5-0p5x-0p5-0p5_divisions_2x2.med");
  const std::string relative_path_to_build_directory =  "../../../../";
  const std::string mesh_file = relative_path_to_build_directory + Files::mesh_folder_path() + "00_salome/2d/square/minus0p5-plus0p5_minus0p5-plus0p5/";  system_biharmonic_coupled._mesh_files_path_relative_to_executable.push_back(mesh_file);
 // =========Mesh file - END ==================


  system_biharmonic_coupled._system_name = "Biharmonic";
  system_biharmonic_coupled._assemble_function = NAMESPACE_FOR_BIHARMONIC_COUPLED :: biharmonic_coupled_equation :: AssembleBilaplaceProblem_AD;

  system_biharmonic_coupled._boundary_conditions_types_and_values             = SetBoundaryCondition_bc_all_dirichlet_homogeneous;

/*
   Domains::square_m05p05::Function_NonZero_on_boundary_4<>   system_biharmonic_coupled_function_zero_on_boundary_1;
   Domains::square_m05p05::Function_NonZero_on_boundary_4_Laplacian<>   system_biharmonic_coupled_function_zero_on_boundary_1_Laplacian;
   system_biharmonic_coupled._assemble_function_for_rhs   = & system_biharmonic_coupled_function_zero_on_boundary_1_Laplacian; //this is the RHS for the auxiliary variable v = -Delta u

   system_biharmonic_coupled._true_solution_function      = & system_biharmonic_coupled_function_zero_on_boundary_1;

*/

  Domains::square_m05p05::Function_Zero_on_boundary_9<> system_biharmonic_coupled_function_zero_on_boundary_1;
   Domains::square_m05p05::Function_Zero_on_boundary_9_sxx<> system_biharmonic_coupled_function_zero_on_boundary_sxx;

  Domains::square_m05p05::Function_Zero_on_boundary_9_sxy<> system_biharmonic_coupled_function_zero_on_boundary_sxy;
  Domains::square_m05p05::Function_Zero_on_boundary_9_syy<> system_biharmonic_coupled_function_zero_on_boundary_syy;
  Domains::square_m05p05::Function_Zero_on_boundary_9_Laplacian<> system_biharmonic_coupled_function_zero_on_boundary_1_Laplacian;


  Domains::square_m05p05::Function_Zero_on_boundary_f_9<> system_biharmonic_coupled_function_zero_on_boundary_f;



  system_biharmonic_coupled._assemble_function_for_rhs = &system_biharmonic_coupled_function_zero_on_boundary_f;
  system_biharmonic_coupled._true_solution_function = &system_biharmonic_coupled_function_zero_on_boundary_1;





  ///@todo if this is not set, nothing happens here. It is used to compute absolute errors
    // ======= System Specifics - END ==================



  // define multilevel mesh
  MultiLevelMesh mlMsh;
  // read coarse level mesh and generate finers level meshes
  double scalingFactor = 1.;
  const std::string mesh_file_total = system_biharmonic_coupled._mesh_files_path_relative_to_executable[0] + "/" + system_biharmonic_coupled._mesh_files[0];
  mlMsh.ReadCoarseMesh(mesh_file_total.c_str(), "seventh", scalingFactor);

  unsigned maxNumberOfMeshes = 5;

  std::vector < std::vector < double > > l2Norm;
  l2Norm.resize(maxNumberOfMeshes);

  std::vector < std::vector < double > > semiNorm;
  semiNorm.resize(maxNumberOfMeshes);

    std::vector<FEOrder> feOrder;
    feOrder.push_back(FIRST);
    feOrder.push_back(SERENDIPITY);
    feOrder.push_back(SECOND);



  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {   // loop on the mesh level

    unsigned numberOfUniformLevels = i + 1;
    unsigned numberOfSelectiveLevels = 0;
    mlMsh.RefineMesh(numberOfUniformLevels , numberOfUniformLevels + numberOfSelectiveLevels, NULL);

    // erase all the coarse mesh levels
    mlMsh.EraseCoarseLevels(numberOfUniformLevels - 1);

    // print mesh info
    mlMsh.PrintInfo();

    l2Norm[i].resize( feOrder.size() );
    semiNorm[i].resize( feOrder.size() );

    for (unsigned j = 0; j < feOrder.size(); j++) {   // loop on the FE Order

      // define the multilevel solution and attach the mlMsh object to it
      MultiLevelSolution mlSol(&mlMsh);


      mlSol.AddSolution("u", LAGRANGE, feOrder[j]);

      mlSol.set_analytical_function("u", & system_biharmonic_coupled_function_zero_on_boundary_1);

      mlSol.AddSolution("sxx", LAGRANGE, feOrder[j]);

      mlSol.set_analytical_function("sxx", & system_biharmonic_coupled_function_zero_on_boundary_sxx);

      // ---------- ADDED: sxy and syy (same FE order as u,v) ----------
      // IMPORTANT: register in this order so assembler receives [u, v, sxy, syy]
      mlSol.AddSolution("sxy", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("sxy", & system_biharmonic_coupled_function_zero_on_boundary_sxy);

      mlSol.AddSolution("syy", LAGRANGE, feOrder[j]);
      mlSol.set_analytical_function("syy", & system_biharmonic_coupled_function_zero_on_boundary_syy);
      // ------------------------------------------------------------


      mlSol.Initialize("All");



      // define the multilevel problem attach the mlSol object to it
      MultiLevelProblem ml_prob(&mlSol);

      ml_prob.set_app_specs_pointer(& system_biharmonic_coupled);
      // ======= Problem, Files ========================
      ml_prob.SetFilesHandler(&files);

      // attach the boundary condition function and generate boundary data
      mlSol.AttachSetBoundaryConditionFunction( system_biharmonic_coupled._boundary_conditions_types_and_values );
      mlSol.GenerateBdc("u", "Steady", & ml_prob);
      mlSol.GenerateBdc("sxx", "Steady", & ml_prob);

      // ---------- generate BDC for sxy, syy as well ----------
      mlSol.GenerateBdc("sxy", "Steady", & ml_prob);
      mlSol.GenerateBdc("syy", "Steady", & ml_prob);
      // -----------------------------------------------------


      // add system Biharmonic in ml_prob as a Linear Implicit System
      NonLinearImplicitSystem& system = ml_prob.add_system < NonLinearImplicitSystem > (system_biharmonic_coupled._system_name);

      // add solution "u" to system
      system.AddSolutionToSystemPDE("u");
      system.AddSolutionToSystemPDE("sxx");

      // ---------- register sxy, syy in the PDE system ----------
      system.AddSolutionToSystemPDE("sxy");
      system.AddSolutionToSystemPDE("syy");
      // -------------------------------------------------------

      // attach the assembling function to system
      system.SetAssembleFunction( system_biharmonic_coupled._assemble_function );

      // initialize and solve the system
      system.init();

      system.MGsolve();



// // //       // convergence for u
      std::pair< double , double > norm = GetErrorNorm_L2_H1_with_analytical_sol(& mlSol, "u",  & system_biharmonic_coupled_function_zero_on_boundary_1);



      l2Norm[i][j]  = norm.first;
      semiNorm[i][j] = norm.second;



      // print solutions
      std::vector < std::string > variablesToBePrinted;
      variablesToBePrinted.push_back("All");

      std::string  an_func = "test";
      VTKWriter vtkIO(&mlSol);
      vtkIO.Write(an_func, Files::_application_output_directory, "biquadratic", variablesToBePrinted, i);

    }
  }


  // FE_convergence::output_convergence_order();


  // ======= L2 - BEGIN  ========================
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "l2 ERROR and ORDER OF CONVERGENCE:\n\n";
  std::cout << "LEVEL\tFIRST\t\t\tSERENDIPITY\t\tSECOND\n";

  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    std::cout << i + 1 << "\t";
    std::cout.precision(14);

    for (unsigned j = 0; j < feOrder.size(); j++) {
      std::cout << l2Norm[i][j] << "\t";
    }

    std::cout << std::endl;

    if (i < maxNumberOfMeshes - 1) {
      std::cout.precision(3);
      std::cout << "\t\t";

      for (unsigned j = 0; j < feOrder.size(); j++) {
        std::cout << log(l2Norm[i][j] / l2Norm[i + 1][j]) / log(2.) << "\t\t\t";
      }

      std::cout << std::endl;
    }

  }
  // ======= L2 - END  ========================



  // ======= H1 - BEGIN  ========================

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "SEMINORM ERROR and ORDER OF CONVERGENCE:\n\n";
  std::cout << "LEVEL\tFIRST\t\t\tSERENDIPITY\t\tSECOND\n";

  for (unsigned i = 0; i < maxNumberOfMeshes; i++) {
    std::cout << i + 1 << "\t";
    std::cout.precision(14);

    for (unsigned j = 0; j < feOrder.size(); j++) {
      std::cout << semiNorm[i][j] << "\t";
    }

    std::cout << std::endl;

    if (i < maxNumberOfMeshes - 1) {
      std::cout.precision(3);
      std::cout << "\t\t";

      for (unsigned j = 0; j < feOrder.size(); j++) {
        std::cout << log(semiNorm[i][j] / semiNorm[i + 1][j]) / log(2.) << "\t\t\t";
      }

      std::cout << std::endl;
    }

  }

  // ======= H1 - END  ========================


  return 0;
}
