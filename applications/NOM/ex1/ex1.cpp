#include "FemusInit.hpp"
#include "MultiLevelSolution.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "TransientSystem.hpp"
#include "LinearImplicitSystem.hpp"

#include "NumericVector.hpp"

#include "CurrentElem.hpp"
#include "ElemType_template.hpp"

#include "petsc.h"
#include "petscmat.h"
#include "PetscMatrix.hpp"
#include "FemusInit.hpp"

#include "include/simpleMatrix.hpp"
#include "include/simpleMatrix.cpp"
#include "include/nom.hpp"
#include "include/nom.cpp"

#include <Eigen/Dense>

using namespace femus;

std::vector<double> SetRhs(const std::vector<std::vector < double >>& x) {
  std::vector<double> value(x.size(), 0.);
  for(unsigned i = 0; i < x.size(); i++) {
    for(unsigned k = 0; k < x[0].size(); k++) {
      // value[i] +=  20 * x[i][k] * x[i][k] * x[i][k] + M_PI * M_PI * cos(M_PI * x[i][k]); //1D ODE
    }
    value[i] = 2 * x[i][0] *(x[i][1] - 1) * (x[i][1] - 2 * x[i][0] + x[i][0] * x[i][1] + 2) * exp(x[i][0] - x[i][1]); //2D POISSON
  }
  return value;
}

std::vector<double> SetAnSol(const std::vector<std::vector < double >>& x) {
  std::vector<double> value(x.size(), 0.);
  bool isDir = false;
  for(unsigned i = 0; i < x.size(); i++) {
    for(unsigned k = 0; k < x[0].size(); k++) if(x[i][k] < 1e-8 || x[i][k] > 1 - 1e-8) isDir = true;
    if(isDir) value[i] = 0;
    else{
      for(unsigned k = 0; k < x[0].size(); k++) {
        // value[i] +=  x[i][k] * x[i][k] * x[i][k] * x[i][k] * x[i][k] - 3 * x[i][k] - cos(M_PI * x[i][k]) + 1; //1D ODE
      }
      value[i] = x[i][0] * (1 - x[i][0]) * x[i][1] * (1 - x[i][1]) * exp(x[i][0] - x[i][1]);
    }
    isDir = false;
  }
  return value;
}

int main(int argc, char** argv)
{
  FemusInit mpinit(argc, argv, MPI_COMM_WORLD);
    
  // Testing the class Nom - initialization

  Nom nom;
  std::vector<double> lengths{1.,1.};
  std::vector<unsigned> nPoints{41,41};
  unsigned dim = lengths.size();
  nom.InitializeSimplestPointStructure(lengths,nPoints);
  nom.SetConstDeltaV(lengths);
  unsigned order = 6;
  unsigned np = (nom.factorial(order+dim)/(nom.factorial(order)*nom.factorial(dim))) - 1;
  unsigned nNeigh = /*5 * order +*/ np + 5 * order;
  std::cout<< "dim = " << dim << " | order = " << order << " | np = " << np << " | nNeigh = " << nNeigh << "\n";
  
  unsigned midPoint = 1;
  for(unsigned d = 0; d < dim; d++) midPoint *= nPoints[d];
  midPoint = (midPoint - (midPoint % 2)) / 2;

  std::cout<<"___________PRINT_X__________________\n";
  nom.PrintX();

//   Settig Dirichlet conditions
  std::vector<std::vector<double>> coords;
  nom.GetCoords(coords);
  std::vector<unsigned> dirCond(coords.size());
  bool isDir = false;
  unsigned cnt = 0;
  for(unsigned i = 0; i < coords.size(); i++){
    for(unsigned d = 0; d < coords[0].size(); d++) if(coords[i][d] < 1e-6 || coords[i][d] > 1 - 1e-6) isDir = true;
    if(isDir){
      dirCond[cnt] = i;
      cnt++;
    }
    isDir = false;
  }
  dirCond.resize(cnt);
  nom.SetBC(dirCond);

  std::cout<<"___________DIRICHLET_BC______________\n";
  for(unsigned i = 0; i < dirCond.size(); i++){
    for(unsigned d = 0; d < coords[0].size(); d++) std::cout << coords[dirCond[i]][d] << " ";
    std::cout << std::endl;
  }
  
  // // Testing the class Nom - creating the maps of neighbours and distances

  nom.PointsAndDistNPtsSupport(nNeigh);
  // nom.SetConstantSupport(0.5);
  // nom.PointsAndDistInConstantSupport();

  // std::map<int, std::vector<int>> map = nom.GetMap();
  std::map<int, std::vector<std::vector<double>>> dist = nom.GetDist();
  std::map<int, std::vector<std::pair<int,double>>> mapN = nom.GetMapN();

  // std::cout<<"___________MAP______________________\n";
  // for(unsigned i = 0; i < map.size(); i++){
  //     std::cout<< i << " | ";
  //     for(unsigned j = 0; j < map[i].size(); j++) std::cout << map[i][j] << " ";
  //     std::cout<<std::endl;
  // }
  std::cout<<"___________MAP_N____________________\n";
  for(unsigned i = 0; i < mapN.size(); i++){
      std::cout<< i << " | ";
      for(unsigned j = 0; j < mapN[i].size(); j++) std::cout << mapN[i][j].first << " ";
      std::cout<<std::endl;
  }
  std::cout<<"___________DIST______________________\n";
  for(unsigned i = 0; i < dist.size(); i++){
      std::cout<< i << " | ";
      for(unsigned j = 0; j < dist[i].size(); j++) {
          for(unsigned k = 0; k < dim; k++) std::cout << dist[i][j][k] << " ";
          std::cout <<" / ";
      }
      std::cout<<std::endl;
  }
  
  // Testing the class Nom - compuiting the operator K
  nom.ComputeOperatorK(0);
  std::vector<std::vector<double>> K = nom.GetK();
  std::cout<<"_______________________________________\n";
  std::cout<<"K = \n";
  for(unsigned d1 = 0; d1 < dim; d1++){
    std::cout<<"{ ";
    for(unsigned d2 = 0; d2 < dim; d2++){
      std::cout<< K[d1][d2] ;
      if(d2 < dim-1) std::cout << ", ";
    }
    std::cout<<" }";
    if(d1 < dim-1) std::cout << ", ";
    std::cout<< std::endl;
  }
  
//   // Testing the class SimpleMatrix
//   std::vector<std::vector<double>> M = {{6,-2,1,5,6},{-4,1,-1,7,8},{1,0,1,-1,2},{1,0,4,-1,1},{0,0,0,0,2}} ;
//   SimpleMatrix mat(M);
//   bool inv = mat.inverse();
//   mat.printInv();
  
// Testing the class Nom - compuiting inverse of the operator K
  nom.ComputeInvK(0);
  std::vector<std::vector<double>> Kinv = nom.GetKinv();
  std::cout<<"_______________________________________\n";
  std::cout<<"K inverse = \n";
  for(unsigned d1 = 0; d1 < dim; d1++){
    std::cout<<"{ ";
    for(unsigned d2 = 0; d2 < dim; d2++){
      std::cout << Kinv[d1][d2];
      if(d2 < dim-1) std::cout << ", ";
    }
    std::cout<<" }";
    if(d1 < dim-1) std::cout << ", ";
    std::cout<< std::endl;
  }
  
//   // Testing matrix-vector multiplication in the class SimpleMatrix
//   SimpleMatrix mat(Kinv);
//   std::vector<double> res = mat.vecMult({1,2,-4});
//   std::cout<<"_______________________________________\n";
//   for(unsigned i = 0; i < res.size(); i++) std::cout << res[i]<< " \n";

  // std::vector<std::vector<double>> field;
  // std::vector<std::vector<double>> coords;
  // nom.GetCoords(coords);
  // field.resize(coords.size(), std::vector<double>(coords[0].size(), 0.));
  // for(unsigned i = 0; i < coords.size(); i++){
  //   for(unsigned d = 0; d < coords[0].size(); d++) {
  //     field[i][d] = coords[i][d] * coords[i][d] * coords[i][d];
  //   }
  // }
  //  double div = 0;
  //  double err = 0;
  // for(unsigned i = 0; i < coords.size(); i++){
  //   div = nom.ComputeNOMDivergence(field, i);
  //   std::cout << i << " coords: ";
  //   for(unsigned d = 0; d < coords[0].size(); d++) std::cout << coords[i][d] << " ";
  //   std::cout<< " div = " << div << " ";
  //   std::cout << " Local value (valid for 2d cubic): " << 3 * (coords[i][0] * coords[i][0] + coords[i][1] * coords[i][1] );
  //   std::cout<<std::endl;
  //   double tmp = 0;
  //   for(unsigned d = 0; d < coords[0].size(); d++) tmp += 3 * coords[i][d] * coords[i][d];
  //   err += (div - tmp) * (div - tmp);
  // }
//   std::cout<<"____________REF________________________\n";
//   for(unsigned i = 0; i < 11; i++){
//    std::cout << 0.1*i << " " << 3 * 0.5 * 0.5 + 3 * 0.1 * i<< "\n";
//   }
//   std::cout<<"_______________________________________\n";
//   for(unsigned i = 0; i < coords.size(); i++){
//     div = nom.ComputeNOMDivergence(field, i);
//     if(coords[i][1] < 0.5001 && coords[i][1] > 0.4999){
//       std::cout << coords[i][0] << " " << div << "\n";
//     }
//   }
//   std::cout<<"___________ERR________________________\n";
//     for(unsigned i = 0; i < coords.size(); i++){
//     div = nom.ComputeNOMDivergence(field, i);
//     if(coords[i][1] < 0.5001 && coords[i][1] > 0.4999){
//       std::cout << coords[i][0] << " " << div  - 3 * 0.5 * 0.5 - 3 * coords[i][0] * coords[i][0]<< "\n";
//     }
//   }
//   std::cout<<"ERR = " << err << "\n";
//
  std::cout<<"_______________________________________\n";
  nom.MultiIndexList(order);
  std::vector<std::vector<int>> list = nom.GetMultiIndexList();

  for(unsigned i = 0; i < list.size();i++){
    std::cout << "(";
    for(unsigned j = 0; j < list[i].size(); j++){
      std::cout<< list[i][j] << ",";
    }
    std::cout << "),";
  }
  std::cout<<std::endl;

//   std::cout<<"_______________________________________\n";
//   std::vector<double> polyIndex;
//   polyIndex = nom.PolyMultiIndex(8,2,0.5);
//   for(unsigned i = 0; i < polyIndex.size(); i++) std::cout << polyIndex[i] << " ";
//   std::cout<< std::endl;
  
  // std::cout<<"________TEST_ComputeHighOrdOperatorK____________\n";
  // nom.ComputeHighOrdOperatorK(0);
  // std::vector<std::vector<double>> KHO=nom.GetKHO();
  // for (unsigned i = 0; i < KHO.size(); i++){
  //   for (unsigned j = 0; j < KHO[0].size(); j++){
  //     std::cout<< KHO[i][j] << " ";
  //   }
  //   std::cout<<std::endl;
  // }

//   std::cout<<"________TEST_ComputeOperatorB__Eigen______\n";
//   nom.ComputeOperatorB(midPoint);
//   Eigen::MatrixXd KHOE = nom.GetKHOE();
//   std::cout << "Operator K: \n"<< KHOE << std::endl;
// 
//   Eigen::MatrixXd B = nom.GetB();
//   std::cout << "Operator B: \n"<< B << std::endl;

//   std::vector<double> field;
//   field.resize(coords.size(), 0.);
//   for(unsigned i = 0; i < coords.size(); i++){
// //     for(unsigned d = 0; d < coords[0].size(); d++) {
// //       field[i] += coords[i][d] * coords[i][d] * coords[i][d];
// //     }
//     field[i] += coords[i][0] * coords[i][0] * coords[i][1] * coords[i][1];
//   }
//   nom.SetField(field);
//   Eigen::VectorXd der=nom.ComputeHighOrdDer(midPoint+4);
//
//   Eigen::MatrixXd KHOE = nom.GetKHOE();
//   Eigen::MatrixXd poly = nom.GetPolyE();
//   Eigen::MatrixXd B = nom.GetB();
// //   std::cout << "Operator K: \n"<< KHOE << std::endl;
// //   std::cout << "Operator Poly: \n"<< poly << std::endl;
// //   std::cout << "Operator B: \n"<< B << std::endl;
//   std::cout << "DERIVATIVES: \n"<< der << std::endl;

//   Matrix and rhs creation
  nom.CreateGlobalEigenMatrix();
  nom.CreateGlobalEigenRhs();
//   Setting the rhs and analytic solution
  std::vector<double> rhs = SetRhs(coords);
  std::vector<double> anSol = SetAnSol(coords);
  nom.SetAnalyticSol(anSol);
//   Assembling the matrix
//   nom.AssembleNonLocalKernelEigen(0.5);
  nom.AssembleLaplacian();
  nom.SetEigenRhs(rhs);
//   Solve the system
  nom.SolveEigen();

// //   Printing matrix, rhs and solution
//   nom.PrintGlobalEigenMatrix();
  // nom.PrintGlobalEigenRhs();
  nom.PrintGlobalEigenSolution();

  std::cout<<"____________ERROR_____________\n";
  std::cout << "L2 error = " << nom.L2Error();





//   nom.CreateGlobalMatrix();

  
  
  
  
  
  
}
