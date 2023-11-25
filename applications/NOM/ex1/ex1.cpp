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

#include "PetscMatrix.hpp"

#include "include/simpleMatrix.hpp"
#include "include/simpleMatrix.cpp"
#include "include/nom.hpp"
#include "include/nom.cpp"

using namespace femus;

int main(int argc, char** argv)
{
  // Testing the class Nom - initialization

  Nom nom;
  std::vector<double> lengths{1.,1.};
  std::vector<unsigned> nPoints{11,11};
  unsigned dim = lengths.size();
  nom.InitializeSimplestPointStructure(lengths,nPoints);
  nom.PrintX();
  
  // Testing the class Nom - creating the maps of neighbours and distances
  nom.SetConstantSupport(0.11);
  nom.PointsAndDistInConstantSupport();
  std::map<int, std::vector<int>> map = nom.GetMap();
  std::map<int, std::vector<std::vector<double>>> dist = nom.GetDist();
  
  for(unsigned i = 0; i < map.size(); i++){
      std::cout<< i << " | ";
      for(unsigned j = 0; j < map[i].size(); j++) std::cout << map[i][j] << " ";
      std::cout<<std::endl;
  }
  std::cout<<"_______________________________________\n";
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
  nom.ComputeInvK();
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

  std::vector<std::vector<double>> field;
  std::vector<std::vector<double>> coords;
  nom.GetCoords(coords);
  field.resize(coords.size(), std::vector<double>(coords[0].size(), 0.));
  for(unsigned i = 0; i < coords.size(); i++){
    for(unsigned d = 0; d < coords[0].size(); d++) {
      field[i][d] = coords[i][d] * coords[i][d] * coords[i][d];    
    } 
  }
   double div = 0;
   double err = 0;
  for(unsigned i = 0; i < coords.size(); i++){
    div = nom.ComputeNOMDivergence(field, i);
    std::cout << i << " coords: ";
    for(unsigned d = 0; d < coords[0].size(); d++) std::cout << coords[i][d] << " ";
    std::cout<< " div = " << div << " ";
    std::cout << " Local value (valid for 2d cubic): " << 3 * (coords[i][0] * coords[i][0] + coords[i][1] * coords[i][1] );
    std::cout<<std::endl;
    double tmp = 0;
    for(unsigned d = 0; d < coords[0].size(); d++) tmp += 3 * coords[i][d] * coords[i][d];
    err += (div - tmp) * (div - tmp);
  }
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
  std::cout<<"ERR = " << err << "\n";
  
  
  
  
  
  
}
