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
  std::vector<unsigned> nPoints{4,4};
  unsigned dim = lengths.size();
  nom.InitializeSimplestPointStructure(lengths,nPoints);
  nom.PrintX();
  
  // Testing the class Nom - creating the maps of neighbours and distances
  nom.SetConstantSupport(0.5);
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
    for(unsigned d2 = 0; d2 < dim; d2++){
      std::cout<<"| " << K[d1][d2];     
    }
    std::cout<<" |\n";
  }
  
  // Testing the class SimpleMatrix
  std::vector<std::vector<double>> M = {{6,-2,1},{-4,1,-1},{1,0,1}} ;
  SimpleMatrix mat(M); 
  bool inv = mat.inverse();
  mat.printInv();
  
}
