#pragma once

#include <fstream>
#include <iostream>     // std::cout, std::ios
#include <sstream>      // std::ostringstream

#include <bits/stdc++.h>
using namespace std;
// #define N 4

namespace femus {
    
  class SimpleMatrix {
    public:
      SimpleMatrix();
      SimpleMatrix(std::vector<std::vector<double>> M);
      ~SimpleMatrix();
      
      void setMatrix(std::vector<std::vector<double>> M);
      std::vector<std::vector<double>> getMatrix();
      void getCofactor(std::vector<std::vector<double>> Mat, int p, int q, int n);
      double determinant(std::vector<std::vector<double>> Mat, int n);
      void adjoint();
      bool inverse();
      void printInv();
      std::vector<std::vector<double>> getInv();
      std::vector<double> matVecMult(std::vector<double> vec);
      std::vector<double> vecMatMult(std::vector<double> vec);
      void Sum(std::vector<std::vector<double>> mat);
      void DiagMatProd(std::vector<double> diag);
    
      
      private:
        unsigned _sz;  
        std::vector<std::vector<double>> _M;
        std::vector<std::vector<double>> _adj;
        std::vector<std::vector<double>> _temp;
        std::vector<std::vector<double>> _inv;

      
  };
  
} // end namespace femus
