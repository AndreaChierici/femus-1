#pragma once

#include <fstream>
#include <iostream>     // std::cout, std::ios
#include <sstream>      // std::ostringstream

namespace femus {
    
  class Nom {
    public:
      Nom();
      ~Nom();
      
      void SetNumberOfNodes(const unsigned &nNodes);
      void SetDimension(const unsigned &dim);
      void xResize();
      void InitializeSimplestPointStructure(const std::vector<double> &lengths, const  std::vector<unsigned> &nPoints);
      void PrintX();
      void comb(vector<vector<double> >& arr);
      
    private:
      unsigned _dim;  
      std::vector<std::vector<double>> _x;
      unsigned _nNodes;

      unsigned _totCount;
      
      
  };
        
    
    
    
    
} // end namespace femus


