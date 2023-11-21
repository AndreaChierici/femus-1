#ifndef __femus_nom_hpp__
#define __femus_nom_hpp__

#include <fstream>
#include <iostream>     // std::cout, std::ios
#include <sstream>      // std::ostringstream

namespace femus {
    
  class Nom {
    public:
      Nom() {
      };
      ~Nom() {};
      
      void SetNumberOfNodes(const unsigned &nNodes);
      void SetDimension(const unsigned &dim);
      void InitializeSimplestPointStructure(const std::vector<double> &lengths, const  std::vector<unsigned> &nPoints);
      
    private:
      unsigned _dim;  
      std::vector<std::vector<double>> _x;
      unsigned _nNodes;
      
      
  };
        
    
    
    
    
} // end namespace femus


#endif
