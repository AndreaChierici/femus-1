#pragma once

#include <fstream>
#include <iostream>     // std::cout, std::ios
#include <sstream>      // std::ostringstream

#include "simpleMatrix.hpp"

namespace femus {
    
  class Nom {
    public:
      Nom();
      ~Nom();
      
      void SetNumberOfNodes(const unsigned &nNodes);
      void SetDimension(const unsigned &dim);
      void xResize();
      void InitializeMap();
      void InitializeDistMap();
      
      void InitializeSimplestPointStructure(const std::vector<double> &lengths, const  std::vector<unsigned> &nPoints);
      void PrintX();
      void comb(vector<vector<double> >& arr);
      
      void SetConstantSupport(double delta);
      void PointsInConstantSupport();
      void DistanceInConstantSupport();
      void PointsAndDistInConstantSupport();
      
      void ComputeOperatorK(unsigned i);
      void ComputeNotHomOperatorK(unsigned i, std::vector<double> vol, std::map<int, std::vector<double>> weight);
      void InitializeVolumesAndWeights(std::vector<double> vol, std::map<int, std::vector<double>> weight);
      
      std::map<int, std::vector<int>> GetMap();
      std::map<int, std::vector<std::vector<double>>> GetDist();
      std::vector<std::vector<double>> GetK();
      
      
    private:
      unsigned _dim;  
      std::vector<std::vector<double>> _x;
      unsigned _nNodes;

      unsigned _totCount;
      
      double _delta;
      std::vector<int> _count;
      std::map<int, std::vector<int>> _suppNodes;
      std::map<int, std::vector<std::vector<double>>> _suppDist;
      
      std::vector<std::vector<double>> _K;
      std::vector<double> _vol;
      std::map<int, std::vector<double>> _weight;
      
      
  };
  
  

        
    
    
    
    
} // end namespace femus


