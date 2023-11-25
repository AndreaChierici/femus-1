#pragma once

#include <fstream>
#include <iostream>     // std::cout, std::ios
#include <sstream>      // std::ostringstream

#include "simpleMatrix.hpp"
// #include "SparseMatrix.hpp"
#include "petscmat.h"  

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
      void GetCoords(std::vector<std::vector<double>> &x);
      void comb(vector<vector<double> >& arr);
      
      void SetConstantSupport(double delta);
      void PointsInConstantSupport();
      void DistanceInConstantSupport();
      void PointsAndDistInConstantSupport();
      
      void ComputeOperatorK(unsigned i);
      void ComputeNotHomOperatorK(unsigned i, std::vector<double> vol, std::map<int, std::vector<double>> weight);
      void ComputeInvK(unsigned i);
      void InitializeVolumesAndWeights(std::vector<double> vol, std::map<int, std::vector<double>> weight);
      
      std::map<int, std::vector<int>> GetMap();
      std::map<int, std::vector<std::vector<double>>> GetDist();
      std::vector<std::vector<double>> GetK();
      std::vector<std::vector<double>> GetKinv();
      
      double ComputeNOMDivergence(std::vector<std::vector<double>> vec, unsigned i);
      std::vector<double> ComputeNOMScalarGradient(std::vector<double> sol, unsigned i);
      
      void CreateGlobalMatrix();
      
      
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
      std::vector<std::vector<double>> _Kinv;
      std::vector<double> _vol;
      std::map<int, std::vector<double>> _weight;
      
      SimpleMatrix _SM;
      
      std::vector<double> _sol;
      std::vector<std::vector<double>> _vecSol;
      
      Mat _A;
      
      
  };
  
  

        
    
    
    
    
} // end namespace femus


