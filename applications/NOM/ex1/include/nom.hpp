#pragma once

#include <fstream>
#include <iostream>     // std::cout, std::ios
#include <sstream>      // std::ostringstream
#include <algorithm>
// #include <iterator>
// #include <set>

#include "simpleMatrix.hpp"
// #include "SparseMatrix.hpp"
#include "petscmat.h"
#include <Eigen/Dense>

// using namespace Eigen;


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
      void SetField(std::vector<double> field);
      void SetConstDeltaV(std::vector<double> dimensions);
      void SetBC(std::vector<unsigned> dirNodes);
      
      void SetConstantSupport(double delta);
      void PointsInConstantSupport();
      void DistanceInConstantSupport();
      void PointsAndDistInConstantSupport();
      void PointsInConstantSupportWithInv();
      void PointsAndDistInConstantSupportWithInv();

      void PointsAndDistNPtsSupport(unsigned npt);
      
      void ComputeOperatorK(unsigned i);
      void ComputeNotHomOperatorK(unsigned i, std::vector<double> vol, std::map<int, std::vector<double>> weight);
      void ComputeInvK(unsigned i);
      void InitializeVolumesAndWeights(std::vector<double> vol, std::map<int, std::vector<double>> weight);
      
      std::map<int, std::vector<int>> GetMap();
      std::map<int, std::vector<std::pair<int,double>>> GetMapN();
      std::map<int, std::vector<std::vector<double>>> GetDist();
      std::vector<std::vector<double>> GetK();
      std::vector<std::vector<double>> GetKinv();
      std::vector<std::vector<double>> GetKHO();
      Eigen::MatrixXd GetKHOE();
      Eigen::MatrixXd GetPolyE();
      std::vector<std::vector<int>> GetMultiIndexList();
      Eigen::MatrixXd GetB();
      
      double ComputeNOMDivergence(std::vector<std::vector<double>> vec, unsigned i);
      std::vector<double> ComputeNOMScalarGradient(std::vector<double> sol, unsigned i);
      
      void SetOrder(unsigned n);
      void MultiIndexList(unsigned n);
      unsigned factorial(unsigned n);
      void combinationUtil(int arr[], int data[], int index, int r);
      std::vector<double> PolyMultiIndex(unsigned i, unsigned j);
      Eigen::MatrixXd DiagLengthHInv(unsigned i);
      Eigen::MatrixXd SelfTensProd(std::vector<double> vec);
      void ComputeHighOrdOperatorK(unsigned i);
      void ComputePolyOperator(unsigned i);
      void ComputeHighOrdKAndPolyOperators(unsigned i);
      void ComputeOperatorB(unsigned i);

      Eigen::VectorXd ComputeHighOrdDer(unsigned i);
      void AssembleLaplacianNode(unsigned i);
      void AssembleLaplacian();
      void SetEigenRhs(std::vector<double> rhs);
      
      void CreateGlobalEigenMatrix();
      void CreateGlobalEigenRhs();
      void PrintGlobalEigenMatrix();
      void PrintGlobalEigenRhs();
      void PrintGlobalEigenSolution();

      void SetAnalyticSol(std::vector<double> sol);
      void SolveEigen();
      double L2Error();

      void CreateGlobalMatrix();
      
      
    private:
      unsigned _dim;  
      std::vector<std::vector<double>> _x;
      unsigned _nNodes;
      std::vector<double> _deltaV;
      std::vector<double> _field;
      std::vector<unsigned> _dirBC;
      double _penalty = 1e5;

      unsigned _totCount;
      
      double _delta;
      std::vector<int> _count;
      std::map<int, std::vector<int>> _suppNodes;
      std::map<int, std::map<int,int>> _suppNodesInv;
      std::map<int, std::vector<std::vector<double>>> _suppDist;

      std::map<int, std::vector<std::pair<int,double>>> _suppNodesN;
      std::vector<double> _h;
      
      std::vector<std::vector<double>> _K;
      std::vector<std::vector<double>> _Kinv;
      std::vector<double> _vol;
      std::map<int, std::vector<double>> _weight;
      
      SimpleMatrix _SM;
      
      std::vector<double> _sol;
      std::vector<std::vector<double>> _vecSol;
      
      std::vector<std::vector<int>> _multiIndexList;
      unsigned _indxDim;
      unsigned _n;
      unsigned _cnt;
      std::vector<std::vector<double>> _KHO;
      Eigen::MatrixXd _HinvE;
      Eigen::MatrixXd _KHOE;
      Eigen::MatrixXd _PolyE;
      Eigen::MatrixXd _BE;

      Eigen::MatrixXd  _ME;
      Eigen::VectorXd _rhsE;
      Eigen::VectorXd _solE;

      std::vector<double> _anSol;


      Mat _A;
      
      
  };
  
  

        
    
    
    
    
} // end namespace femus


