#include "nom.hpp"

namespace femus {

Nom::Nom() {

}

Nom::~Nom() {

}

void Nom::SetNumberOfNodes(const unsigned &nNodes){_nNodes = nNodes;}

void Nom::SetDimension(const unsigned &dim) {_dim = dim;}

void Nom::xResize() {_x.resize(_nNodes, vector<double>(_dim));}

// This function initialized an Hyperrectangle domain with points
void Nom::InitializeSimplestPointStructure(const std::vector<double> &lengths, //lenght of the side of the domain in any direction
                                      const  std::vector<unsigned> &nPoints //number of points per direction
                                     ){
  if(lengths.size() != nPoints.size()){
    std::cerr<<"InitializeSimplestPointStructure: lengths and nPoints vectors sizes not matching";
    abort();
  }
  else{
    unsigned dim = nPoints.size();
    unsigned totNodes = 1;
    std::vector<double> h(lengths.size());
    for(unsigned k = 0; k < dim; k++) {
      totNodes *= nPoints[k];
      if(nPoints[k] == 0 || nPoints[k] == 1) {std::cerr<< "InitializeSimplestPointStructure: zero or one point along a direction initialized";}
      h[k] = lengths[k] / (nPoints[k] - 1);
    }
    SetNumberOfNodes(totNodes);
    SetDimension(dim);
    xResize();

    _count.resize(_dim,0);
    _totCount = 0;

//     TODO: all this function should be generalized with a recursive algorithm
    if(_dim == 2){
      for(unsigned j = 0; j < nPoints[1]; j++){
        _count[0]=0;
        doRecursion(_dim, nPoints, h);
        _count[1]++;
      }
    }
    else if(_dim == 3){
      for(unsigned k = 0; k < nPoints[2]; k++){
        _count[1]=0;
        for(unsigned j = 0; j < nPoints[1]; j++){
          _count[0]=0;
          doRecursion(_dim, nPoints, h);
          _count[1]++;
        }
        _count[2]++;
      }
    }
  }
  return;
}

void Nom::doRecursion(int baseCondition, const std::vector<unsigned> &nPoints, std::vector<double> &h){
  for(unsigned i = 0; i < nPoints[0]; i++){
    for(unsigned d = 0; d < _dim; d++){
      _x[_totCount][d] = h[d] * _count[d];
    }
    _totCount++;
    _count[0]++;
  }
}




void Nom::PrintX(){
  for(unsigned i = 0; i < _x.size(); i++){
    for(unsigned k = 0; k < _x[0].size(); k++){
      std::cout<< _x[i][k] << " ";
    }
    std::cout<<std::endl;
  }
}

} // end namespace femus
