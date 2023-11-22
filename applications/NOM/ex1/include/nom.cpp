#include "nom.hpp"

namespace femus {

Nom::Nom() {

}

Nom::~Nom() {

}

void Nom::SetNumberOfNodes(const unsigned &nNodes){_nNodes = nNodes;}

void Nom::SetDimension(const unsigned &dim) {_dim = dim;}

void Nom::xResize() {_x.resize(_nNodes, vector<double>(_dim));}

// This function initialized an Hyperrectangle domain with nPoints per dimension
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

    std::vector<std::vector<double>> coord(_dim);
    for(unsigned d = 0; d < _dim; d++) coord[d].resize(nPoints[d]);
    for(unsigned d = 0; d < _dim; d++){
      for(unsigned i = 0; i < nPoints[d]; i++){  
      coord[d][i] = h[d] * i;  
      }
    }
    comb(coord);
  }
  return;
}

void Nom::comb(std::vector<std::vector<double> >& arr){
    // number of arrays
    int n = arr.size();
    _totCount = 0;
    
    // to keep track of next element in each of the n arrays
    int* indices = new int[n];
 
    // initialize with first element's index
    for (int i = 0; i < n; i++)
        indices[i] = 0;
 
    while (1) {
        // print current combination
        for (int i = 0; i < n; i++)  _x[_totCount][i] = arr[i][indices[i]];
        _totCount++;
 
        // find the rightmost array that has more elements left after the current element in that array
        int next = n - 1;
        while (next >= 0 && 
              (indices[next] + 1 >= arr[next].size()))
            next--;
 
        // no such array is found so no more 
        // combinations left
        if (next < 0)
            return;
 
        // if found move to next element in that array
        indices[next]++;
 
        // for all arrays to the right of this array current index again points to first element
        for (int i = next + 1; i < n; i++)
            indices[i] = 0;
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
