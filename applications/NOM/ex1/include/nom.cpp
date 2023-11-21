void SetNumberOfNodes(const unsigned &nNodes) _nNodes = nNodes;

void SetDimension(const unsigned &dim) _dim = dim;

void xResize() _x.resize(_nNodes, vector<double>(_dim));

// This function initialized an Hyperrectangle domain with points
void InitializeSimplestPointStructure(const std::vector<double> &lengths, //lenght of the side of the domain in any direction
                                      const  std::vector<unsigned> &nPoints //number of points per direction
                                     ){
  if(lengths.size() != nPoints.size()){
    std::cerr<<"InitializeSimplestPointStructure: lengths and nPoints vectors sizes not matching";
    abort();  
  }
  else{
    unsigned totNodes = 1;  
    std::vector<double> h(lengths.size());
    for(unsigned k = 0; k < nPoints.size(); k++) {
      totNodes *= nPoints[k];
      if(nPoints[k] == 0) {std::cerr<< "InitializeSimplestPointStructure: zero points along a direction initialized"};
      h[k] = lengths[k] / nPoints[k];
    }
    SetNumberOfNodes(totNodes);
    SetDimension(nPoints.size());
    xResize();

    int count = 0;
    for(unsigned k = 0; k < nPoints.size(); k++) {
      for(unsigned i = 0; i < nPoints[k])  
//       _x[count][k] = 
          a
      count++;
    }
  }
  return;
}
