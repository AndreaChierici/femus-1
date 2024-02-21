#include "nom.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Cholesky>

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

// This function initialized an Hyperrectangle domain with nPoints per dimension
void Nom::InitPointStructureNLBC(const std::vector<double> &lengths, //lenght of the side of the domain in any direction
                                 const  std::vector<unsigned> &nPoints, //number of points per direction
                                 const int  &pointsD // number of mesh points over the external Dirichlet BC area in every direction
                                     ){
  if(lengths.size() != nPoints.size()){
    std::cerr<<"InitPointStructureNLBC: lengths and nPoints vectors sizes not matching";
    abort();
  }
  else{
    unsigned dim = nPoints.size();
    unsigned totNodes = 1;
    std::vector<double> h(lengths.size());
    std::vector<double> nPointsWithBC(lengths.size());
    for(unsigned k = 0; k < dim; k++) {
      nPointsWithBC[k] = nPoints[k] + 2 * pointsD;
      totNodes *= (nPointsWithBC[k]);
      if(nPoints[k] == 0 || nPoints[k] == 1) {std::cerr<< "InitializeSimplestPointStructure: zero or one point along a direction initialized";}
      h[k] = lengths[k] / (nPoints[k] - 1);
    }
    SetNumberOfNodes(totNodes);
    SetDimension(dim);
    xResize();

    std::vector<std::vector<double>> coord(_dim);
    for(unsigned d = 0; d < _dim; d++) coord[d].resize(nPointsWithBC[d]);
    for(unsigned d = 0; d < _dim; d++){
      for(int i = 0; i < nPointsWithBC[d]; i++){
        coord[d][i] = h[d] * (i - pointsD);
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

void Nom::SetConstDeltaV(std::vector<double> dimensions){
  if(dimensions.size() != _dim){
    std::cerr << "In function SetConstDeltaV: wrong dimensions\n";
    abort();
  }
  double volume = 1.;
  for(unsigned d = 0; d < dimensions.size(); d++) volume *= dimensions[d];
  _deltaV.resize(_nNodes, volume/_nNodes);
}

// TODO This function sets a field with the same node numbering of the mesh
void Nom::SetField(std::vector<double> field){
  _field = field;
}

// This function sets all the nodes recorded in dirNodes with a Dirichlet BC (_dirBC=1)
void Nom::SetBC(std::vector<unsigned> dirNodes){
  _dirBC.resize(_nNodes, 0);
  for(unsigned i = 0; i < dirNodes.size(); i++){
    _dirBC[dirNodes[i]] = 1;
  }
}

void Nom::GetCoords(std::vector<std::vector<double>> &x){
  x.resize(_x.size(), std::vector<double>(_x[0].size()));  
  for(unsigned i = 0; i < _x.size(); i++){
    for(unsigned k = 0; k < _x[0].size(); k++){
      x[i][k] =  _x[i][k];
    }
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

void Nom::InitializeMap(){
  for(unsigned i = 0; i < _nNodes; i++) {
    _suppNodes[i].resize(_nNodes);
    _suppNodesN[i].resize(_nNodes);
    _h.resize(_nNodes,0.);
  }
}

void Nom::InitializeDistMap(){
  for(unsigned i = 0; i < _nNodes; i++) {
      _suppDist[i].resize(_nNodes, std::vector<double>(_dim));
  }
}

// TODO
void Nom::SetConstantSupport(double delta){ _delta = delta;}

// This function fills BOTH the maps of the neighbours elements and of the distances
// given a constant support delta
void Nom::PointsAndDistInConstantSupport(){
  InitializeMap();
  InitializeDistMap();
  _count.resize(_nNodes, 0);  
  for(unsigned i = 0; i < _nNodes; i++){
    for(unsigned j = i+1; j < _nNodes; j++){
      double dist = 0;
      for(unsigned d = 0; d < _dim; d++) dist += (_x[i][d] - _x[j][d]) * (_x[i][d] - _x[j][d]);
      if(dist <= _delta * _delta) {
        _suppNodes[i][_count[i]] = j;
        _suppNodes[j][_count[j]] = i;
        _suppNodesN[i][_count[i]] = {j,_delta};
        _suppNodesN[j][_count[j]] = {i,_delta};
        for(unsigned d = 0; d < _dim; d++) {
          _suppDist[i][_count[i]][d] = _x[j][d] - _x[i][d];
          _suppDist[j][_count[j]][d] = _x[i][d] - _x[j][d];
        }
        _count[i]++;
        _count[j]++;
      }
    }
  }
  for(unsigned i = 0; i < _nNodes; i++) {
    _suppNodes[i].resize(_count[i]);
    _suppNodesN[i].resize(_count[i]);
    _suppDist[i].resize(_count[i]);
    _h[i]=_delta;
  }
}

// This function fills BOTH the maps of the neighbours elements and of the distances
// given an assigned number of points in the support
void Nom::PointsAndDistNPtsSupport(unsigned npt){
  InitializeMap();
  InitializeDistMap();
  for(unsigned i = 0; i < _nNodes; i++){
    // _suppNodesN[i].resize(_nNodes, {1.e5, 1.e5});
    for(unsigned j = 0; j < _nNodes; j++){
      if(i != j){
        double dist = 0;
        for(unsigned d = 0; d < _dim; d++) dist += (_x[i][d] - _x[j][d]) * (_x[i][d] - _x[j][d]);
          _suppNodesN[i][j] = {j, sqrt(dist)};
      }
      else{
        _suppNodesN[i][j] = {j, 1e5};
      }
    }
  }
  
  for(unsigned i = 0; i < _nNodes; i++){
    std::sort( _suppNodesN[i].begin(),  _suppNodesN[i].end(), [=](std::pair<int, double>& a, std::pair<int, double>& b){return a.second < b.second;});
    _suppNodesN[i].resize(npt);
  }

  for(unsigned i = 0; i < _nNodes; i++){
    _suppDist[i].resize(npt);
    for(unsigned j = 0; j < npt; j++){
      for(unsigned d = 0; d < _dim; d++) _suppDist[i][j][d] = _x[_suppNodesN[i][j].first][d] - _x[i][d];
    }
    double distMax = 0;
    double distMin = 0;
    for(unsigned d = 0; d < _dim; d++) {
      distMax += _suppDist[i][npt-1][d] * _suppDist[i][npt-1][d];
      distMin += _suppDist[i][0][d] * _suppDist[i][0][d];
    }
//     _h[i] = (sqrt(distMax) + sqrt(distMin)) / 2.;
    _h[i] = 1.1 * sqrt(distMax);
  }
}

// This function fills BOTH the maps of the neighbours elements and of the distances
void Nom::PointsAndDistInConstantSupportWithInv(){
  InitializeMap();
  InitializeDistMap();
  _count.resize(_nNodes, 0);
  for(unsigned i = 0; i < _nNodes; i++){
    std::map<int, int> innerMap;
    for(unsigned j = 0/*i+1*/; j < _nNodes; j++){
      if(i != j){
        double dist = 0;
        for(unsigned d = 0; d < _dim; d++) dist += (_x[i][d] - _x[j][d]) * (_x[i][d] - _x[j][d]);
        if(dist <= _delta * _delta) {
          _suppNodes[i][_count[i]] = j;
          innerMap[j] = _count[i];
          for(unsigned d = 0; d < _dim; d++) {
            _suppDist[i][_count[i]][d] = _x[j][d] - _x[i][d];
          }
          _count[i]++;
        }
      }
    }
    _suppNodesInv[i] = innerMap;
  }

  for(unsigned i = 0; i < _nNodes; i++) {
    _suppNodes[i].resize(_count[i]);
    _suppDist[i].resize(_count[i]);
  }
}

std::map<int, std::vector<int>> Nom::GetMap(){
  return _suppNodes;  
}

std::map<int, std::vector<std::pair<int,double>>> Nom::GetMapN(){
  return _suppNodesN;
}

std::map<int, std::vector<std::vector<double>>> Nom::GetDist(){
  return _suppDist;    
}

std::vector<std::vector<double>> Nom::GetK(){
  return _K;    
}

std::vector<std::vector<double>> Nom::GetKinv(){
  return _Kinv;    
}

std::vector<std::vector<double>> Nom::GetKHO(){
  return _KHO;    
}

Eigen::MatrixXd Nom::GetKHOE(){
  return _KHOE;
}

Eigen::MatrixXd Nom::GetPolyE(){
  return _PolyE;
}

Eigen::MatrixXd Nom::GetB(){
  return _BE;
}

// This function computes the operator K on a node i in NOM theory when weight_i = 1 / V_i
void Nom::ComputeOperatorK(unsigned i){
  _K.resize(_dim, std::vector<double>(_dim));  
  for(int i=0; i< _K.size(); i++) std::fill(_K[i].begin(),_K[i].end(),0.);
  for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
    for(unsigned d1 = 0; d1 < _dim; d1++){
      for(unsigned d2 = 0; d2 < _dim; d2++){
        _K[d1][d2] += _suppDist[i][j][d1] * _suppDist[i][j][d2];
      }
    }
  }  
}

// This function computes the operator K on a node i in NOM theory when weight_i != 1 / V_i
void Nom::ComputeNotHomOperatorK(unsigned i, std::vector<double> vol, std::map<int, std::vector<double>> weight){
  InitializeVolumesAndWeights(vol, weight);
  _K.resize(_dim, std::vector<double>(_dim, 0.));  
  for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
    for(unsigned d1 = 0; d1 < _dim; d1++){
      for(unsigned d2 = 0; d2 < _dim; d2++){
        _K[d1][d2] += _suppDist[i][j][d1] * _suppDist[i][j][d2] * _weight[i][j] * _vol[i];
      }
    }
  }  
}

void Nom::ComputeInvK(unsigned i){
  ComputeOperatorK(i);  
  SimpleMatrix _SM(_K);  
  _SM.inverse();
  _Kinv = _SM.getInv();    
}

void Nom::InitializeVolumesAndWeights(std::vector<double> vol, std::map<int, std::vector<double>> weight){
  if(vol.size() != _nNodes || weight.size() != _nNodes){
    std::cerr<<"Not consinstent number of weights or volumes in function InitializeVolumesAndWeights\n";
    abort();
  } 
  _vol = vol;  
  _weight = weight;
  for(unsigned i = 0; i < vol.size(); i++){
    if(vol[i] * vol[i] < 1e-20){
      std::cerr<<"Volume = 0 in function InitializeVolumesAndWeights\n";  
      abort();
    }   
  }
}

double Nom::ComputeNOMDivergence(std::vector<std::vector<double>> vec, unsigned i){
  double div = 0;  
  if(_dim != vec[0].size()){
    std::cerr << "In function ComputeNOMDivergence: dimension of vec not consinstent with _dim";
    abort();    
  }
  else{
    ComputeInvK(i);
    SimpleMatrix _SM(_Kinv); 
    std::vector<double> Km1r(_dim, 0.);
    for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
      Km1r = _SM.matVecMult(_suppDist[i][j]);
      for(unsigned d = 0; d < _dim; d++) div += (vec[_suppNodesN[i][j].first][d] - vec[i][d]) * (Km1r[d]);
    }
  }
}

std::vector<double> Nom::ComputeNOMScalarGradient(std::vector<double> sol, unsigned i){ 
 std::vector<double> grad(_dim, 0.);  
      ComputeInvK(i);
    SimpleMatrix _SM(_Kinv); 
    std::vector<double> sum(_dim, 0.);
    for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
      for(unsigned d = 0; j < _dim; d++){  
        sum[d] += sol[j] * _suppDist[i][j][d]; 
      }
    }
    grad = _SM.vecMatMult(sum);   
  return grad;
}

void Nom::SetOrder(unsigned n){
  _n = n;  
}

// This function generates all the combinations indices for derivatives with
// order less or equal then n
void Nom::combinationUtil(int arr[], int data[],
                    int index, int r) {
    if (index == r) { 
        unsigned sum = 0;
        for (int j = 0; j < r; j++) sum += data[j];
        if(sum != 0 && sum <= _n ) {
          for (int j = 0; j < r; j++) _multiIndexList[_cnt][j] = data[j];
          _cnt++;
        }

        return; 
    } 
    for (int i = 0; i <= _n; i++) { 
        data[index] = arr[i]; 
        combinationUtil(arr, data, index+1, r); 
    } 
} 

// This function creates the list _multiIndexList (alpha on the paper) of all
// the possible derivatives, depending on the oder n and the dimension _dim
void Nom::MultiIndexList(unsigned n){
  SetOrder(n);  
  _indxDim = factorial(_n + _dim) / (factorial(_n) * factorial(_dim)) - 1;
  _I = Eigen::MatrixXd::Identity(_indxDim, _indxDim);
  
  _multiIndexList.resize(_indxDim, std::vector<int>(_dim));
  _cnt = 0;
  
  int arr[_n + 1];
  for(unsigned nn = 0; nn <= _n; nn++) arr[nn] = nn;
  
  int data[_dim]; 
 
  combinationUtil(arr, data, 0, _dim); 
}

unsigned Nom::factorial(unsigned n) {
    if (n == 0 || n == 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

std::vector<std::vector<int>> Nom::GetMultiIndexList(){
  return _multiIndexList;
}

// j = 0,1,2,.. indicates the second index of _suppNodes (is NOT the actual value of the node)
std::vector<double> Nom::PolyMultiIndex(unsigned i, unsigned j){
  std::vector<double> poly(_indxDim,1.);

  for(unsigned indx = 0; indx < _indxDim; indx++){
    for(unsigned d = 0; d < _dim; d++){
     if(_multiIndexList[indx][d] > 0){
       poly[indx] *=  pow(_suppDist[i][j][d] / _h[i], _multiIndexList[indx][d]);
     }
    }
  }
  return poly;
}

// Computation of the inverse of diagonal of the matrix H composed by the permutations of the
// powers of the lengths
Eigen::MatrixXd Nom::DiagLengthHInv(unsigned i){
  Eigen::MatrixXd Hinv;
  Hinv.resize(_indxDim,_indxDim);
  Hinv.setZero();

  for(unsigned indx = 0; indx < _indxDim; indx++){
    double prod = 1.;
    for(unsigned d = 0; d < _dim; d++){
     if(_multiIndexList[indx][d] > 0){
       double num = pow(_h[i], _multiIndexList[indx][d]);
       double den = factorial(_multiIndexList[indx][d]);
       prod *= den / num;
     }
    }
    Hinv(indx,indx) = prod;
  }
  return Hinv;
}

Eigen::MatrixXd Nom::SelfTensProd(std::vector<double> vec){
  unsigned sz = vec.size();
  Eigen::MatrixXd sol;
  sol.resize(sz,sz);
  sol.setZero();
  for(unsigned i = 0; i < sz; i++){
    for(unsigned j = 0; j < sz; j++){
      sol(i,j) += vec[i] * vec[j];
    }
  }
  return sol;
}

void Nom::ComputeHighOrdOperatorK(unsigned i){
  if(_suppNodesN[i].size() < _indxDim){
    std::cerr<< "In function ComputeHighOrdOperatorK: not enough nodes in the support\n";
    abort();
  }
  else{
    _KHOE.resize(_indxDim,_indxDim);
    _KHOE.setZero();
    std::vector<double> polyIndx;
    _HinvE = DiagLengthHInv(i);
    Eigen::MatrixXd tensProd;
    for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
      polyIndx = PolyMultiIndex(i, j);
      tensProd = SelfTensProd(polyIndx);
      _KHOE = _KHOE + tensProd;
    }

    // _KHOE=_KHOE.inverse();
    _KHOE = _KHOE.llt().solve(_I);  // Cholesky Decomposition
    _KHOE=_HinvE*_KHOE;
  }
}

void Nom::ComputePolyOperator(unsigned i){
  if(_suppNodesN[i].size() < _indxDim){
    std::cerr<< "In function PolyOperator: not enough nodes in the support\n";
    abort();
  }
  else{
    _PolyE.resize(_indxDim,_suppNodesN[i].size());
    _PolyE.setZero();
    std::vector<double> polyIndx;
    for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
      polyIndx = PolyMultiIndex(i, j);
      for(unsigned jj = 0; jj < _indxDim; jj++){
        _PolyE(jj, j) = polyIndx[jj];
      }
    }
  }
}

// This function computes both the operators K_i and p^h_{wi}
void Nom::ComputeHighOrdKAndPolyOperators(unsigned i){
  if(_suppNodesN[i].size() < _indxDim){
    std::cerr<< "In function ComputeHighOrdKAndPolyOperators: not enough nodes in the support | supp Nodes = " << _suppNodesN[i].size() << " _indxDim = " << _indxDim <<"\n";
    abort();
  }
  else{
    _KHOE.resize(_indxDim,_indxDim);
    _KHOE.setZero();
    _PolyE.resize(_indxDim,_suppNodesN[i].size());
    _PolyE.setZero();
    std::vector<double> polyIndx;
    _HinvE = DiagLengthHInv(i);
    Eigen::MatrixXd tensProd;

    std::vector<double> weights(_suppNodesN[i].size());
    double sumWeight = 0.;
    for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
      double dist = 0.;
      for(unsigned d = 0; d < _dim; d++) dist += (_suppDist[i][j][d] * _suppDist[i][j][d]);
      dist = sqrt(dist);
      double r = dist / _h[i];
      weights[j] = ((r<1)?pow(1-r, 5) : 0)- 6 * ((r<2./3.)? pow(2./3. - r,5) : 0) + 15 * ((r<1./3.)? pow(1./3. - r,5) : 0);
//       weights[j] = (1. / pow(dist,_dim));
      sumWeight += weights[j];
    }

    for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
      polyIndx = PolyMultiIndex(i, j);
      for(unsigned jj = 0; jj < _indxDim; jj++){
        _PolyE(jj, j) = (weights[j] / sumWeight) * polyIndx[jj] /** _deltaV[_suppNodesN[i][j].first]*/;
      }
      tensProd =  (weights[j] / sumWeight) * SelfTensProd(polyIndx) /** _deltaV[_suppNodesN[i][j].first]*/;
      _KHOE = _KHOE + tensProd;
    }

    double scaleK =  1. / pow(_KHOE.determinant(), 1./ _KHOE.rows());
    double scaleHinv =  1. / pow(_HinvE.determinant(), 1./ _HinvE.rows());

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(_KHOE);
    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

    std::cout  << "i and Cond: " << i << " " <<cond<<std::endl;

    _KHOE=(scaleK*_KHOE).inverse();
    _KHOE=scaleHinv*_HinvE*_KHOE;

    _scale[i] = (1. / scaleK) * scaleHinv;

  }
}

void Nom::ComputeOperatorB(unsigned i){
  _BE.resize(_indxDim,_suppNodesN[i].size() + 1);
  _BE.setZero();
  Eigen::VectorXd colSum(_indxDim);
  colSum.setZero();
  Eigen::MatrixXd prodKP;

  ComputeHighOrdKAndPolyOperators(i);
  prodKP = _KHOE * _PolyE;
  for(unsigned j = 0; j < prodKP.cols(); j++){
    for(unsigned i = 0; i < prodKP.rows(); i++){
      colSum(i) -= prodKP(i,j);
    }
  }
  _BE << colSum,prodKP;
}

Eigen::VectorXd Nom::ComputeHighOrdDer(unsigned i){
  ComputeOperatorB(i);
  Eigen::VectorXd neighVal(_suppNodesN[i].size()+1);
  neighVal(0) = _field[i];
  for(unsigned j = 0; j < _suppNodesN[i].size(); j++){
    neighVal(j+1) = _field[_suppNodesN[i][j].first];
  }
  return _BE*neighVal;
}

void Nom::AssembleLaplacianNode(unsigned i){
  std::vector<unsigned> coeff(_indxDim, 0);
  std::vector<int> secDer(_dim, 0);
  unsigned cnt = 0;
  for(unsigned indx = 0; indx < _indxDim; indx++){
    for(unsigned d = 0; d < _dim; d++){
      secDer[d] = 2;
      if(secDer == _multiIndexList[indx]) {
        coeff[cnt] = indx;
        cnt++;
      }
      secDer[d] = 0;
    }
  }
  coeff.resize(cnt);
  if(cnt != _dim){
    std::cerr << "Not consinstent numer of derivatives found in function AssembleLaplacian\n";
    abort();
  }
  else{
    ComputeOperatorB(i);
    for(unsigned rowB = 0; rowB < coeff.size(); rowB++){
      _ME(i,i) += _BE(coeff[rowB],0);
      for(unsigned j = 1; j < _BE.cols(); j++){
        _ME(i,_suppNodesN[i][j-1].first) += _BE(coeff[rowB],j);
      }
    }
  }
}

void Nom::AssembleLaplacian(){
  _startTime = clock();
  _scale.resize(_nNodes, 1.);
  for(unsigned i = 0; i < _nNodes; i++){
    if(_dirBC[i] == 0){
      AssembleLaplacianNode(i);
    }
    else{
      _ME(i,i) = _penalty;
    }
  }
  std::cout<<"\n --- END ASSEMBLY --- \n";
  std::cout << "Laplacian Assembly time = " << static_cast<double>(clock() - _startTime) / CLOCKS_PER_SEC << std::endl;
}

void Nom::CreateGlobalEigenMatrix(){
  _ME.resize(_nNodes,_nNodes);
  _ME.setZero();
}

void Nom::CreateGlobalEigenRhs(){
  _rhsE.resize(_nNodes);
  _rhsE.setZero();
}

void Nom::SetEigenRhs(std::vector<double> rhs){
  if(rhs.size() != _nNodes){
    std::cerr << "In function SetEigenRhs: the vector rhs size is wrong\n";
    abort();
  }
  else{
    for(unsigned i = 0; i < _nNodes; i++){
      if(_dirBC[i] == 0){
        _rhsE(i) = _scale[i] * rhs[i];
      }
      else{
        _rhsE(i) = 0; // TODO only homogeneous Dir BC implemented
        // for(unsigned d = 0; d < _dim; d++){
        //   _rhsE(i) += _penalty * _x[i][d] * _x[i][d]; // TODO only non-homogeneous Dir BC implemented
        // }
      }
    }
  }
}

void Nom::SolveEigen(){
  _solE = _ME.inverse() * _rhsE;
}


void Nom::PrintGlobalEigenMatrix(){
  std::cout<< std::endl <<"Global Matrix: \n" << _ME << std::endl;
}

void Nom::PrintGlobalEigenRhs(){
  std::cout<< std::endl <<"RHS: \n" << _rhsE << std::endl;
}

void Nom::PrintGlobalEigenSolution(){
  std::cout<< std::endl <<"Solution: (coords | solution)\n";
  for(unsigned i = 0; i < _nNodes; i++){
    for(unsigned d = 0; d < _dim; d++) std::cout << _x[i][d] << "    |    ";
    std::cout << _solE(i) << "  |  " << _anSol[i] <<"\n";
  }
}

void Nom::SetAnalyticSol(std::vector<double> sol){
  _anSol = sol;
}

double Nom::L2Error(){
  double norm = 0.;
  if(_anSol.size() != _solE.size()){
    std::cerr << "In L2Error: the solution and the analytic solution don't have the same size \n";
    abort();
  }
  else{
    double num = 0.;
    double den = 0.;
    for(unsigned i = 0; i < _solE.size(); i++){
      num += (_solE(i) - _anSol[i]) * (_solE(i) - _anSol[i]) * _deltaV[i];
      den += _anSol[i] * _anSol[i] * _deltaV[i];
    }
    norm = sqrt(num / den);
  }
  return norm;
}


double Nom::GetKernel(unsigned i, unsigned j, double s){
//   double dist = _suppNodesN[i][j].second;
//   double Cns = s * pow(2.,2.*s) * tgamma ((_dim + 2. * s) / 2.) / (pow(M_PI, 0.5*_dim) * tgamma(1 - s));
//   double kernel = Cns / (2. * pow(dist, _dim + 2 * s)); // fractional laplacian kernel
  
  double kernel = 1.; //4. / (M_PI  * _delta * _delta * _delta * _delta);
  return kernel;
}

void Nom::AssembleNonLocalKernelNode(unsigned i, double s){
  for(unsigned j = 0; j < _suppNodesN[i].size(); j++) {
    double kernel = GetKernel(i,j,s);
    _ME(i,_suppNodesN[i][j].first) += 2. * kernel; 
    _ME(i,i) -= 2. * kernel;
  }
}

void Nom::AssembleNonLocalKernelEigen(double s){
    _scale.resize(_nNodes, 1.);
  for(unsigned i = 0; i < _nNodes; i++){
    if(_dirBC[i] == 0){
      AssembleNonLocalKernelNode(i, s);
    }
    else{
      _ME(i,i) = _penalty;
    }
  }
  std::cout<<"\n --- END ASSEMBLY NONLOCAL KERNEL--- \n";
  
}









void Nom::CreateGlobalMatrix(){
  PetscInt m = _nNodes;
  PetscInt n = _nNodes;
  PetscInt nz = _nNodes;
  PetscInt nnz[_nNodes];
  for(unsigned i = 0; i < _nNodes; i++) nnz[i] = _suppNodesN[i].size();
  MatCreateSeqAIJ(MPI_COMM_WORLD, m, n, nz, nnz, &_A);
}



} // end namespace femus
