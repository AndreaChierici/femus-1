 #include "simpleMatrix.hpp"

namespace femus {
    
SimpleMatrix::SimpleMatrix() {

}
 
SimpleMatrix::~SimpleMatrix(){
     
}

SimpleMatrix::SimpleMatrix(std::vector<std::vector<double>> M){
    if(M.size() != M[0].size()){
      std::cerr<<"Not squared Matrix initialized in class SimpleMatrix\n";
      abort();
    }
  _sz = M.size();
  _M = M; 
}

void SimpleMatrix::setMatrix(std::vector<std::vector<double>> M){
  if(M.size() != M[0].size()){
    std::cerr<<"Not squared Matrix initialized in class SimpleMatrix\n";
    abort();
  }
  _sz = M.size();
  _M = M;   
}

std::vector<std::vector<double>> SimpleMatrix::getMatrix(){
  return _M;  
}
 
 
// Function to get cofactor of _M[p][q] in temp[][]. n is current dimension of _M
void SimpleMatrix::getCofactor(std::vector<std::vector<double>> Mat, int p, int q, int n)
{
    int i = 0, j = 0;
    
    _temp.resize(_sz, std::vector<double>(_sz));
 
    // Looping for each element of the matrix
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            //  Copying into temporary matrix only those
            //  element which are not in given row and
            //  column
            if (row != p && col != q) {
                _temp[i][j++] = Mat[row][col];
 
                // Row is filled, so increase row index and
                // reset col index
                if (j == n - 1) {
                    j = 0;
                    i++;
                }
            }
        }
    }
}
 
/* Recursive function for finding determinant of matrix.
   n is current dimension of A[][]. */
double SimpleMatrix::determinant(std::vector<std::vector<double>> Mat, int n)
{
    double D = 0; // Initialize result
 
    //  Base case : if matrix contains single element
    if (n == 1)
        return Mat[0][0];
 
    int sign = 1; // To store sign multiplier
 
    // Iterate for each element of first row
    for (int f = 0; f < n; f++) {
        // Getting Cofactor of A[0][f]
        getCofactor(Mat, 0, f, n);
        D += sign * Mat[0][f] * determinant(_temp, n - 1);
 
        // terms are to be added with alternate sign
        sign = -sign;
    }
 
    return D;
}
 
// Function to get adjoint of A[N][N] in adj[N][N].
void SimpleMatrix::adjoint()
{
    _adj.resize(_sz, std::vector<double>(_sz));
    if (_sz == 1) {
        _adj[0][0] = 1;
        return;
    }
 
    // temp is used to store cofactors of A[][]
    int sign = 1;
    _temp.resize(_sz, std::vector<double>(_sz, 0.));
 
    for (int i = 0; i < _sz; i++) {
        for (int j = 0; j < _sz; j++) {
            // Get cofactor of A[i][j]
            getCofactor(_M, i, j, _sz);
 
            // sign of adj[j][i] positive if sum of row
            // and column indexes is even.
            sign = ((i + j) % 2 == 0) ? 1 : -1;
 
            // Interchanging rows and columns to get the
            // transpose of the cofactor matrix
            _adj[j][i] = (sign) * (determinant(_temp, _sz - 1));
        }
    }
}
 
// Function to calculate and store inverse, returns false if
// matrix is singular
bool SimpleMatrix::inverse()
{
    _inv.resize(_sz, std::vector<double>(_sz, 0.));
    // Find determinant of A[][]
    double det = determinant(_M, _sz);
    if (det == 0) {
        cout << "Singular matrix, can't find its inverse";
        return false;
    }
 
    // Find adjoint
    adjoint();
 
    // Find Inverse using formula "inverse(A) =
    // adj(A)/det(A)"
    for (int i = 0; i < _sz; i++)
        for (int j = 0; j < _sz; j++)
            _inv[i][j] = _adj[i][j] / det;
 
    return true;
}
 
void SimpleMatrix::printInv()
{
    for (int i = 0; i < _sz; i++) {
        for (int j = 0; j < _sz; j++)
            cout << _inv[i][j] << " ";
        cout << endl;
    }
}

std::vector<std::vector<double>> SimpleMatrix::getInv(){
  return _inv;  
}

std::vector<double> SimpleMatrix::matVecMult(std::vector<double> vec){
  std::vector<double> result(_sz, 0.);  
  if(vec.size() != _sz){
    std::cerr << "vecMult: Size of the vector not consinstent with size of the matrix\n";
    abort();
  }
  else{
    for(unsigned i = 0; i < _sz; i++){
      for(unsigned j = 0; j < _sz; j++){
         result[i] +=  _M[i][j] * vec[j];    
      }  
    }
  }
  return result;
}

std::vector<double> SimpleMatrix::vecMatMult(std::vector<double> vec){
  std::vector<double> result(_sz, 0.);  
  if(vec.size() != _sz){
    std::cerr << "vecMult: Size of the vector not consinstent with size of the matrix\n";
    abort();
  }
  else{
    for(unsigned i = 0; i < _sz; i++){
      for(unsigned j = 0; j < _sz; j++){
         result[i] +=  vec[i] * _M[j][i] ;    
      }  
    }
  }
  return result;
}

void SimpleMatrix::Sum(std::vector<std::vector<double>> mat){
  if(mat.size() != _M.size() || mat[0].size() != _M[0].size()){
    std::cerr << "Error in SimpleMatrix::Sum: sum of matrices with different shapes\n";
    abort();
  }
  else{
    for(unsigned i = 0; i < _M.size(); i++){
        for(unsigned j = 0; j < _M[0].size(); j++){
          _M[i][j] += mat[i][j];
      }
    }
  }
  return;
}

// This function computes the product between the diagonal matrix identified by the diagonal diag
// and _M
void SimpleMatrix::DiagMatProd(std::vector<double> diag){
  if(_M.size() != diag.size()){
    std::cerr << "Error in SimpleMatrix::DiagMatProd: product of matrices with different shapes\n" ;
    abort();
  }
  else{
    for(unsigned i = 0; i < _M.size(); i++){
        for(unsigned j = 0; j < _M[0].size(); j++){
          _M[i][j] *= diag[i];
      }
    }  
  }
  return;
}


// void SimpleMatrix::GeneratePETSCMatrix(){
//   PetscInt M = _M.size();
//   PetscInt N = _M[0].size();
//   // MatCreateDense(MPI_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL,&_MPetsc);
//   MatCreateSeqAIJ(PETSC_COMM_SELF,M,N,0,NULL,&_MPetsc);
//   MatSetValues(&_MPetsc, M, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], const PetscScalar v[], InsertMode addv)
// }



} // end namespace femus
