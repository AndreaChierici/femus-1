/*=========================================================================

 Program: FEMUS
 Module: PetscHIPSparseMatrix
 Authors: based on PetscMatrix by Simone Bnà, Eugenio Aulisa, Giorgio Bornia

 Copyright (c) FEMTTU
 All rights reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __femus_algebra_PetscHIPSparseMatrix_hpp__
#define __femus_algebra_PetscHIPSparseMatrix_hpp__

#include "FemusConfig.hpp"

#ifdef HAVE_PETSC

#include "PetscMatrix.hpp"

namespace femus {

// =======================================================
// PETSc matrix backed by AIJHIPSPARSE (GPU-accelerated).
// Inherits all functionality from PetscMatrix; only the
// matrix creation (init / update_sparsity_pattern) is
// overridden to use HIPSPARSE matrix types.
// =======================================================

  class PetscHIPSparseMatrix : public PetscMatrix {

    public:
      PetscHIPSparseMatrix() : PetscMatrix() {}
      PetscHIPSparseMatrix(Mat m) : PetscMatrix(m) {}

      void init (const int m, const int n, const int m_l, const int n_l,
                 const int nnz = 0, const int noz = 0);
      void init (const  int m, const  int n, const  int m_l, const  int n_l,
                 const std::vector< int > & n_nz, const std::vector< int > & n_oz);

      void update_sparsity_pattern (int m, int n, int m_l, int n_l,
                                    const std::vector<int>  n_oz, const std::vector<int>  n_nz);

      void matrix_PtAP (const SparseMatrix &mat_P, const SparseMatrix &mat_A, const bool &reuse);
  };


} //end namespace femus


#endif // HAVE_PETSC
#endif // __femus_algebra_PetscHIPSparseMatrix_hpp__
