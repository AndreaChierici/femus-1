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

#include "FemusConfig.hpp"

#ifdef HAVE_PETSC

#include "PetscHIPSparseMatrix.hpp"
#include "Parallel.hpp"
#include <mpi.h>

namespace femus {

// -----------------------------------------------------------------------
  void PetscHIPSparseMatrix::init(const int m, const int n,
                                   const int m_l, const int n_l,
                                   const int nnz, const int noz) {
    _m = m;
    _n = n;
    _m_l = m_l;
    _n_l = n_l;

    if(this->initialized()) this->clear();
    this->_is_initialized = true;

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int ierr = 0;

    if(numprocs == 1) {
      assert((m_l == m) && (n_l == n));
      ierr = MatCreateSeqAIJ(MPI_COMM_WORLD, m, n, nnz, PETSC_NULLPTR, &_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATSEQAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetFromOptions(_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }
    else {
      parallel_only();
      ierr = MatCreate(MPI_COMM_WORLD, &_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetSizes(_mat, m_l, n_l, m, n);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATMPIAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatMPIAIJSetPreallocation(_mat, nnz, PETSC_NULLPTR, noz, PETSC_NULLPTR);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }

    this->zero();
  }

// -----------------------------------------------------------------------
  void PetscHIPSparseMatrix::init(const int m, const int n,
                                   const int m_l, const int n_l,
                                   const std::vector<int> & n_nz,
                                   const std::vector<int> & n_oz) {
    _m = m;
    _n = n;
    _m_l = m_l;
    _n_l = n_l;

    if(this->initialized()) this->clear();
    this->_is_initialized = true;

    int n_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    int ierr = 0;

    if(n_procs == 1) {
      assert(static_cast<int>(n_nz.size()) == _m_l);
      ierr = MatCreateSeqAIJ(MPI_COMM_WORLD, _m, _n, 0, &n_nz[0], &_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATSEQAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetFromOptions(_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }
    else {
      parallel_only();
      assert(static_cast<int>(n_nz.size()) == _m_l && static_cast<int>(n_oz.size()) == _m_l);
      ierr = MatCreate(MPI_COMM_WORLD, &_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetSizes(_mat, _m_l, _n_l, _m, _n);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATMPIAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatMPIAIJSetPreallocation(_mat, 1, &n_nz[0], 100, &n_oz[0]);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }
    this->zero();
  }

// -----------------------------------------------------------------------
  void PetscHIPSparseMatrix::update_sparsity_pattern(
    int m_global, int n_global,
    int m_local, int n_local,
    const std::vector<int> n_nz,
    const std::vector<int> n_oz) {

    if(this->initialized()) this->clear();
    this->_is_initialized = true;

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int ierr = 0;

    if(numprocs == 1) {
      assert((m_local == m_global) && (n_local == n_global));
      if(n_nz.empty())
        ierr = MatCreateSeqAIJ(MPI_COMM_WORLD, m_global, n_global,
                               PETSC_DEFAULT, (int*) PETSC_NULLPTR, &_mat);
      else
        ierr = MatCreateSeqAIJ(MPI_COMM_WORLD, m_global, n_global,
                               PETSC_DEFAULT, (int*) &n_nz[0], &_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATSEQAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetFromOptions(_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }
    else {
      parallel_only();
      ierr = MatCreate(MPI_COMM_WORLD, &_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetSizes(_mat, m_local, n_local, m_global, n_global);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATMPIAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      if(n_nz.empty()) {
        ierr = MatMPIAIJSetPreallocation(_mat, 0, 0, 0, 0);
      }
      else {
        ierr = MatMPIAIJSetPreallocation(_mat, 0, (int*) &n_nz[0], 0, (int*) &n_oz[0]);
      }
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetFromOptions(_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }

    this->zero();
  }


// -----------------------------------------------------------------------
  void PetscHIPSparseMatrix::matrix_PtAP(const SparseMatrix &mat_P,
                                          const SparseMatrix &mat_A,
                                          const bool &mat_reuse) {
    const PetscMatrix* A = static_cast<const PetscMatrix*>(&mat_A);
    A->close();
    const PetscMatrix* P = static_cast<const PetscMatrix*>(&mat_P);
    P->close();

    int ierr;

    Mat Acpu, Pcpu;
    ierr = MatConvert(const_cast<PetscMatrix*>(A)->mat(), MATAIJ, MAT_INITIAL_MATRIX, &Acpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatConvert(const_cast<PetscMatrix*>(P)->mat(), MATAIJ, MAT_INITIAL_MATRIX, &Pcpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    this->clear();

    Mat resultCpu;
    ierr = MatPtAP(Acpu, Pcpu, MAT_INITIAL_MATRIX, 1.0, &resultCpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    const char *hipType = (numprocs == 1) ? MATSEQAIJHIPSPARSE : MATMPIAIJHIPSPARSE;

    ierr = MatConvert(resultCpu, hipType, MAT_INITIAL_MATRIX, &_mat);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatDestroy(&resultCpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    this->_is_initialized = true;
    MatGetSize(_mat, &_m, &_n);
    MatGetLocalSize(_mat, &_m_l, &_n_l);
    _destroy_mat_on_exit = true;

    ierr = MatDestroy(&Acpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatDestroy(&Pcpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
  }

} //end namespace femus


#endif // HAVE_PETSC
