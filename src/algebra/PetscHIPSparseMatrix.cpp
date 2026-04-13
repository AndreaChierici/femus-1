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

    PetscInt pm = static_cast<PetscInt>(m), pn = static_cast<PetscInt>(n);
    PetscInt pml = static_cast<PetscInt>(m_l), pnl = static_cast<PetscInt>(n_l);
    PetscInt pnnz = static_cast<PetscInt>(nnz), pnoz = static_cast<PetscInt>(noz);

    if(numprocs == 1) {
      assert((m_l == m) && (n_l == n));
      ierr = MatCreateSeqAIJ(MPI_COMM_WORLD, pm, pn, pnnz, PETSC_NULLPTR, &_mat);
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
      ierr = MatSetSizes(_mat, pml, pnl, pm, pn);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATMPIAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatMPIAIJSetPreallocation(_mat, pnnz, PETSC_NULLPTR, pnoz, PETSC_NULLPTR);
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
      std::vector<PetscInt> p_nnz(n_nz.begin(), n_nz.end());
      ierr = MatCreateSeqAIJ(MPI_COMM_WORLD, _m, _n, 0, p_nnz.data(), &_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATSEQAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetFromOptions(_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }
    else {
      parallel_only();
      assert(static_cast<int>(n_nz.size()) == _m_l && static_cast<int>(n_oz.size()) == _m_l);
      std::vector<PetscInt> p_nnz(n_nz.begin(), n_nz.end());
      std::vector<PetscInt> p_noz(n_oz.begin(), n_oz.end());
      ierr = MatCreate(MPI_COMM_WORLD, &_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetSizes(_mat, _m_l, _n_l, _m, _n);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetType(_mat, MATMPIAIJHIPSPARSE);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatMPIAIJSetPreallocation(_mat, 1, p_nnz.data(), 100, p_noz.data());
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }
    this->zero();
  }

// -----------------------------------------------------------------------
  // NOTE: parameter names corrected to match the declaration in
  // PetscHIPSparseMatrix.hpp / SparseMatrix.hpp.  The 5th positional
  // arg is the off-diagonal count (n_oz) and the 6th is the diagonal
  // count (n_nz).  Previously these names were swapped, which caused
  // MatMPIAIJSetPreallocation to receive diagonal/off-diagonal hints
  // in the wrong order.
  void PetscHIPSparseMatrix::update_sparsity_pattern(
    int m_global, int n_global,
    int m_local, int n_local,
    const std::vector<int> n_oz,
    const std::vector<int> n_nz) {

    if(this->initialized()) this->clear();
    this->_is_initialized = true;

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    int ierr = 0;

    std::vector<PetscInt> p_nnz(n_nz.begin(), n_nz.end());
    std::vector<PetscInt> p_noz(n_oz.begin(), n_oz.end());

    if(numprocs == 1) {
      assert((m_local == m_global) && (n_local == n_global));
      if(n_nz.empty())
        ierr = MatCreateSeqAIJ(MPI_COMM_WORLD, m_global, n_global,
                               PETSC_DEFAULT, PETSC_NULLPTR, &_mat);
      else
        ierr = MatCreateSeqAIJ(MPI_COMM_WORLD, m_global, n_global,
                               PETSC_DEFAULT, p_nnz.data(), &_mat);
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
        ierr = MatMPIAIJSetPreallocation(_mat, 0, p_nnz.data(), 0, p_noz.data());
      }
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      ierr = MatSetFromOptions(_mat);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
    }

    this->zero();
  }


void PetscHIPSparseMatrix::close() const {
    parallel_only();
    int ierr = 0;
    ierr = MatAssemblyBegin(_mat, MAT_FINAL_ASSEMBLY);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatAssemblyEnd(_mat, MAT_FINAL_ASSEMBLY);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

      const char* t;
      MatGetType(_mat, &t);
      if (std::string(t) != MATMPIAIJHIPSPARSE && std::string(t) != MATSEQAIJHIPSPARSE) {
	      int numprocs;
	      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	      const MatType gpuType = (numprocs == 1) ? MATSEQAIJHIPSPARSE : MATMPIAIJHIPSPARSE;
	      
	      Mat& matRef = const_cast<Mat&>(_mat);
	      ierr = MatConvert(matRef, gpuType, MAT_INPLACE_MATRIX, &matRef);
	      CHKERRABORT(MPI_COMM_WORLD, ierr);
      }
}


void PetscHIPSparseMatrix::zero() {
 assert(this->initialized());
  MatSetOption(_mat, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
  MatSetOption(_mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  int ierr = MatZeroEntries(_mat);
  CHKERRABORT(MPI_COMM_WORLD, ierr);
  // No assembly call - just zero the entries
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
    this->clear();

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    const MatType cpuType = (numprocs == 1) ? MATSEQAIJ : MATAIJ;
    const MatType gpuType = (numprocs == 1) ? MATSEQAIJHIPSPARSE : MATMPIAIJHIPSPARSE;

    Mat Acpu, Pcpu;
    ierr = MatConvert(const_cast<PetscMatrix*>(A)->mat(), cpuType, MAT_INITIAL_MATRIX, &Acpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatConvert(const_cast<PetscMatrix*>(P)->mat(), cpuType, MAT_INITIAL_MATRIX, &Pcpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    Mat resultCpu;
    ierr = MatPtAP(Acpu, Pcpu, MAT_INITIAL_MATRIX, 1.0, &resultCpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    ierr = MatConvert(resultCpu, gpuType, MAT_INITIAL_MATRIX, &_mat);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    ierr = MatDestroy(&resultCpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatDestroy(&Acpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatDestroy(&Pcpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    this->_is_initialized = true;
    { PetscInt pm, pn; MatGetSize(_mat, &pm, &pn); _m = static_cast<int>(pm); _n = static_cast<int>(pn); }
    { PetscInt pml, pnl; MatGetLocalSize(_mat, &pml, &pnl); _m_l = static_cast<int>(pml); _n_l = static_cast<int>(pnl); }
    _destroy_mat_on_exit = true;
  }

  void PetscHIPSparseMatrix::matrix_ABC(const SparseMatrix &mat_A,
                                        const SparseMatrix &mat_B,
                                        const SparseMatrix &mat_C,
                                        const bool &mat_reuse) {

    const PetscMatrix* A = static_cast<const PetscMatrix*>(&mat_A);
    A->close();
    const PetscMatrix* B = static_cast<const PetscMatrix*>(&mat_B);
    B->close();
    const PetscMatrix* C = static_cast<const PetscMatrix*>(&mat_C);
    C->close();

    int ierr;
    this->clear();

    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    const MatType cpuType = (numprocs == 1) ? MATSEQAIJ : MATAIJ;
    const MatType gpuType = (numprocs == 1) ? MATSEQAIJHIPSPARSE : MATMPIAIJHIPSPARSE;

    // Convert inputs to CPU AIJ for MatMatMatMult
    // (MatMatMatMult does not support HIPSparse operands)
    Mat Acpu, Bcpu, Ccpu;
    ierr = MatConvert(const_cast<PetscMatrix*>(A)->mat(), cpuType, MAT_INITIAL_MATRIX, &Acpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatConvert(const_cast<PetscMatrix*>(B)->mat(), cpuType, MAT_INITIAL_MATRIX, &Bcpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatConvert(const_cast<PetscMatrix*>(C)->mat(), cpuType, MAT_INITIAL_MATRIX, &Ccpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    // Compute R*K*P on CPU
    Mat resultCpu;
    ierr = MatMatMatMult(Acpu, Bcpu, Ccpu, MAT_INITIAL_MATRIX, 1.0, &resultCpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    // Convert result to HIPSparse
    ierr = MatConvert(resultCpu, gpuType, MAT_INITIAL_MATRIX, &_mat);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    ierr = MatDestroy(&resultCpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatDestroy(&Acpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatDestroy(&Bcpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    ierr = MatDestroy(&Ccpu);
    CHKERRABORT(MPI_COMM_WORLD, ierr);

    this->_is_initialized = true;
    { PetscInt pm, pn; MatGetSize(_mat, &pm, &pn); _m = pm; _n = pn; }
    { PetscInt pml, pnl; MatGetLocalSize(_mat, &pml, &pnl); _m_l = pml; _n_l = pnl; }
    _destroy_mat_on_exit = true;
  }

} //end namespace femus


#endif // HAVE_PETSC
