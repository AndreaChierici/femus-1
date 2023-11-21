#include "FemusInit.hpp"
#include "MultiLevelSolution.hpp"
#include "MultiLevelProblem.hpp"
#include "VTKWriter.hpp"
#include "TransientSystem.hpp"
#include "LinearImplicitSystem.hpp"

#include "NumericVector.hpp"

#include "CurrentElem.hpp"
#include "ElemType_template.hpp"

#include "petsc.h"
#include "petscmat.h"
#include "PetscMatrix.hpp"

#include "PetscMatrix.hpp"

#include "include/nom.hpp"

using namespace femus;

int main(int argc, char** argv)
{
  Nom nom;
  std::vector<double> lengths{1.,1.};
  std::vector<unsigned> nPoints{2,2};
//   nom.InitializeSimplestPointStructure(lengths,nPoints);  
}
