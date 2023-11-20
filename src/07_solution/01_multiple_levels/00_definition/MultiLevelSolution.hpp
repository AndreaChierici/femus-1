/*=========================================================================

Program: FEMuS
Module: MultiLevelSolution
Authors: Eugenio Aulisa, Simone Bnà, Giorgio Bornia

Copyright (c) FEMuS
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __femus_solution_MultiLevelSolution_hpp__
#define __femus_solution_MultiLevelSolution_hpp__

//----------------------------------------------------------------------------
// includes :
//----------------------------------------------------------------------------
#include "ParallelObject.hpp"
#include "Solution.hpp"
#include "MultiLevelMesh.hpp"
#include "FElemTypeEnum.hpp"
#include "BDCTypeEnum.hpp"
#include "Writer.hpp"
#include "FunctionBase.hpp"

#include <vector>
#include <memory>


namespace femus {


class MultiLevelProblem;

/**
 * This class is a black box container to handle multilevel solutions.
 */

class MultiLevelSolution : public ParallelObject {

// === Constructors / Destructor  - BEGIN =================
public:
    
    /** Constructor */
    MultiLevelSolution(MultiLevelMesh *ml_msh);

    /** Destructor */
    ~MultiLevelSolution();

    /** this is the destructor that can be called explicitly, instead of the automatic destructor */
    void clear();
 
// === Constructors / Destructor  - END =================
    
    
// === BASIC SOL MANIPULATION - BEGIN =================
public:
    /** To be Added */
    void AddSolution(const char name[], const FEFamily fefamily, const FEOrder order, unsigned tmorder = 0, const bool &Pde_type = 1);

    /** To be Added */
    void AddSolution(const char name[], const FEFamily fefamily, const FEOrder order_v,  const FEOrder order_b, unsigned tmorder = 0, const bool &Pde_type = 1);

    /** If you want to add a vector whose components are treated the same way */
    void AddSolutionVector(const unsigned n_components, const std::string name, const FEFamily fefamily, const FEOrder order, unsigned tmorder=0, const bool &Pde_type=1);

    /** To be Added */
    void AddSolutionLevel();

    /** To be Added */
    void ResizeSolution_par(const unsigned new_size);
    
    std::vector< unsigned > solution_start_and_end(const std::string name);
  
    /** Add one Solution to another, at all levels */
    void add_solution(const unsigned index_read, const unsigned index_write);
    
// === BASIC SOL MANIPULATION - END =================
    
// === NAME & INDEX  - BEGIN =================
public:
    /** To be Added */
    unsigned GetSolutionSize() {
        return _solType.size();
    }

    /** To be Added */
    const unsigned GetSolutionSize() const {
        return _solType.size();
    }
    
    
    /** To be Added */
    char* GetSolutionName(unsigned i) {
        return _solName[i];
    }

    /** To be Added */
    std::vector <char*>  GetSolName() {
        return _solName;
    }
    
    /** To be Added */
    std::vector <std::string>  GetSolName_string_vec() {
        
        std::vector <std::string> solName_strings(_solName.size());
        
              for(unsigned s = 0; s < solName_strings.size(); s++){
                  solName_strings[s] = _solName[s];
              }
        
        return solName_strings;
    }
    
    /** To be Added */
    unsigned GetIndex(const char name[]) const;


private:
    
    /** Vector size: number of added solutions. */
    std::vector < char* >                    _solName;

// === NAME & INDEX  - END =================

// === MESH, MULTILEVEL - BEGIN =================
public:
       /** duplicate of GetSolutionLevel, to be removed @todo */
    Solution* GetLevel(const unsigned i) {
        return _solution[i];
    };
    
       /** To be Added */
    Solution* GetSolutionLevel(const unsigned i) {
        return _solution[i];
    };

    /** To be Added */
    const Solution* GetSolutionLevel(const unsigned i) const {
        return _solution[i];
    };
    
    // member data
    MultiLevelMesh* _mlMesh;

     // *******************************************************

    void RefineSolution( const unsigned &gridf );
    void CoarsenSolutionByOneLevel_wrong( const unsigned &gridf );
    void CoarsenSolutionByOneLevel( const unsigned &gridf );
    
    void fill_at_level_from_level(const unsigned lev_out, const unsigned lev_in, const MultiLevelSolution & ml_sol_in);
        
private:
    
    /** Vector size: number of levels */
    std::vector < Solution* >  _solution;
    
    /** Number of levels */
    unsigned short  _gridn;
    

// === MESH, MULTILEVEL - END =================


// === SPACE DISCRETIZATION (FE) - BEGIN =================
public:
    
    FEFamily GetSolutionFamily(const unsigned& i){
      return _family[i];  
    }
    
    const FEFamily GetSolutionFamily(const std::string & sol_name) const;
    
    FEOrder GetSolutionOrder(const unsigned& i){
      return _order[i];    
    }
    
    const FEOrder GetSolutionOrder(const std::string & sol_name) const;

    /** To be Added */
    int GetSolutionType(unsigned i) {
        return _solType[i];
    }

    /** To be Added */
    const int GetSolutionType(unsigned i) const {
        return _solType[i];
    }
    
    /** To be Added */
    unsigned GetSolutionType(const char name[]);

    /** To be Added */
    unsigned GetSolType(const char name[]);

    /** To be Added */
    std::vector <int>  GetSolType() {
        return _solType;
    }
    
    
private:
    
    /** Vector size: number of added solutions. Tells the FE index */
    std::vector < int >                      _solType;
    /** Vector size: number of added solutions. */
    std::vector < FEFamily >                 _family;
    /** Vector size: number of added solutions. */
    std::vector < FEOrder >                  _order;
    
// === SPACE DISCRETIZATION (FE) - END =================


// === TIME EVOLUTION (NOT DISCRETIZATION) - BEGIN =================
public:
    /** To be Added */
    int   GetSolutionTimeOrder(unsigned i) {
        return _solTimeOrder[i];
    }

    const int   GetSolutionTimeOrder(const std::string & sol_name) const;

    void CopySolutionToOldSolution();

    
private:
    
    /** Vector size: number of added solutions. 0 = steady, 2 = time-dependent */
    std::vector < int >                      _solTimeOrder;
// === TIME EVOLUTION (NOT DISCRETIZATION) - END =================

// === INITIALIZATION (Initial Conditions) - BEGIN ===============
public:
    
    /** Initial condition function pointer typedef */
    typedef double (*InitFunc) (const std::vector < double >& x);

    /** duplicate */
    typedef double (*InitFuncMLProb) (const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[]);

    /** To be Added */
    void Initialize(const char name[], InitFunc func = NULL);

    void Initialize(const char * name, InitFuncMLProb func, const MultiLevelProblem * ml_prob);
    
    /** @todo At all levels, initialize Sol. By default, Sol is set to zero. Otherwise, a function is passed. 
      * In that case, and if the Solution is time-dependent, then both Sol and SolOld are initialized. */
    void Initialize(const char name[], InitFunc func, InitFuncMLProb funcMLProb, const MultiLevelProblem *ml_prob);
  
template < class LIST_OF_CTRL_FACES >
  void InitializeBasedOnControlFaces(const char name[], InitFuncMLProb func, const MultiLevelProblem* ml_prob) {
    InitializeBasedOnControlFaces<LIST_OF_CTRL_FACES>(name, NULL, func, ml_prob);
  }
    
    /** A Solution is by default initialized to zero, or by a provided function     */
template < class LIST_OF_CTRL_FACES >
  void InitializeBasedOnControlFaces(const char name[], InitFunc func, InitFuncMLProb funcMLProb, const MultiLevelProblem* ml_prob);
  
  
    inline void Set(const char name[], InitFuncMLProb funcMLProb, const MultiLevelProblem *ml_prob);
    
    void UpdateSolution(const char name[], InitFunc func, const double& time);
    
    
// === INITIALIZATION (Initial Conditions) - END =================
    
// === INITIALIZATION FUNCTION - BEGIN =================
public:
    
  void set_analytical_function(const char * name,  Math::Function< double > * func_in) {    
      const unsigned level_to_pick_from = 0; ///@todo
    GetSolutionLevel(level_to_pick_from)->set_analytical_function(name, func_in);
  }

   Math::Function< double > * get_analytical_function(const char * name) const {
             const unsigned level_to_pick_from = 0; ///@todo
    return GetSolutionLevel(level_to_pick_from)->get_analytical_function(name);

  }
  
// === INITIALIZATION FUNCTION - END =================
    
// === FILE OUTPUT - BEGIN =============
public:

    /** To be Added */
    Writer* GetWriter() {return _writer; }

    /** To be Added */
    const Writer* GetWriter() const {return _writer; }

    /** To be Added */
    void SetWriter(const WriterEnum format) { _writer = Writer::build(format,this).release(); }

    
    
private:
    
    /** Multilevel solution writer */
    Writer* _writer;

// === FILE OUTPUT - END ===============
    

// === RESTART - BEGIN =============

public:
    
    void SaveSolution(const char* filename, const double &time=0.);
    void SaveSolution(const char* filename, const unsigned &iteration);
    
    void LoadSolution(const char* filename);
    void LoadSolution(const unsigned &level, const char* filename);
    
// === RESTART - END =============

    
//=========== 
///==== @todo from now on, it is more stuff that shouldn't be in a basic Solution object      
//=========== 
      
// === Boundary Conditions - THERE IS TIME DEPENDENT STUFF HERE - BEGIN =================
public:
    
    /** Boundary condition function pointer typedef */
    typedef bool (*BoundaryFunc) (const std::vector < double >& x, const char name[], double &value, const int FaceName, const double time);
    
    /** duplicate */
    typedef bool (*BoundaryFuncMLProb) (const MultiLevelProblem * ml_prob, const std::vector < double >& x, const char name[], double &value, const int FaceName, const double time);

    
    /** To be Added */
    void AttachSetBoundaryConditionFunction( BoundaryFunc SetBoundaryConditionFunction );
    
    void AttachSetBoundaryConditionFunction( BoundaryFuncMLProb SetBoundaryConditionFunction );
    
    

    /** To be Added */
    void GenerateBdc(const char name[], const char bdc_type[] = "Steady", const MultiLevelProblem * ml_prob = NULL);


    /** To be Added */
    void UpdateBdc(const double time);

    /** To be Added */
    void GenerateBdc( const unsigned int k, const unsigned grid0, const double time );
    void GenerateRKBdc(const unsigned int &solIndex, const std::vector<unsigned> &solKiIndex, 
                       const unsigned int &grid0, const std::vector < double> & time,  const double &time0, 
                       const double &dt, const double* AI);
    
    /** for NONLOCAL problems, _Bdc must be 0 on all the volume constraint */
    void GenerateBdcOnVolumeConstraint(const std::vector<unsigned> &volumeConstraintFlags, const unsigned &solIndex, const unsigned &grid0);

    /** To be Added */
    BDCType GetBoundaryCondition(const std::string varname, const unsigned int facename) const;

    /** To be Added */
    bool Ishomogeneous(const std::string varname, const unsigned int facename) const;

    /** @deprecated */
    void InitializeBdc();
    
    /** @deprecated */
    void SetBoundaryCondition_new(const std::string name, const std::string facename, const BDCType bdctype = DIRICHLET,
                              const bool istimedependent = false, FunctionBase* func = NULL);

    /** To be Added */
    FunctionBase* GetBdcFunction(const std::string varname, const unsigned int facename) const;

    BoundaryFunc GetBdcFunction() {
      return _SetBoundaryConditionFunction;
    }

    BoundaryFunc GetBdcFunction() const {
      return _SetBoundaryConditionFunction;
    }

    BoundaryFuncMLProb GetBdcFunctionMLProb() {
      return _SetBoundaryConditionFunctionMLProb;
    }

    BoundaryFuncMLProb GetBdcFunctionMLProb() const {
      return _SetBoundaryConditionFunctionMLProb;
    }

    /** To be Added */
    char* GetBdcType(unsigned i) {
        return _bdcType[i];
    };

    bool _useParsedBCFunction;

    
    
private:
    
    /** To be Added */
    bool Ishomogeneous(const unsigned int var, const unsigned int facename) const;

    /** Vector size: number of added solutions. Inner vector size: number of faces of the domain boundary */
    std::vector < std::vector <BDCType> >         _boundaryConditions;
    /** Vector size: number of added solutions. Inner vector size: number of faces of the domain boundary. Says if the Boundary Condition is homogeneous */
    std::vector < std::vector <bool> >            _isHomogeneous;
    /** Vector size: number of added solutions. Inner vector size: number of faces of the domain boundary */
    std::vector < std::vector <FunctionBase *> >  _nonHomogeneousBCFunction;
    
    /** Vector size: number of added solutions. */
    std::vector < char* >                    _bdcType;
    
    /** boundary condition function pointer */
    BoundaryFunc _SetBoundaryConditionFunction;
    /** boundary condition function pointer */
    BoundaryFuncMLProb _SetBoundaryConditionFunctionMLProb;
    /** Flag to tell whether the BC function has been set */
    bool _bdcFuncSet;
    /** Flag to tell whether the BC function has been set */
    bool _bdcFuncSetMLProb;

    /** To be Added */
    BDCType GetBoundaryCondition(const unsigned int var, const unsigned int facename) const;

    /** To be Added */
    FunctionBase* GetBdcFunction(const unsigned int var, const unsigned int facename) const;

    /** Problem pointer for Boundary Conditions */
    const MultiLevelProblem* _mlBCProblem;
    
// === Boundary Conditions - THERE IS TIME DEPENDENT STUFF HERE - END =================


// === Solution as Unknown of System - BEGIN =================
    
private:
    
    /** Vector size: number of added solutions. Tells whether the Solution is an unknown of a PDE or not */
    std::vector < bool >                     _pdeType;
// === Solution as Unknown of System - END =================

   
// === NULL SPACE (of what here? for pressure variable, when you pinpoint it) - BEGIN =================
public:
    
    /** To be Added */
    void AssociatePropertyToSolution(const char solution_name[], const char solution_property[], const bool &bool_property = true);

  void FixSolutionAtOnePoint( const char sol[] ){
      _fixSolutionAtOnePoint[GetIndex(sol)] = true ;
      for(unsigned ig = 1; ig < _gridn; ig++){
        _solution[ig]->RemoveNullSpace(GetIndex(sol));
      }
    }
    /** To be Added */
    bool  TestIfSolutionIsPressure(unsigned i) {
        return _testIfPressure[i];
    }

private:
    /** Vector size: number of added solutions. */
    std::vector < bool >                     _testIfPressure;
    /** Vector size: number of added solutions. */
    std::vector < bool >                     _fixSolutionAtOnePoint;
    /** Vector size: number of added solutions. */
    std::vector < bool >                     _addAMRPressureStability;


// === NULL SPACE (of what here? for pressure variable, when you pinpoint it) - END =================

    
// === FSI - BEGIN =================
public:
    
    /** To be Added */
    unsigned GetSolutionPairIndex(const unsigned& i) const{
      return _solPairIndex[i];
    }
    
    unsigned GetSolutionPairInverseIndex(const unsigned& i) const{
      return _solPairInverseIndex[i];
    }
    
    
    /** To be Added */
    void PairSolution(const char solution_name[], const char solution_pair[]);

    
    void SetIfFSI(const bool &FSI = true){
	_FSI = FSI; 
	for(unsigned i=0;i<_gridn;i++){
	  _solution[i]->SetIfFSI(FSI);
	}
    }
      
    bool GetIfFSI(){
      return _FSI; 
    }
    
    
private:
    
    /** FSI: @todo this should be in a separate FSI environment */
    bool _FSI;

    /** Vector size: number of added solutions. */
    std::vector <unsigned>                   _solPairIndex;
    std::vector <unsigned>                   _solPairInverseIndex;
    
// === FSI - END =================

    

};



inline
BDCType MultiLevelSolution::GetBoundaryCondition(const unsigned int var, const unsigned int facename) const {
    return _boundaryConditions[var][facename];
}

inline
bool MultiLevelSolution::Ishomogeneous(const unsigned int var, const unsigned int facename) const {
    return _isHomogeneous[var][facename];
}

inline
FunctionBase* MultiLevelSolution::GetBdcFunction(const unsigned int var, const unsigned int facename) const {
    return _nonHomogeneousBCFunction[var][facename];
}

inline
BDCType MultiLevelSolution::GetBoundaryCondition(const std::string varname, const unsigned int facename) const {
    unsigned int var = GetIndex(varname.c_str());
    return _boundaryConditions[var][facename];
}

inline
bool MultiLevelSolution::Ishomogeneous(const std::string varname, const unsigned int facename) const {
    unsigned int var = GetIndex(varname.c_str());
    return _isHomogeneous[var][facename];
}

inline
FunctionBase* MultiLevelSolution::GetBdcFunction(const std::string varname, const unsigned int facename) const {
    unsigned int var = GetIndex(varname.c_str());
    return _nonHomogeneousBCFunction[var][facename];
}

inline 
void MultiLevelSolution::Set(const char * name, InitFuncMLProb funcMLProb, const MultiLevelProblem * ml_prob) {
    Initialize(name, funcMLProb, ml_prob);
}

} //end namespace femus



#include "NumericVector.hpp"  //this is here because the template needs to be in the header file (implicit instantiation)
#include "CurrentElem.hpp"

namespace femus {
    
 template < class LIST_OF_CTRL_FACES >
  void MultiLevelSolution::InitializeBasedOnControlFaces(const char name[], InitFunc func, InitFuncMLProb funcMLProb, const MultiLevelProblem* ml_prob) {

    

   std::vector< unsigned > sol_start_end =  solution_start_and_end(std::string (name));
      
      
    for(unsigned i = sol_start_end[0]; i < sol_start_end[1]; i++) {
      unsigned sol_type = _solType[i];

      for(unsigned ig = 0; ig < _gridn; ig++) {

        _solution[ig]->ResizeSolutionVector(_solName[i]);
        _solution[ig]->_Sol[i]->zero();

        if(func || funcMLProb) {
          double value = 0.;

          if(sol_type < NFE_FAMS_C_ZERO_LAGRANGE) {
              abort();
//             for(int isdom = _iproc; isdom < _iproc + 1; isdom++) {
//               for(int iel = _mlMesh->GetLevel(ig)->_elementOffset[isdom];
//                   iel < _mlMesh->GetLevel(ig)->_elementOffset[isdom + 1]; iel++) {
//                 unsigned nloc_dof = _mlMesh->GetLevel(ig)->GetElementDofNumber(iel, sol_type);
// 
//                 for(int j = 0; j < nloc_dof; j++) {
//                   unsigned inode_Metis = _mlMesh->GetLevel(ig)->GetSolutionDof(j, iel, sol_type);
//                   unsigned icoord_Metis = _mlMesh->GetLevel(ig)->GetSolutionDof(j, iel, 2);
//                   std::vector < double > xx(3);
//                   xx[0] = (*_mlMesh->GetLevel(ig)->_topology->_Sol[0])(icoord_Metis);
//                   xx[1] = (*_mlMesh->GetLevel(ig)->_topology->_Sol[1])(icoord_Metis);
//                   xx[2] = (*_mlMesh->GetLevel(ig)->_topology->_Sol[2])(icoord_Metis);
// 
//                   value = (func) ? func(xx) : funcMLProb(ml_prob, xx, name);
// 
//                   _solution[ig]->_Sol[i]->set(inode_Metis, value);
// 
//                   if(_solTimeOrder[i] == 2) {
//                     _solution[ig]->_SolOld[i]->set(inode_Metis, value);
//                   }
//                 }
//               }
//             }
          }
          else if(sol_type < NFE_FAMS) {
              
              const double offset_to_include_line = 1.e-8;
              
               const unsigned solType_coords = CONTINUOUS_BIQUADRATIC;

              const unsigned dim = _mlMesh->GetDimension();
              
  CurrentElem < double > geom_element_iel(dim, _mlMesh->GetLevel(ig) );
  
              for(int iel = _mlMesh->GetLevel(ig)->_elementOffset[_iproc];
                  iel < _mlMesh->GetLevel(ig)->_elementOffset[_iproc + 1]; iel++) {
                  value = 0.;

// ------- - BEGIN
    geom_element_iel.set_coords_at_dofs_and_geom_type(iel, solType_coords);
        
    const short unsigned ielGeom = geom_element_iel.geom_type();

    geom_element_iel.set_elem_center_3d(iel, solType_coords);
// ------- - END

              
	  for(unsigned iface = 0; iface < _mlMesh->GetLevel(ig)->GetElementFaceNumber(iel); iface++) {
       
          
        const int bdry_index = _mlMesh->GetLevel(ig)->el->GetFaceElementIndex(iel, iface);
        
        if (bdry_index < 0) {
        const unsigned int face_index_in_domain = - ( bdry_index + 1);
        
        //compute face element center - BEGIN
       geom_element_iel.set_coords_at_dofs_bdry_3d(iel, iface, solType_coords);
 
       geom_element_iel.set_elem_center_bdry_3d();

       const unsigned ielGeom_bdry = _mlMesh->GetLevel(ig)->GetElementFaceType(iel, iface);    
        //compute face element center - END
        

            for(unsigned f = 0; f <  LIST_OF_CTRL_FACES ::_face_with_extremes_index_size; f++) {
                
                  if (face_index_in_domain == /*ctrl::*/LIST_OF_CTRL_FACES :: _face_with_extremes_index[f]) {
                const unsigned number_of_tangential_direction_components = LIST_OF_CTRL_FACES ::   _num_of_tang_components_per_face ;
                                          
                unsigned cond_for_tang_component_all = 1;      

                            for(unsigned t = 0; t < number_of_tangential_direction_components; t++) {

                          const bool cond_for_tang_component_t = (geom_element_iel.get_elem_center_bdry_3d()[ /*ctrl::*/LIST_OF_CTRL_FACES ::tangential_direction_to_Gamma_control(face_index_in_domain, number_of_tangential_direction_components)[t] ] >
                                            /*ctrl::*/LIST_OF_CTRL_FACES ::_face_with_extremes_extremes_on_tang_surface[f][t][0] + offset_to_include_line &&
                                   geom_element_iel.get_elem_center_bdry_3d()[ /*ctrl::*/LIST_OF_CTRL_FACES ::tangential_direction_to_Gamma_control(face_index_in_domain, number_of_tangential_direction_components)[t] ] <
                                            /*ctrl::*/LIST_OF_CTRL_FACES ::_face_with_extremes_extremes_on_tang_surface[f][t][1] - offset_to_include_line
                                  );      
                                
                          if (  cond_for_tang_component_t == false )  { cond_for_tang_component_all *= 0; }

                            }
                            
                 if (cond_for_tang_component_all == true) {  value = 1.; }         
                            
            }


        }//end face_contol_index loop      
          
          
//                 value = (func) ? func(xx) : funcMLProb(ml_prob, xx, name);
                
                unsigned placeholder_index = 0/*2*/;

                unsigned solDof = _mlMesh->GetLevel(ig)->GetSolutionDof(placeholder_index, iel, sol_type);

                _solution[ig]->_Sol[i]->set(solDof, value);

                if(_solTimeOrder[i] == 2) {
                  _solution[ig]->_SolOld[i]->set(solDof, value);
                }
                
        }
              } //end iface
                
              } //end iel

          }

          _solution[ig]->_Sol[i]->close();

          if(_solTimeOrder[i] == 2) {
            _solution[ig]->_SolOld[i]->close();
          }
        }
        
      }
    }

    return;
  }



} //end namespace femus



#endif

