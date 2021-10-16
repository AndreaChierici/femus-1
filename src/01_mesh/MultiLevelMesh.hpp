/*=========================================================================

Program: FEMUS
Module: MultiLevelMesh
Authors: Eugenio Aulisa, Simone Bnà, Giorgio Bornia

Copyright (c) FEMUS
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __femus_mesh_MultiLevelMesh_hpp__
#define __femus_mesh_MultiLevelMesh_hpp__


#include <vector>
#include "ElemTypeEnum.hpp"
#include "GeomElTypeEnum.hpp"
#include "WriterEnum.hpp"
#include "Writer.hpp"
#include <vector>
namespace femus {


//------------------------------------------------------------------------------
// Forward declarations
//------------------------------------------------------------------------------
class elem_type;
class Mesh;
class Domain;

/**
* This class is a black box container to handle multilevel mesh.
*/

class MultiLevelMesh {

public:

    /** Constructor */
    MultiLevelMesh();

    MultiLevelMesh(const unsigned short &igridn,const unsigned short &igridr,
                   const char mesh_file[], const char GaussOrder[], const double Lref,
                   bool (* SetRefinementFlag)(const std::vector < double > &x,
                           const int &ElemGroupNumber,const int &level));
    
    /** Destructor */
    ~MultiLevelMesh();

    /** Read the coarse-mesh from an input file (call the right reader from the extension) */
    void ReadCoarseMesh(const char mesh_file[], const char GaussOrder[], const double Lref);

    /** Read the coarse-mesh from an input file (call the right reader from the extension) */
    void ReadCoarseMesh(const char mesh_file[], const char GaussOrder[], const double Lref, const bool read_groups, const bool read_boundary_groups);
    
    /** For every Geometric Element type appearing in the mesh, initialize evaluations at quadrature points, for all FE families  */
    void BuildFETypesBasedOnExistingCoarseMeshGeomElements(const char GaussOrder[]);
    
    void PrepareAllLevelsForRefinement();
    
    void ReadCoarseMeshOnlyFileReadingBeforePartitioning(const char mesh_file[], const double Lref, const bool read_groups, const bool read_boundary_groups);

    void ReadCoarseMeshOnlyFileReading(const char mesh_file[], const double Lref, const bool read_groups, const bool read_boundary_groups);
    
    /** Built-in cube-structured mesh generator */
    void GenerateCoarseBoxMesh( const unsigned int nx,
                               const unsigned int ny,
                               const unsigned int nz,
                               const double xmin, const double xmax,
                               const double ymin, const double ymax,
                               const double zmin, const double zmax,
                               const ElemType type,
                               const char GaussOrder[]
                             );

    /** Refine the coarse mesh (totally or selectively (according to the SetRefinementFlag user-function) ):
       the first argument is the number of uniformly refined levels;
       the second argument is the total number of refined levels: uniform + selective */
    void RefineMesh(const unsigned short &igridn, 
                    const unsigned short &igridr,
                    bool (* SetRefinementFlag)(const std::vector < double >& x, const int &ElemGroupNumber,const int &level) );

    /** Add a partially refined mesh level in the AMR alghorithm **/
    void AddAMRMeshLevel();
    
    
    /** Get the mesh pointer to level i */
    Mesh* GetLevelZero(const unsigned i) {
        return _level0[i];
    };

    /** Get the mesh pointer to level i */
    Mesh* GetLevel(const unsigned i) {
        return _level[i];
    };

    /** Get the mesh pointer to level i */
    const Mesh* GetLevel(const unsigned i) const {
        return _level[i];
    };

    /** Get the number of grid */
    unsigned GetNumberOfLevels() {
        return _gridn;
    };

    /** Get the number of grid */
    const unsigned GetNumberOfLevels() const {
        return _gridn;
    };

    /** Erase levels_to_be_erased levels from the mesh array */
    void EraseCoarseLevels(unsigned levels_to_be_erased);
   
    /** Print the mesh info for each level */
    void PrintInfo();

    // data
    const elem_type *_finiteElement[6][5];
    
    /** To be Added */
    Writer* GetWriter() const {return _writer; }

    /** To be Added */
    void SetWriter(const WriterEnum format) { _writer = Writer::build(format,this).release(); }
    
    /** Get the dimension of the problem (1D, 2D, 3D) */
    const unsigned GetDimension() const;

    /** Domain (optional) */
    Domain* GetDomain() const;
    
    /** Domain (optional) */
    void    SetDomain(Domain* );    
    
protected:

private:
    
    /**  */
    unsigned short _gridn0;
    unsigned short _gridn;

    /** Array of meshes,  dimension: number of levels */
    std::vector <Mesh*> _level0;
    std::vector <Mesh*> _level;

    /** Flag to denote what Geometric Elements are in the given Mesh */
    std::vector <bool> _finiteElementGeometryFlag;
    
    /** MultilevelMesh  writer */
    Writer* _writer;
    
    /** Domain (optional) */
    Domain* _domain;
};


} //end namespace femus



#endif
