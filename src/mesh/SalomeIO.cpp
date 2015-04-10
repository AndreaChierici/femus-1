/*=========================================================================

 Program: FEMUS
 Module: SalomeIO
 Authors: Sureka Pathmanathan, Giorgio Bornia
 
 Copyright (c) FEMTTU
 All rights reserved. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

//local include
#include "SalomeIO.hpp"
#include "Mesh.hpp"

//C++ include
#include <cassert>
#include <cstdio>
#include <fstream>

namespace femus {
  
   const std::string SalomeIO::group_name_begin = "FAS";
   const std::string SalomeIO::group_name_end   = "ELEME";
   const std::string SalomeIO::mesh_ensemble = "ENS_MAA";
   const std::string SalomeIO::aux_zeroone = "-0000000000000000001-0000000000000000001";
   const std::string SalomeIO::elem_list = "MAI";
   const std::string SalomeIO::connectivity = "NOD";
   const std::string SalomeIO::node_list = "NOE";
   const std::string SalomeIO::coord_list = "COO";
   const std::string SalomeIO::dofobj_indices = "NUM";
   const uint SalomeIO::max_length = 100;  ///@todo this length of the menu string is conservative enough...

  
  //How to determine a general connectivity:
  //you have to align the element with respect to the x-y-z (or xi-eta-zeta) reference frame,
  //and then look at the order in the med file.
  //For every node there is a location, and you have to put that index in that x-y-z location.
  //Look NOT at the NUMBERING, but at the ORDER!
  
// SALOME HEX27  
//         1------17-------5
//        /|              /|
//       / |             / |
//      8  |   21      12  |
//     /   9      22   /   13
//    /    |          /    |
//   0------16-------4     |
//   | 20  |   26    |  25 |
//   |     2------18-|-----6       zeta  
//   |    /          |    /          ^    
//  11   /  24       15  /           |   eta
//   | 10      23    |  14           |  /
//   | /             | /             | /
//   |/              |/              |/ 
//   3-------19------7               -------> xi  

// TEMPLATE HEX27  (for future uses)
//         X------X--------X
//        /|              /|
//       / |             / |
//      X  |   X        X  |
//     /   X      X    /   X
//    /    |          /    |
//   X------X--------X     |
//   | X   |   X     |  X  |
//   |     X------X--|-----X      zeta  
//   |    /          |    /         ^    
//   X   /  X        X   /          |   eta
//   |  X      X     |  X           |  /
//   | /             | /            | /
//   |/              |/             |/ 
//   X-------X-------X              -------> xi   

  
 const unsigned SalomeIO::SalomeToFemusVertexIndex[N_GEOM_ELS][27]= 
   {
     
    {4,7,3,0,5,6,2,1,15,19,11,16,13,18,9,17,12,14,10,8,23,25,22,24,20,21,26},  //HEX27
    {0,1,2,3,4,5,6,7,8,9},  //TET10
    
    {
      3, 11,5, 9, 10,4,
      12,17,14,15,16,13,
      0, 8, 2, 6, 7, 1
    },  //WEDGE18
    
    {0,1,2,3,4,5,6,7,8}, //QUAD9
    {0,1,2,3,4,5},       //TRI6
    {0,1,2}              //EDGE3
  };

  
const unsigned SalomeIO::SalomeToFemusFaceIndex[N_GEOM_ELS][6]= 
  {
    {0,4,2,5,3,1},
    {0,1,2,3},
    {2,1,0,4,3},
    {0,1,2,3},
    {0,1,2},
    {0,1}
  };

  /// @todo extend to Wegdes (aka Prisms)
void SalomeIO::read(const std::string& name, vector < vector < double> > &coords, const double Lref, std::vector<bool> &type_elem_flag) {
   
    Mesh& mesh = GetMesh();
    mesh.SetLevel(0);

    hsize_t dims[2];
    
    hid_t  file_id = H5Fopen(name.c_str(),H5F_ACC_RDWR, H5P_DEFAULT);
    
    hid_t  gid = H5Gopen(file_id,mesh_ensemble.c_str(),H5P_DEFAULT);
      
    hsize_t     n_menus;
    hid_t status= H5Gget_num_objs(gid, &n_menus);  // number of menus
    if(status !=0) { std::cout << "Number of mesh menus not found"; abort(); }
    
    std::vector<int>  itype_vol;
     itype_vol.resize(n_menus);  /// @todo will modify this
     
    std::vector<std::string>  mesh_menus;
    std::vector<std::string>  group_menus;
    
   // compute number of groups and number of meshes ===============
    unsigned n_groups = 0;
    unsigned n_meshes = 0;
   
    for (unsigned j=0; j<n_menus; j++) {
      
     char *  menu_names_j = new char[max_length];
     H5Gget_objname_by_idx(gid,j,menu_names_j,max_length); ///@deprecated see the HDF doc to replace this
     std::string tempj(menu_names_j);
     
  
     if  (tempj.substr(0,5).compare("Group") == 0) {
       n_groups++;
       group_menus.push_back(tempj);
     }
     else if  (tempj.substr(0,4).compare("Mesh") == 0) {
       n_meshes++;
       mesh_menus.push_back(tempj);
     }
     
     
    }
   // compute number of groups and number of meshes ===============
    
    for (unsigned j=0; j< n_meshes; j++) {
      
     std::string tempj = mesh_menus[j];

    std::string el_fem_type_vol_j(""); 
    std::string el_fem_type_bd_j("");

      
     itype_vol[j] =  ReadFE(file_id,el_fem_type_vol_j,el_fem_type_bd_j,tempj);
     
    //   // read control data ******************** A
//   mesh.SetDimension(dim);  // DONE  this is determined in the ReadFE routine
//   mesh.SetNumberOfElements(nel); //DONE below
//   mesh.SetNumberOfNodes(nvt);  //DONE below
    
    //   // end read control data **************** A

    //   // read NODAL COORDINATES **************** C
   std::string coord_dataset = mesh_ensemble +  "/" + tempj + "/" +  aux_zeroone + "/" + node_list + "/" + coord_list + "/";  ///@todo here we have to loop

  hid_t dtset = H5Dopen(file_id,coord_dataset.c_str(),H5P_DEFAULT);
  
  // SET NUMBER OF NODES
  hid_t filespace = H5Dget_space(dtset);    /* Get filespace handle first. */
  hid_t status  = H5Sget_simple_extent_dims(filespace, dims, NULL);
  if(status ==0) std::cerr << "SalomeIO::read dims not found";
  // reading xyz_med
  unsigned int n_nodes = dims[0]/mesh.GetDimension();
  double   *xyz_med = new double[dims[0]];
  std::cout << " Number of nodes in med file " <<  n_nodes << " " <<  std::endl;
  
  mesh.SetNumberOfNodes(n_nodes);
  
  // SET NODE COORDINATES  
    coords[0].resize(n_nodes);
    coords[1].resize(n_nodes);
    coords[2].resize(n_nodes);

  status=H5Dread(dtset,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,xyz_med);
  H5Dclose(dtset);
  
   if (mesh.GetDimension()==3) {
    for (unsigned j=0; j<n_nodes; j++) {
      coords[0][j] = xyz_med[j]/Lref;
      coords[1][j] = xyz_med[j+n_nodes]/Lref;
      coords[2][j] = xyz_med[j+2*n_nodes]/Lref;
    }
  }

  else if (mesh.GetDimension()==2) {
    for (unsigned j=0; j<n_nodes; j++) {
      coords[0][j] = xyz_med[j]/Lref;
      coords[1][j] = xyz_med[j+n_nodes]/Lref;
      coords[2][j] = 0.;
    }
  }

  else if (mesh.GetDimension()==1) {
    for (unsigned j=0; j<n_nodes; j++) {
      coords[0][j] = xyz_med[j]/Lref;
      coords[1][j] = 0.;
      coords[2][j] = 0.;
    }
  }
  
  delete[] xyz_med;
    
    //   // end read NODAL COORDINATES ************* C

    
    //   // read ELEMENT/cell ******************** B
  std::string my_mesh_name_dir = mesh_ensemble +  "/" + tempj + "/" +  aux_zeroone + "/" + elem_list + "/";  ///@todo here we have to loop
  
   // Getting  connectivity structure from file *.med
  std::string node_name_dir = my_mesh_name_dir + el_fem_type_vol_j + "/" + connectivity;
  dtset = H5Dopen(file_id,node_name_dir.c_str(),H5P_DEFAULT);
  filespace = H5Dget_space(dtset);    /* Get filespace handle first. */
  status  = H5Sget_simple_extent_dims(filespace, dims, NULL);
  if(status ==0) {std::cerr << "SalomeIO::read dims not found"; abort();}
  const int dim_conn = dims[0];

  // DETERMINE NUMBER OF NODES PER ELEMENT
  uint Node_el;
        if      ( el_fem_type_vol_j.compare("HE8") == 0 ) Node_el = 8;
	else if ( el_fem_type_vol_j.compare("H20") == 0 ) Node_el = 20;  
	else if ( el_fem_type_vol_j.compare("H27") == 0 ) Node_el = 27; 
	
	else if ( el_fem_type_vol_j.compare("TE4") == 0 ) Node_el = 4;     
	else if ( el_fem_type_vol_j.compare("T10") == 0 ) Node_el = 10;     
	
	else if ( el_fem_type_vol_j.compare("QU4") == 0 ) Node_el = 4;     
	else if ( el_fem_type_vol_j.compare("QU8") == 0 ) Node_el = 8;     
	else if ( el_fem_type_vol_j.compare("QU9") == 0 ) Node_el = 9;     
	
	else if ( el_fem_type_vol_j.compare("TR3") == 0 ) Node_el = 3;     
	else if ( el_fem_type_vol_j.compare("TR6") == 0 ) Node_el = 6;   

	else if ( el_fem_type_vol_j.compare("SE3") == 0 ) Node_el = 3;     
	else if ( el_fem_type_vol_j.compare("SE2") == 0 ) Node_el = 2;    
        else { std::cout << "SalomeIO::read: element not supported"; abort(); }
  
  unsigned int n_elements = dim_conn/Node_el;
  
  // SET NUMBER OF ELEMENTS
   mesh.SetNumberOfElements(n_elements);
  
  int   *conn_map = new  int[dim_conn];
  std::cout << " Number of elements in med file " <<  n_elements <<  std::endl;
  status=H5Dread(dtset,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,conn_map);
  if(status !=0) {std::cout << "SalomeIO::read: connectivity not found"; abort();}
  H5Dclose(dtset);

 mesh.el = new elem(n_elements);    ///@todo check where this is going to be deleted
 
  
  for (unsigned iel=0; iel<n_elements; iel++) {
    mesh.el->SetElementGroup(iel,1);
    unsigned nve = Node_el;  /// @todo this is only one element type
    if (nve==27) {
      type_elem_flag[0]=type_elem_flag[3]=true;
      mesh.el->AddToElementNumber(1,"Hex");
      mesh.el->SetElementType(iel,HEX);
    } else if (nve==10) {
      type_elem_flag[1]=type_elem_flag[4]=true;
      mesh.el->AddToElementNumber(1,"Tet");
      mesh.el->SetElementType(iel,TET);
    } else if (nve==18) {
      type_elem_flag[2]=type_elem_flag[3]=type_elem_flag[4]=true;
      mesh.el->AddToElementNumber(1,"Wedge");
      mesh.el->SetElementType(iel,WEDGE);
    } else if (nve==9) {
      type_elem_flag[3]=true;
      mesh.el->AddToElementNumber(1,"Quad");
      mesh.el->SetElementType(iel,QUAD);
    }
    else if (nve==6 && mesh.GetDimension()==2) {
      type_elem_flag[4]=true;
      mesh.el->AddToElementNumber(1,"Triangle");
      mesh.el->SetElementType(iel,TRI);
    }
    else if (nve==3 && mesh.GetDimension()==1) {
      mesh.el->AddToElementNumber(1,"Line");
      mesh.el->SetElementType(iel,LINE);
    } else {
      std::cout<<"Error! Invalid element type in reading  File!"<<std::endl;
      std::cout<<"Error! Use a second order discretization"<<std::endl;
      exit(0);
    }
    for (unsigned i=0; i<nve; i++) {
      unsigned inode = SalomeToFemusVertexIndex[mesh.el->GetElementType(iel)][i];
      mesh.el->SetElementVertexIndex(iel,inode,conn_map[iel+i*n_elements]);
    }
  }

    //   // end read  ELEMENT/CELL **************** B
    
  // clean
  delete [] conn_map;

        }   //end meshes
        
   mesh.el->SetElementGroupNumber(n_groups);
     //   // read GROUP **************** E
     
     for (unsigned j=0; j<n_groups; j++) {
       
     std::string tempj = group_menus[j];
   
       /// @todo check the underscores according to our naming standard
       
       // strip the first number to get the group number        //strip the second number to get the group material
       int gr_name = atoi(tempj.substr(6,1).c_str());  
       int gr_mat =  atoi(tempj.substr(8,1).c_str());

      std::string el_fem_type_vol_j(""); 
      std::string el_fem_type_bd_j("");

      
     ReadFE(file_id,el_fem_type_vol_j,el_fem_type_bd_j,tempj);
       
    std::string group_dataset = mesh_ensemble +  "/" + tempj + "/" +  aux_zeroone + "/" + elem_list + "/" + el_fem_type_vol_j + "/" + dofobj_indices;  ///@todo here we have to loop
    
  hid_t dtset = H5Dopen(file_id,group_dataset.c_str(),H5P_DEFAULT);
  hid_t filespace = H5Dget_space(dtset);    /* Get filespace handle first. */
  hid_t status  = H5Sget_simple_extent_dims(filespace, dims, NULL);
  int * elem_indices = new int[dims[0]];
  status=H5Dread(dtset,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,elem_indices);

  for (unsigned i=0; i < dims[0]; i++) {
           mesh.el->SetElementGroup(i, gr_name);
           mesh.el->SetElementMaterial(i,gr_mat);
    }
	
	
   H5Dclose(dtset);
   delete [] elem_indices;
   
    }
     //   // end read GROUP **************** E
 
    status = H5Fclose(file_id);
 

// 
//   unsigned nbcd;
 
//   // read boundary **************** D
//   inf.open(name.c_str());
//   if (!inf) {
//     std::cout<<"Generic-mesh file "<< name << " cannot read boudary\n";
//     exit(0);
//   }
//   for (unsigned k=0; k<nbcd; k++) {
//     while (str2.compare("CONDITIONS") != 0) inf >> str2;
//     inf >> str2;
//     int value;
//     unsigned nface;
//     inf >>value>>str2>>nface>>str2>>str2;
//     value=-value-1;
//     for (unsigned i=0; i<nface; i++) {
//       unsigned iel,iface;
//       inf>>iel>>str2>>iface;
//       iel--;
//       iface=SalomeIO::SalomeToFemusFaceIndex[mesh.el->GetElementType(iel)][iface-1u];
//       mesh.el->SetFaceElementIndex(iel,iface,value);
//     }
//     inf >> str2;
//     if (str2.compare("ENDOFSECTION") != 0) {
//       std::cout<<"error boundary data mesh"<<std::endl;
//       exit(0);
//     }
//   }
//   inf.close();
//   // end read boundary **************** D
 
}




int  SalomeIO::ReadFE(
  hid_t file_id,
  std::string & el_fem_type_vol,
  std::string & el_fem_type_bd,
  const  std::string menu_name
) {
  
  Mesh& mesh = GetMesh();
    /// @todo this determination of the dimension from the mesh file would not work with a 2D mesh embedded in 3D
  std::string my_mesh_name_dir = mesh_ensemble +  "/" + menu_name + "/" +  aux_zeroone + "/" + elem_list + "/";  ///@todo here we have to loop
  
  hid_t       gid = H5Gopen(file_id,my_mesh_name_dir.c_str(),H5P_DEFAULT); // group identity
  hsize_t     n_fem_type;
  hid_t status= H5Gget_num_objs(gid, &n_fem_type);  // number of links
  if(status !=0) {std::cout << "SalomeIO::read_fem_type:   H5Gget_num_objs not found"; abort();}

  
  FindDimension(gid,menu_name,n_fem_type);
  
  
  // ---------  Reading Element type
  std::map< std::string, int > fem_type_vol;// Salome fem table name (vol)
  fem_type_vol["HE8"] = 5;  fem_type_vol["H20"] = 20; fem_type_vol["H27"] = 12;
  fem_type_vol["TE4"] = 4;  fem_type_vol["T10"] = 11;
  fem_type_vol["QU4"] = 3;  fem_type_vol["QU8"] = 19; fem_type_vol["QU9"] = 10;
  fem_type_vol["TR3"] = 2;  fem_type_vol["TR6"] = 9;
  fem_type_vol["SE2"] = 0;  fem_type_vol["SE3"] = 0;  // not valid in 3D
  if(mesh.GetDimension()==3) {    // not valid in 3D as boundary
    fem_type_vol["QU4"] = 0; fem_type_vol["QU8"] = 0; fem_type_vol["QU9"] = 0;
    fem_type_vol["TR3"] = 0; fem_type_vol["TR6"] = 0;
  }
  std::map< std::string, int > fem_type_bd; // Salome fem table name (surface)
  fem_type_bd["HE8"] = 0;  fem_type_bd["H20"] = 0;  fem_type_bd["H27"] = 0;
  fem_type_bd["TE4"] = 0;  fem_type_bd["T10"] = 0;  // not valid in 2D
  fem_type_bd["QU4"] = 3;  fem_type_bd["QU8"] = 19;  fem_type_bd["QU9"] = 10;
  fem_type_bd["TR3"] = 2;  fem_type_bd["TR6"] = 9;
  fem_type_bd["SE2"] = 1;  fem_type_bd["SE3"] = 8;
  if(mesh.GetDimension()==3) {    // not valid in 3D as boundary
    fem_type_bd["SE2"] = 0;  fem_type_bd["SE3"] = 0;
  }


  // Get the element name from MESH_NAME_DIR in the file med (el_name)
  char **el_fem_type = new char*[n_fem_type];
  int index_vol=0;  int index_bd=0;
  for(int i=0; i<(int)n_fem_type; i++) {
    el_fem_type[i]=new char[4];
    H5Lget_name_by_idx(file_id,my_mesh_name_dir.c_str(), H5_INDEX_NAME, H5_ITER_INC,i,el_fem_type[i],4, H5P_DEFAULT);
    if(fem_type_vol[el_fem_type[i]]!=0) index_vol=i;
    if(fem_type_bd[el_fem_type[i]]!=0) index_bd=i;
  }

  // LIBMESH volume fem type (itype_vol) and MEDfem (el_fem_type_vol-el_fem_type_bd)
  int itype_vol= fem_type_vol[el_fem_type[index_vol]]; assert(itype_vol!=0);
  el_fem_type_vol=el_fem_type[index_vol];
  el_fem_type_bd=el_fem_type[index_bd];
  // clean
  for(int i=0; i<(int)n_fem_type; i++) delete[] el_fem_type[i];
  delete[] el_fem_type; 
  fem_type_vol.clear();
  fem_type_bd.clear();
  
  return itype_vol;
}


void  SalomeIO::FindDimension(hid_t gid, const  std::string menu_name, hsize_t n_fem_type) {
  
  
    Mesh& mesh = GetMesh();
    uint mydim = 1;  //this is the initial value, then it will be updated below
    mesh.SetDimension(mydim);
    

  std::vector<char*> elem_list;  elem_list.resize(n_fem_type);
    for (unsigned j=0; j < elem_list.size(); j++) {
      elem_list[j] = new char[max_length];
      H5Gget_objname_by_idx(gid,j,elem_list[j],max_length); ///@deprecated see the HDF doc to replace this
      std::string tempj(elem_list[j]);
      
      if ( tempj.compare("HE8") == 0 || 
	   tempj.compare("H20") == 0 ||  
	   tempj.compare("H27") == 0 || 
	   tempj.compare("TE4") == 0 || 
	   tempj.compare("T10") == 0    )      { mydim = 3; } 
      else if ( tempj.compare("QU4") == 0 || 
	        tempj.compare("QU8") == 0 ||  
	        tempj.compare("QU9") == 0 || 
	        tempj.compare("TR3") == 0 || 
	        tempj.compare("TR6") == 0    ) { mydim = 2; } 
      
      if ( mydim > mesh.GetDimension() ) mesh.SetDimension(mydim); 
	
    }  //end for
    
  
  return;
}






} //end namespace femus
