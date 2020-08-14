/*=========================================================================

 Program: FEMUS
 Module: FemusInit
 Authors: Eugenio Aulisa, Giorgio Bornia
 
 Copyright (c) FEMTTU
 All rights reserved. 

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __femus_quadrature_WedgeGaussPoints_hpp__
#define __femus_quadrature_WedgeGaussPoints_hpp__

#include <vector>
#include <string>

namespace femus {

  
  class wedge_gauss {
  public:
    static const unsigned GaussPoints[7];
    static const double *Gauss[7];  
    static const double Gauss0[4][1];
    static const double Gauss1[4][8];
    static const double Gauss2[4][21];
    static const double Gauss3[4][52];
    static const double Gauss4[4][95];
    static const double Gauss5[4][168];
    static const double Gauss6[4][259];
  };  
     

     
} //end namespace femus     


#endif