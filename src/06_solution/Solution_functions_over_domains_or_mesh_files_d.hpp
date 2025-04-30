#ifndef __femus_solution_functions_over_domains_or_mesh_files_d_hpp__
#define __femus_solution_functions_over_domains_or_mesh_files_d_hpp__


#include "Function_d.hpp"

using namespace femus;


namespace Domains {
    

namespace  square_m05p05  {

template <class type = double>
class Function_Zero_on_boundary_7 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return sin(pi*x[0]) * sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[1]);
    }

    type target(const std::vector<type>& x) const {
        return 0.;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = pi * sin(2.*pi*x[0]) * pow(sin(pi*x[1]), 2);
        solGrad[1] = pi * sin(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 2.*pi*pi * cos(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
     + 2.*pi*pi * cos(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_Laplacian : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 2.*pi*pi * cos(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
               + 2.*pi*pi * cos(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);
    }

    type target(const std::vector<type>& x) const  {
        return sin(pi*x[0]) * sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -4.*pi*pi*pi * sin(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
             + 2.*pi*pi*pi * cos(2.*pi*x[1]) * sin(2.*pi*x[0]);
        solGrad[1] = -4.*pi*pi*pi * sin(2.*pi*x[1]) * pow(sin(pi*x[0]), 2)
             + 2.*pi*pi*pi * cos(2.*pi*x[0]) * sin(2.*pi*x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const{
        return -16.*pi*pi*pi*pi * cos(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
           -16.*pi*pi*pi*pi * cos(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);
    }

private:
    static constexpr double pi = acos(-1.);
};



template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s1 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 2. * sin(pi*x[0]) * sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[1]);
    }

    type target(const std::vector<type>& x) const {
        return 0.;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2.*pi * sin(2.*pi*x[0]) * pow(sin(pi*x[1]), 2);
        solGrad[1] = 2.*pi * sin(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 4.*pi*pi * cos(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
         + 4.*pi*pi * cos(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s2 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return - 2.*pi*pi * cos(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
     + 2.*pi*pi * cos(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);
    }

    type target(const std::vector<type>& x) const {
        return 0.;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
       solGrad[0] = 4.*pi*pi*pi * sin(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
           + 2.*pi*pi*pi * cos(2.*pi*x[1]) * sin(2.*pi*x[0]);

       solGrad[1] = -4.*pi*pi*pi * cos(2.*pi*x[0]) * sin(2.*pi*x[1])
           + 2.*pi*pi*pi * sin(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);

        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 4.*pi*pi*pi*pi * cos(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
         + 4.*pi*pi*pi*pi * cos(2.*pi*x[1]) * pow(sin(pi*x[0]), 2);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_7_deviatoric_s3 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 8. * pi * pi * sin(pi *x[0]) * sin(pi *x[1]) - 2. - 2. * sin(pi*x[0]) * sin(pi*x[0]) * sin(pi*x[1]) * sin(pi*x[1]) ;
    }

        type target(const std::vector<type>& x) const {
        return 0.;
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 8.*pi*pi*pi * cos(pi * x[0]) * sin(pi * x[1])
               - 4.*pi * sin(pi * x[0]) * cos(pi * x[0]) * pow(sin(pi * x[1]), 2);
        solGrad[1] = 8.*pi*pi*pi * sin(pi * x[0]) * cos(pi * x[1])
               - 4.*pi * sin(pi * x[1]) * cos(pi * x[1]) * pow(sin(pi * x[0]), 2);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -16.*pi*pi*pi*pi * sin(pi*x[0]) * sin(pi*x[1])
           - 4.*pi*pi * (cos(2.*pi*x[0]) * pow(sin(pi*x[1]), 2)
                      + cos(2.*pi*x[1]) * pow(sin(pi*x[0]), 2));
    }

private:
    static constexpr double pi = acos(-1.);
};


}


} //end Domains

#endif
