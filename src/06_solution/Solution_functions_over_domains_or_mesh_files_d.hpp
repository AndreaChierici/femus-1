#ifndef __femus_solution_functions_over_domains_or_mesh_files_d_hpp__
#define __femus_solution_functions_over_domains_or_mesh_files_d_hpp__


#include "Function_d.hpp"

using namespace femus;


namespace Domains {
    

namespace  square_m05p05  {

template <class type = double>
class Function_Zero_on_boundary_8 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = 2. * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = 2. * pi * sin(2. * pi * x[0]) * cos(2.* pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -8. * pi * pi * sin(2.* pi * x[0]) * sin(2.*pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_8_Laplacian : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -8.*pi*pi * sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -16. * pi * pi * pi * cos(2. * pi*x[0]) * sin(2. * pi*x[1]);
        solGrad[1] = -16. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2.* pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 64. * pi * pi * pi * pi * sin(2. * pi*x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};


template <class type = double>
class Function_Zero_on_boundary_8_deviatoric_s1 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -4. * pi * pi * sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * cos(2.* pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * sin(2.* pi * x[0]) * cos(2. * pi * x[1]);
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 32. * pi * pi * pi * pi * sin(2.* pi * x[0]) * sin(2. * pi * x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_8_deviatoric_s2 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return 4. * pi * pi * cos(2. * pi * x[0]) * cos(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * sin(2. * pi * x[0]) * cos(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * cos(2. * pi * x[0]) * sin( 2. * pi*x[1] );
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return -16. * pi * pi * pi * pi * cos(2.*pi*x[0]) * cos(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};

template <class type = double>
class Function_Zero_on_boundary_8_deviatoric_s3 : public Math::Function<type> {

public:
    type value(const std::vector<type>& x) const {
        return -4. * pi * pi * sin(2. * pi * x[0]) * sin(2. * pi * x[1]);
    }

    std::vector<type> gradient(const std::vector<type>& x) const {
        std::vector<type> solGrad(x.size(), 0.);
        solGrad[0] = -8. * pi * pi * pi * cos(2. * pi * x[0]) * sin(2. * pi * x[1]);
        solGrad[1] = -8. * pi * pi * pi * sin(2. * pi * x[0]) * cos( 2. * pi*x[1] );
        return solGrad;
    }

    type laplacian(const std::vector<type>& x) const {
        return 32. * pi * pi * pi * pi * sin(2.*pi*x[0]) * sin(2.*pi*x[1]);
    }

private:
    static constexpr double pi = acos(-1.);
};





}


} //end Domains

#endif
