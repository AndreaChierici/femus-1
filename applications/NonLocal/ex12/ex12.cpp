#pragma once

#include <iostream>
#include <omp.h>
#include<vector>

#pragma omp requires unified_shared_memory

class Shape {
  public:
    Shape() {};
    virtual ~Shape() {};
    #pragma omp begin declare target
    virtual double GetPerimeter(const std::vector<double>& sides) const = 0;
    #pragma omp end declare target
  protected:
    unsigned nSides;
};

class Square : public Shape {
  public:
    Square(): Shape() {
      nSides = 4;
    };
    ~Square() {};

    #pragma omp begin declare target
    double GetPerimeter(const std::vector<double>& sides) const {
      double p = 0;
      for(unsigned i = 0; i < nSides; i++) {
        p += sides[i];
      }
      return p;
    }
    #pragma omp end declare target
};


class Triangle : public Shape {
  public:
    Triangle(): Shape() {
      nSides = 3;
    };
    ~Triangle() {};

    #pragma omp begin declare target
    double GetPerimeter(const std::vector<double>& sides) const {
      double p = 0;
      for(unsigned i = 0; i < nSides; i++) {
        p += sides[i];
      }
      return p;
    }
    #pragma omp end declare target
};


//#define THREAD_NUM 8
int main(int argc, char* argv[])
{

  unsigned nobj = 1000;
  std::vector<std::vector<double>> sides(nobj);
  std::vector<Shape*> shape(nobj);


  Triangle* tri = new Triangle();

  for(unsigned i = 0; i < nobj; i++) {
    if(i % 2 == 0) {
      sides[i].assign(4, 1);
      shape[i] =  new Square();
    }
    else {
      sides[i].assign(3, 1);
      shape[i] =  new Triangle();
    }
  }

//#pragma omp target teams distribute parallel for //num_teams(192) thread_limit(192)
  for(unsigned i = 0; i < nobj; i++) {
    double p = shape[i]->GetPerimeter(sides[i]);
    std::cerr << p << " ";
  }
  std::cerr << std::endl;


  for(unsigned i = 0; i < nobj; i++) delete shape[i];

  return 0;
}
