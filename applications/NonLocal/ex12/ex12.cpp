#include "FemusInit.hpp"
#include "MultiLevelSolution.hpp"
#include "MultiLevelProblem.hpp"
#include "CurrentElem.hpp"
#include "LinearImplicitSystem.hpp"

#include "PolynomialBases.hpp"

#include "CutFemWeight.hpp"

#include "CDWeights.hpp"

#include <vector>
#include <cmath>
#include <iostream>

#include "Fem.hpp"

using namespace std;
using namespace femus;

#include <iostream>
#include <omp.h>
#include <unistd.h>
#define THREAD_NUM 8
int main(int argc, char* argv[])
{
    FemusInit mpinit(argc, argv, MPI_COMM_WORLD);

    #pragma omp parallel num_threads(THREAD_NUM)
     //omp_set_thread_num(THREAD_NUM); // set number of threads in "parallel" blocks
    // #pragma omp parallel
    {
 //       usleep(5000 * omp_get_thread_num()); // do this to avoid race condition while printing
      //  std::cerr << "Number of available threads: " << omp_get_num_threads() << std::endl;
        // each thread can also get its own number
//	std::cerr << "Current thread number: " << omp_get_thread_num() << std::endl;
        std::cerr << "Hello, World!" << std::endl;
    }
    return 0;
}
