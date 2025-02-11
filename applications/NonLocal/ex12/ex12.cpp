#pragma omp requires unified_shared_memory

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
#include "BestFitPlane.hpp"
#include "GenerateTriangles.hpp"

using namespace std;
using namespace femus;

// HIP header
#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>

//OpenMP header
#include <omp.h>

#define NUM_THREADS 16
#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

__global__ void
hip_helloworld(unsigned omp_id, int* A_d)
{
    // Note: the printf command will only work if printf is enabled in your build.
    printf("Hello World... from HIP thread = %u\n", omp_id);

    A_d[omp_id] = omp_id;
}

int main(int argc, char* argv[])
{
FemusInit mpinit(argc, argv, MPI_COMM_WORLD);
	
    int* A_h, * A_d;
    size_t Nbytes = NUM_THREADS * sizeof(int);

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf("info: running on device %s\n", props.name);

    A_h = (int*)malloc(Nbytes);
    CHECK(hipMalloc(&A_d, Nbytes));
    for (int i = 0; i < NUM_THREADS; i++) {
        A_h[i] = 0;
    }
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    // Beginning of parallel region
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        fprintf(stderr, "Hello World... from OMP thread = %d\n",
               omp_get_thread_num());

//        hipLaunchKernelGGL(hip_helloworld, dim3(1), dim3(1), 0, 0, omp_get_thread_num(), A_d);
    }
    // Ending of parallel region

    hipStreamSynchronize(0);
    CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
    printf("Device Results:\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        printf("  A_d[%d] = %d\n", i, A_h[i]);
    }

    printf ("PASSED!\n");

    free(A_h);
    CHECK(hipFree(A_d));
    return 0;
}



