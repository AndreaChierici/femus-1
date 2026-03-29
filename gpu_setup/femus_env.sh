#bin/bash

unset SLEPC_DIR
unset SLEPC_PATH
unset PETSC_DIR
unset PETSC_PATH
module purge
module use --append $HOME/femus_deps/boost_module
export PETSC_DIR=$HOME/femus_deps/petsc
export SLEPC_DIR=$HOME/femus_deps/slepc

module load rocm/7.2.0
module load amdclang openmpi boost
echo " Modules currently loaded: "
module list
echo " Setting HSA_XNACK to 1 for unified shared memory "
export HSA_XNACK=1
ulimit -s unlimited
export FEMUS_USE_HIP=1

echo " To compile, go in $HOME/repos/femus-1/build and do: "
echo " Configuration Step: "
echo ' cmake  -DCMAKE_CXX_FLAGS="-fopenmp --offload-arch=gfx942" .. '
echo " Build step: "
echo " make -j "
echo " "
echo " "
echo " Run with mpirun -n 4 --map-by numa --bind-to numa ./NonLocal_ex10 3 2 0 1 "
