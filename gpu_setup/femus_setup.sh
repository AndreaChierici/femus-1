#bin/bash

cd $HOME
mkdir femus_deps
cd femus_deps
pwd
mkdir petsc
mkdir slepc
mkdir eigen3
mkdir boost
mkdir boost_module

export PETSC_PATH=$PWD/petsc
export SLEPC_PATH=$PWD/slepc
export EIGEN3_PATH=$PWD/eigen3
export BOOST_PATH=$PWD/boost
export BOOST_MOD=$PWD/boost_module

module load rocm/7.2.0 openmpi

cd $HOME
mkdir repos
cd repos

git clone -b v3.24.1  https://gitlab.com/petsc/petsc.git
cd petsc
export PETSC_SOURCE=$PWD

# Patch ScaLAPACK.py: override CDEFS to fix broken Fortran mangling
# detection with AMD flang (LLVMFlang) in ScaLAPACK's CMake
python3 -c "
import os
f = os.path.join('config','BuildSystem','config','packages','ScaLAPACK.py')
txt = open(f).read()
old = '''  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DLAPACK_LIBRARIES=\"'+self.libraries.toString(self.blasLapack.dlib)+'\"')
    args.append('-DSCALAPACK_BUILD_TESTS=OFF')
    return args'''
new = '''  def formCMakeConfigureArgs(self):
    args = config.package.CMakePackage.formCMakeConfigureArgs(self)
    args.append('-DLAPACK_LIBRARIES=\"'+self.libraries.toString(self.blasLapack.dlib)+'\"')
    args.append('-DSCALAPACK_BUILD_TESTS=OFF')
    if self.compilers.fortranManglingDoubleUnderscore:
      args.append('-DCDEFS=Add__')
    elif self.compilers.fortranMangling == \"underscore\":
      args.append('-DCDEFS=Add_')
    elif self.compilers.fortranMangling == \"caps\":
      args.append('-DCDEFS=UPPER')
    elif self.compilers.fortranMangling == \"unchanged\":
      args.append('-DCDEFS=NOCHANGE')
    return args'''
assert old in txt, 'ScaLAPACK.py patch target not found; the file may have changed'
open(f,'w').write(txt.replace(old, new))
print('ScaLAPACK.py patched successfully')
"

./configure --with-debugging=0 --with-x=0 COPTFLAGS="-O3 -march=native -mtune=native" CXXOPTFLAGS="-O3 -march=native -mtune=native" FOPTFLAGS="-O3 -march=native -mtune=native" HIPOPTFLAGS="-O3 -march=native -mtune=native" --download-fblaslapack=1 --download-hdf5=1 --download-metis=1 --download-parmetis=1 --with-shared-libraries=1 --download-blacs=1 --download-scalapack=1 --download-mumps=1 --download-suitesparse=1 --with-hip-arch=gfx942 --with-mpi=1 --with-mpi-dir=$MPI_PATH --prefix=$PETSC_PATH --with-hip=1 --with-hip-dir=$ROCM_PATH

make PETSC_DIR=$PETSC_SOURCE PETSC_ARCH=arch-linux-c-opt all

make PETSC_DIR=$PETSC_SOURCE PETSC_ARCH=arch-linux-c-opt install

cd ..
git clone -b v3.24.1 https://gitlab.com/slepc/slepc.git
cd slepc
export SLEPC_SOURCE=$PWD
export PETSC_DIR=$PETSC_PATH
./configure --prefix=$SLEPC_PATH --with-clean
make SLEPC_DIR=$SLEPC_SOURCE PETSC_DIR=$PETSC_PATH
make SLEPC_DIR=$SLEPC_SOURCE PETSC_DIR=$PETSC_PATH install

cd ..
git clone --branch nightly https://gitlab.com/libeigen/eigen.git
cd eigen
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$EIGEN3_PATH -DCHOLMOD_LIBRARIES=$PETSC_PATH/lib -DCHOLMOD_INCLUDES=$PETSC_PATH/include -DKLU_LIBRARIES=$PETSC_PATH/lib -DKLU_INCLUDES=$PETSC_PATH/include ..
make install

cd ../..
git clone https://github.com/amd/HPCTrainingDock.git
cd HPCTrainingDock
./extras/scripts/boost_setup.sh --module-path $BOOST_MOD --install-path $BOOST_PATH --build-boost 1
module use --append $BOOST_MOD
mkdir -p $BOOST_MOD/boost
mv $BOOST_MOD/1.82.0.lua $BOOST_MOD/boost/1.82.0.lua
module load boost

cd ..
#git clone --branch aac6_amd https://github.com/AndreaChierici/femus-1.git 
#cd femus-1
#mkdir build && cd build
#export SLEPC_DIR=$SLEPC_PATH
#module load amdclang

#cmake  -DCMAKE_CXX_FLAGS="-fopenmp --offload-arch=gfx942 "  ..
#make -j
