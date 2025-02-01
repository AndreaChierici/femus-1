MyFEMuS
======

Welcome to the MyFEMuS project! MyFEMuS is a fork of the FEMuS project administered mainly by Eugenio Aulisa.
The manual intallation, will also work with the FEMuS project.

For the FEMuS project automatic installation, as well as the Mac-specific installation, see below.


<!-- ![alt tag](https://github.com/FeMTTU/femus/blob/master/doc/images/logo.jpg?raw=true) -->
<!-- ![alt tag](https://github.com/FeMTTU/femus/blob/master/doc/images/FSI.jpg?raw=true) -->

Step by step MyFEMuS manual setup  (largely tested on OpenSuse)
======

Install PETSC

From the directory $INSTALLATION_DIR clone petsc

    git clone -b release https://gitlab.com/petsc/petsc.git
    
    cd petsc
    
Configure, compile, test PETSC with the following options
    
    ./configure --with-debugging=0 --with-x=1 COPTFLAGS="-O3 -march=native -mtune=native" CXXOPTFLAGS="-O3 -march=native -mtune=native" FOPTFLAGS="-O3 -march=native -mtune=native" --download-openmpi=1 --download-fblaslapack=1 --download-hdf5=1 --download-metis=1 --download-parmetis=1 --with-shared-libraries=1 --download-blacs=1 --download-scalapack=1 --download-mumps=1 --download-suitesparse

    make PETSC_DIR=$INSTALLATION_DIR/petsc PETSC_ARCH=arch-linux2-c-opt all

    make PETSC_DIR=$INSTALLATION_DIR/petsc PETSC_ARCH=arch-linux2-c-opt test
 
======

Install SLEPC

From the directory $INSTALLATION_DIR clone slepc

    git clone -b release https://gitlab.com/slepc/slepc.git

    cd slepc
    
Configure, compile, test SLEPC with the following options
    
    export PETSC_DIR=$INSTALLATION_DIR/petsc 
    
    export PETSC_ARCH=arch-linux2-c-opt
    
    ./configure
    
    make SLEPC_DIR=$PWD all
    
    make SLEPC_DIR=$PWD test

======

Install MyFEMuS 

Be sure you have installed al least gcc 7, cmake, cmake-gui. Fparser may be handy for some applications but it is not required. 

Clone the MyFEMuS source code from the github repository

From the directory $INSTALLATION_DIR clone MyFEMuS

    https://github.com/eaulisa/MyFEMuS.git
    
    cd MyFEMuS
    
I generally export the following variables in the ./bashrc file in my user home, so that are available everywhere, otherwise you will need to export them all the times. 
    
    export PETSC_DIR=$INSTALLATION_DIR/petsc 
    
    export PETSC_ARCH=arch-linux2-c-opt
    
    export SLEPC_DIR=$INSTALLATION_DIR/slepc 
    
Configure MyFEMuS using cmake-gui. 

    cmake-gui 

    Where is the source code: $INSTALLATION_DIR/MyFEMuS
    
    Where to build the binaries: $INSTALLATION_DIR/feumsbin
    
    CMAKE_BUILD_TYPE choose between release (default) or debug
    
    Press Configure button
    
    Press Generate button

Compile
    
    cd $INSTALLATION_DIR/femusbin
    
    make
    
Run. All applications are built in the folder $INSTALLATION_DIR/femusbin/applications/..
    
======
    
# MyFEMUS INSTALL WITH DEPENDENCIES ON ODYSSEY (UO), for the for the amd_tools hackathoon
### For help contact Giacomo Capodaglio, Eugenio Aulisa, or Andrea Chierici

ssh on the Odyssey server and load enviromental variables

    bash
    source /storage/packages/Modules/amd-hpc-training-modulefiles/setup-env.sh
    module load openmpi

Create and export the FEMUS_INSTALL and the REPO folders. <br>
All libraries will be downloaded and configured into the REPO (in my case ~/repos) folder <br>
All libraries will be installed into the FEMUS_INSTALL (~/install/femus) folder

    export PETSC_PATH=$FEMUS_INSTALL/petsc
    export SLEPC_PATH=$FEMUS_INSTALL/slepc
    export EIGEN3_PATH=$FEMUS_INSTALL/eigen3
    export UCX_WARN_UNUSED_ENV_VARS=n
    export BOOST_ROOT=/storage/packages/e4s/24.11/mvapich-4.0-rocm6.3.0/spack/opt/spack/linux-rhel8-x86_64_v3/gcc-11.2.0/boost-1.79.0-t6mg37revd5l3fbtewvyie3wndqxxadk

### PETSC INSTALL
From the REPO folder

    git clone -b release https://gitlab.com/petsc/petsc.git
    cd petsc

    ./configure --with-debugging=0 --with-x=0 COPTFLAGS="-O3 -march=native -mtune=native" CXXOPTFLAGS="-O3 -march=native -mtune=native" FOPTFLAGS="-O3 -march=native -mtune=native" HIPOPTFLAGS="-O3 -march=native -mtune=native" --download-fblaslapack=1 --download-hdf5=1 --download-metis=1 --download-parmetis=1 --with-shared-libraries=1 --download-blacs=1 --download-scalapack=1 --download-mumps=1 --download-suitesparse=1 --with-hip-arch=gfx942 --with-mpi=1 --with-mpi-dir=$OPENMPI_ROOT --prefix=$PETSC_PATH --with-hip=1 --with-hip-dir=$ROCM_PATH

The next three commands are given by PETSc after configuring. Copy and paste them from the terminal

    make PETSC_DIR=$REPO/petsc PETSC_ARCH=arch-linux-c-opt all
    make PETSC_DIR=$REPO/petsc PETSC_ARCH=arch-linux-c-opt install

    make PETSC_DIR=$PETSC_PATH PETSC_ARCH="" check

Expected output:

Running PETSc check examples to verify correct installation<br>
Using PETSC_DIR=$PETSC_PATH and PETSC_ARCH=<br>
C/C++ example src/snes/tutorials/ex19 run successfully with 1 MPI process<br>
C/C++ example src/snes/tutorials/ex19 run successfully with 2 MPI processes<br>
C/C++ example src/snes/tutorials/ex19 run successfully with HIP<br>
C/C++ example src/snes/tutorials/ex19 run successfully with MUMPS<br>
C/C++ example src/snes/tutorials/ex19 run successfully with SuiteSparse<br>
C/C++ example src/vec/vec/tests/ex47 run successfully with HDF5<br>
Fortran example src/snes/tutorials/ex5f run successfully with 1 MPI process<br>
Completed PETSc check examples

    export PETSC_DIR=$PETSC_PATH


### SLEPC INSTALL
From the REPO folder

    git clone -b release https://gitlab.com/slepc/slepc.git
    cd slepc

    ./configure --prefix=$SLEPC_PATH

The next three commands are given by SLEPc after configuring. Copy and paste them from the terminal

    make SLEPC_DIR=$REPO/slepc PETSC_DIR=$PETSC_PATH
    make SLEPC_DIR=$REPO/slepc PETSC_DIR=$PETSC_PATH install

    make SLEPC_DIR=$SLEPC_PATH PETSC_DIR=$PETSC_PATH PETSC_ARCH="" check

Expected output:

Running SLEPc check examples to verify correct installation <br>
Using SLEPC_DIR=$SLEPC_PATH, PETSC_DIR=$PETSC_PATH, and PETSC_ARCH= <br>
C/C++ example src/eps/tests/test10 run successfully with 1 MPI process <br>
C/C++ example src/eps/tests/test10 run successfully with 2 MPI processes <br>
Fortran example src/eps/tests/test7f run successfully with 1 MPI process <br>
C/C++ example src/eps/tests/test10 run successfully with HIP <br>
Completed SLEPc check examples <br>

    export SLEPC_DIR=$SLEPC_PATH

### EIGEN3 INSTALL
From the REPO folder

    git clone -b 3.4.0 https://gitlab.com/libeigen/eigen.git
    cd eigen
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=$EIGEN3_PATH -DCHOLMOD_LIBRARIES=$PETSC_PATH/lib -DCHOLMOD_INCLUDES=$PETSC_PATH/include -DKLU_LIBRARIES=$PETSC_PATH/lib -DKLU_INCLUDES=$PETSC_PATH/include ../
    make install

### FEMUS INSTALL
From the REPO folder

    git clone -b amd_tools https://github.com/eaulisa/MyFEMuS.git

    cd MyFemus
<!-- ### Add to CMakeLists.txt
     INCLUDE_DIRECTORIES($ENV{BOOST_ROOT}/include)
     ### Add to cmake-modules/FindPETSc.cmake at line 326
     set(PETSC_EXECUTABLE_RUNS YES) -->


### Build with the gcc compiler
From the FEMUS_INSTALL folder

    mkdir femus_gcc
    cd femus_gcc

    ccmake -B ./ -S $REPO/MyFEMuS/ -DPETSC_EXECUTABLE_RUNS=yes

[c] configure as many times as needed it for [g] to appear
[g] generate

    make -j 12


### Build with the amdclang compiler
From the FEMUS_INSTALL folder

    mkdir femus_amd
    cd femus_amd

    module load amdclang

    ccmake -B ./ -S $REPO/MyFEMuS/ -DPETSC_EXECUTABLE_RUNS=yes

[c] configure as many times as needed it for [g] to appear
[g] generate

    make -j 12


### Build with the hip compiler
From the FEMUS_INSTALL folder

    mkdir femus_hip
    cd femus_hip

    export CXX=$ROCM_PATH/bin/hipcc

    ccmake -B ./ -S $REPO/MyFEMuS/ -DPETSC_EXECUTABLE_RUNS=yes

[c] configure as many times as needed it for [g] to appear
[g] generate

    make -j 12


### Set up the enviroment on a new ssh

ssh on the Odyssey server

    bash
    source /storage/packages/Modules/amd-hpc-training-modulefiles/setup-env.sh
    module load openmpi

    export FEMUS_INSTALL=~/install/femus/
    export REPO=~/repos

    export PETSC_DIR=$FEMUS_INSTALL/petsc
    export SLEPC_DIR=$FEMUS_INSTALL/slepc

    export UCX_WARN_UNUSED_ENV_VARS=n
    export BOOST_ROOT=/storage/packages/e4s/24.11/mvapich-4.0-rocm6.3.0/spack/opt/spack/linux-rhel8-x86_64_v3/gcc-11.2.0/boost-1.79.0-t6mg37revd5l3fbtewvyie3wndqxxadk

======

FEMuS automatic configuration, contact Giorgio Bornia for support.
======

Welcome to the FEMuS project! FEMuS is an open-source Finite Element C++ library 
built on top of PETSc, which allows scientists to build and solve multiphysics 
problems with multigrid and domain decomposition techniques.


<!-- ![alt tag](https://github.com/FeMTTU/femus/blob/master/doc/images/logo.jpg?raw=true) -->
<!-- ![alt tag](https://github.com/FeMTTU/femus/blob/master/doc/images/FSI.jpg?raw=true) -->

Setup
=====


Clone the FEMuS source code from the github repository:


    git clone https://github.com/FeMTTU/femus.git

   
You need PETSc for FEMuS to work.
If PETSc is not already installed in your machine, the script "install_petsc.sh" in contrib/scripts/ will install it automatically,
with the following syntax:
  
    ./femus/contrib/scripts/install_petsc.sh --prefix-external my_dir 
  

where "my_dir" is the directory, either absolute or relative, in which you want PETSc to be installed 
(please put it outside of the femus repo directory, to prevent from potential git tracking).

 Source the "configure_femus.sh" script and execute the function "fm_set_femus" in order to set some environment variables:

    source femus/contrib/scripts/configure_femus.sh

    fm_set_femus  --prefix-external my_dir --method-petsc opt
   

  Create the build directory, cd to it and run cmake:
   
    mkdir femus.build

    cd femus.build

    cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE="[Debug Release RelWithDebInfo MinSizeRel None]"  ../femus

=====


FEMuS Mac installation, contact Anthony Gruber for support.  Note that the optional FParser and Libmesh functionality is not yet usable.
======

Download homebrew

    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Install the following packages 

    Brew install gcc
    Brew install make
    Brew install open-mpi
    Brew install metis
    Brew install parmetis
    Brew install hdf5
    Brew install scalapack
    Brew install boost
    Brew install --cask cmake (for cmake-gui)
    Brew install pkg-config (maybe not necessary)

Install PETSc

From the directory $INSTALLATION_DIR clone petsc

    git clone -b release https://gitlab.com/petsc/petsc.git petsc 
    
    cd petsc
    
Configure PETSc with the following options (tested 12/14/2021 on MBP 2021 -- replace directories according to your own homebrew installations)
    
    ./configure --with-debugging=0 --with-shared-libraries --with-mpi-dir=/opt/homebrew/Cellar/open-mpi/4.1.2 --with-hdf5-dir=/opt/homebrew/Cellar/hdf5/1.12.1 --with-boost-dir=/opt/homebrew/Cellar/boost/1.76.0 --with-metis-dir=/opt/homebrew/Cellar/metis/5.1.0 --with-parmetis-dir=/opt/homebrew/Cellar/parmetis/4.0.3_5 --with-scalapack-dir=/opt/homebrew/Cellar/scalapack/2.1.0_3 --download-mumps --download-blacs --download-suitesparse

Follow the console prompts to compile and test the PETSc library.  All tests should pass.

Install SLEPc

From the directory $INSTALLATION_DIR clone slepc

    git clone -b release https://gitlab.com/slepc/slepc

Put the following lines in your .zshrc (or just run them in local scope from the shell)

    export PETSC_DIR=$INSTALLATION_DIR/petsc 

    export PETSC_ARCH=arch-darwin-c-opt

    export SLEPC_DIR=$INSTALLATION_DIR/slepc

Configure SLEPc

    ./configure

Follow the console prompts to compile and test the SLEPc library.  All tests should pass.

Install MyFEMuS

From the directory $INSTALLATION_DIR clone MyFEMuS and make a directory for the binaries

    git clone https://github.com/agrubertx/MyFEMuS.git

    mkdir femusbin

Navigate to MyFEMuS and checkout branch "anthony"

    cd MyFEMuS

    git checkout anthony

Configure MyFEMuS using cmake-gui. 

    cmake-gui 

    Where is the source code: $INSTALLATION_DIR/MyFEMuS
    
    Where to build the binaries: $INSTALLATION_DIR/feumsbin
    
    CMAKE_BUILD_TYPE choose between release (default) or debug
    
    Press Configure button
    
    Press Generate button

Compile
    
    cd $INSTALLATION_DIR/femusbin
    
    make
    
Run. All applications are built in the folder $INSTALLATION_DIR/femusbin/applications/..



Authors
========

Eugenio Aulisa

Simone Bnà

Giorgio Bornia

Anthony Gruber



License
========

FEMuS is an open-source software distributed under the LGPL license, version 2.1
