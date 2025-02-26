#!/bin/bash

# unset MPICXX 
# unset MPICC
# export C_INCLUDE_PATH=$ROCM_PATH/llvm/include
# export CPLUS_INCLUDE_PATH=$ROCM_PATH/llvm/include
# export LIBRARY_PATH=${EBROOTBOOST}/lib:${EBROOTBUILDTOOLS}/lib:${ROCM_PATH}/lib

SOURCE_DIR=/pfs/lustrep1/projappl/project_465000412/juananto/software/quda
BUILD_DIR=/pfs/lustrep1/projappl/project_465000412/juananto/software/quda/build

#  -D CMAKE_DISABLE_SOURCE_CHANGES=ON \
#  -D CMAKE_DISABLE_IN_SOURCE_BUILD=ON \

#  -D CMAKE_C_FLAGS="--gcc-toolchain=$GCC_PATH/snos/" \
#  -D CMAKE_CXX_FLAGS="--gcc-toolchain=$GCC_PATH/snos/" \
#  -D ROCM_CXX_FLAGS="--gcc-toolchain=$GCC_PATH/snos/" \
cmake  \
 -D ROCM_PATH=${ROCM_PATH} \
 -D CMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/clang \
 -D CMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
 -D CMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
 -D CMAKE_BUILD_TYPE=Release \
 -D QUDA_TARGET_TYPE=HIP \
 -D QUDA_BUILD_ALL_TESTS=ON \
 -D QUDA_MPI=ON \
 -D QUDA_DIRAC_CLOVER=ON \
 -D QUDA_DIRAC_CLOVER_HASENBUSCH=ON \
 -D QUDA_DIRAC_DOMAIN_WALL=OFF \
 -D QUDA_DIRAC_NDEG_TWISTED_MASS=ON \
 -D QUDA_DIRAC_NDEG_TWISTED_CLOVER=ON \
 -D QUDA_DIRAC_STAGGERED=ON \
 -D QUDA_DIRAC_TWISTED_MASS=ON \
 -D QUDA_DIRAC_TWISTED_CLOVER=ON \
 -D QUDA_DIRAC_WILSON=ON \
 -D QUDA_FORCE_GAUGE=ON \
 -D QUDA_FORCE_HISQ=OFF \
 -D QUDA_GAUGE_ALG=ON \
 -D QUDA_GAUGE_TOOLS=OFF \
 -D QUDA_MULTIGRID=ON \
 -D QUDA_INTERFACE_MILC=OFF \
 -D QUDA_INTERFACE_CPS=OFF \
 -D QUDA_INTERFACE_QDP=ON \
 -D QUDA_INTERFACE_TIFR=OFF \
 -D QUDA_DOWNLOAD_USQCD=ON \
 -D QUDA_GPU_ARCH=gfx90a \
 -D AMDGPU_TARGETS=gfx90a \
 -D GPU_TARGETS=gfx90a \
 -S "${SOURCE_DIR}" \
 -B "${BUILD_DIR}"






# #!/bin/bash

# # Set up directories
# BUILD_DIR=.
# SOURCE_DIR=/users/juananto/project_465000412/juananto/software/quda
# CACHE_DIR=$HOME/.cache/quda

# # # Create necessary directories
# # mkdir -p $CACHE_DIR/cpm
# # mkdir -p $CACHE_DIR/git
# # export CPM_SOURCE_CACHE=$CACHE_DIR/cpm



# # unset MPICXX 
# # unset MPICC
# # export C_INCLUDE_PATH=$ROCM_PATH/llvm/include
# # export CPLUS_INCLUDE_PATH=$ROCM_PATH/llvm/include
# # export LIBRARY_PATH=${EBROOTBOOST}/lib:${EBROOTBUILDTOOLS}/lib:${ROCM_PATH}/lib


# #  -D CMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/clang \
# #  -D CMAKE_C_FLAGS="--gcc-toolchain=$GCC_PATH/snos/" \
# #  -D CMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \
# #  -D CMAKE_CXX_FLAGS="--gcc-toolchain=$GCC_PATH/snos/" \
# #  -D ROCM_PATH=${ROCM_PATH} \
# #  -D ROCM_CXX_FLAGS="--gcc-toolchain=$GCC_PATH/snos/" \
# #  -D CMAKE_HIP_COMPILER=${ROCM_PATH}/llvm/bin/clang++ \

# #  -D CPM_SOURCE_CACHE=${CPM_SOURCE_CACHE} \



# cmake  \
#  -D CMAKE_BUILD_TYPE=Release \
#  -D QUDA_TARGET_TYPE=HIP \
#  -D QUDA_BUILD_ALL_TESTS=ON \
#  -D QUDA_MPI=ON \
#  -D QUDA_DIRAC_CLOVER=ON \
#  -D QUDA_DIRAC_CLOVER_HASENBUSCH=ON \
#  -D QUDA_DIRAC_DOMAIN_WALL=OFF \
#  -D QUDA_DIRAC_NDEG_TWISTED_MASS=ON \
#  -D QUDA_DIRAC_NDEG_TWISTED_CLOVER=ON \
#  -D QUDA_DIRAC_STAGGERED=ON \
#  -D QUDA_DIRAC_TWISTED_MASS=ON \
#  -D QUDA_DIRAC_TWISTED_CLOVER=ON \
#  -D QUDA_DIRAC_WILSON=ON \
#  -D QUDA_FORCE_GAUGE=ON \
#  -D QUDA_FORCE_HISQ=OFF \
#  -D QUDA_GAUGE_ALG=ON \
#  -D QUDA_GAUGE_TOOLS=OFF \
#  -D QUDA_MULTIGRID=ON \
#  -D QUDA_INTERFACE_MILC=OFF \
#  -D QUDA_INTERFACE_CPS=OFF \
#  -D QUDA_INTERFACE_QDP=ON \
#  -D QUDA_INTERFACE_TIFR=OFF \
#  -D QUDA_DOWNLOAD_USQCD=ON \
#  -D QUDA_GPU_ARCH=gfx90a \
#  -D AMDGPU_TARGETS=gfx90a \
#  -D GPU_TARGETS=gfx90a \
#  -B ${BUILD_DIR} \
#  -S ${SOURCE_DIR}

# #  -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=ON \
# #  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
