#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#################################################################################

BUILD_SUFFIX=lc_blueos-clang-upstream-2019.08.15_omptarget
RAJA_HOSTCONFIG=../tpl/RAJA/host-configs/lc-builds/blueos/clang_X.cmake

rm -rf build_${BUILD_SUFFIX} >/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

module load cmake/3.14.5

## NOTE: RAJA tests are turned off due to compilation issues.

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/tce/packages/clang/clang-upstream-2019.08.15/bin/clang++ \
  -DENABLE_CUDA=Off \
  -DENABLE_OPENMP=On \
  -DENABLE_TARGET_OPENMP=On \
  -DOpenMP_CXX_FLAGS="-fopenmp;-fopenmp-targets=nvptx64-nvidia-cuda" \
  -DENABLE_ALL_WARNINGS=Off \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
