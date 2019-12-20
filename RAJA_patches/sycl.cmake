#
## Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-689114
##
## All rights reserved.
##
## This file is part of RAJA.
##
## For details about use and distribution, please read RAJA/LICENSE.
##

set(RAJA_COMPILER "RAJA_COMPILER_CLANG" CACHE STRING "")

set(CMAKE_CXX_COMPILER "clang++" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-v -O3 -fsycl -std=c++14" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O3 -g -fsycl -std=c++14" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG " -O0 -g -fsycl -std=c++14" CACHE STRING "")
set(CMAKE_CXX_LINK_FLAGS "-fsycl -lOpenCL"  CACHE STRING "")

set(RAJA_RANGE_ALIGN 4 CACHE INT "")
set(RAJA_RANGE_MIN_LENGTH 32 CACHE INT "")
set(RAJA_DATA_ALIGN 64 CACHE INT "")

set(RAJA_HOST_CONFIG_LOADED On CACHE Bool "")
