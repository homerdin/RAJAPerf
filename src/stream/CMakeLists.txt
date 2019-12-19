###############################################################################
# Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

blt_add_library(
  NAME stream
  SOURCES ADD.cpp
          ADD-Cuda.cpp
          ADD-OMPTarget.cpp
          ADD-Sycl.cpp
          COPY.cpp 
          COPY-Cuda.cpp
          COPY-OMPTarget.cpp
          COPY-Sycl.cpp
          DOT.cpp 
          DOT-Cuda.cpp 
          DOT-OMPTarget.cpp 
          MUL.cpp 
          MUL-Cuda.cpp 
          MUL-OMPTarget.cpp 
          MUL-Sycl.cpp
          TRIAD.cpp 
          TRIAD-Cuda.cpp 
          TRIAD-OMPTarget.cpp 
          TRIAD-Sycl.cpp
  DEPENDS_ON common ${RAJA_PERFSUITE_DEPENDS}
  )
