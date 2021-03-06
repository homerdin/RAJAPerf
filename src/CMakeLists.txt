###############################################################################
# Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

include_directories(.)

add_subdirectory(common)
add_subdirectory(apps)
add_subdirectory(basic)
add_subdirectory(lcals)
add_subdirectory(polybench)
add_subdirectory(stream)

set(RAJA_PERFSUITE_EXECUTABLE_DEPENDS
    common
    apps
    basic
    lcals
    polybench
    stream)
list(APPEND RAJA_PERFSUITE_EXECUTABLE_DEPENDS ${RAJA_PERFSUITE_DEPENDS}) 

if(ENABLE_NO_LIBS)
  remove_definitions(-DRUN_RAJA_SEQ -DRUN_OPENMP )

blt_add_executable(
  NAME raja-perf-no-libs.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  apps/AppsData.cpp
  apps/DEL_DOT_VEC_2D.cpp
  apps/DEL_DOT_VEC_2D-OMPTarget.cpp
  apps/DEL_DOT_VEC_2D-Sycl.cpp
  apps/ENERGY.cpp
  apps/ENERGY-OMPTarget.cpp
  apps/ENERGY-Sycl.cpp
  apps/FIR.cpp
  apps/FIR-OMPTarget.cpp
  apps/FIR-Sycl.cpp
  apps/PRESSURE.cpp
  apps/PRESSURE-OMPTarget.cpp
  apps/PRESSURE-Sycl.cpp
  apps/LTIMES.cpp
  apps/LTIMES-OMPTarget.cpp
  apps/LTIMES_NOVIEW.cpp
  apps/LTIMES_NOVIEW-OMPTarget.cpp
  apps/LTIMES_NOVIEW-Sycl.cpp
  apps/WIP-COUPLE.cpp
  apps/VOL3D.cpp
  apps/VOL3D-OMPTarget.cpp
  apps/VOL3D-Sycl.cpp
  basic/DAXPY.cpp
  basic/DAXPY-OMPTarget.cpp
  basic/DAXPY-Sycl.cpp
  basic/IF_QUAD.cpp
  basic/IF_QUAD-OMPTarget.cpp
  basic/IF_QUAD-Sycl.cpp
  basic/INIT3.cpp
  basic/INIT3-OMPTarget.cpp
  basic/INIT3-Sycl.cpp
  basic/INIT_VIEW1D.cpp
  basic/INIT_VIEW1D-OMPTarget.cpp
  basic/INIT_VIEW1D-Sycl.cpp
  basic/INIT_VIEW1D_OFFSET.cpp
  basic/INIT_VIEW1D_OFFSET-OMPTarget.cpp
  basic/INIT_VIEW1D_OFFSET-Sycl.cpp
  basic/MULADDSUB.cpp
  basic/MULADDSUB-OMPTarget.cpp
  basic/MULADDSUB-Sycl.cpp
  basic/NESTED_INIT.cpp
  basic/NESTED_INIT-OMPTarget.cpp
  basic/NESTED_INIT-Sycl.cpp
  basic/REDUCE3_INT.cpp
  basic/REDUCE3_INT-OMPTarget.cpp
  basic/REDUCE3_INT-Sycl.cpp
  basic/TRAP_INT.cpp
  basic/TRAP_INT-OMPTarget.cpp
  lcals/DIFF_PREDICT.cpp
  lcals/DIFF_PREDICT-OMPTarget.cpp
  lcals/DIFF_PREDICT-Sycl.cpp
  lcals/EOS.cpp
  lcals/EOS-OMPTarget.cpp
  lcals/EOS-Sycl.cpp
  lcals/FIRST_DIFF.cpp
  lcals/FIRST_DIFF-OMPTarget.cpp
  lcals/FIRST_DIFF-Sycl.cpp
  lcals/HYDRO_1D.cpp
  lcals/HYDRO_1D-OMPTarget.cpp
  lcals/HYDRO_1D-Sycl.cpp
  lcals/HYDRO_2D.cpp
  lcals/HYDRO_2D-OMPTarget.cpp
  lcals/HYDRO_2D-Sycl.cpp
  lcals/INT_PREDICT.cpp
  lcals/INT_PREDICT-OMPTarget.cpp
  lcals/INT_PREDICT-Sycl.cpp
  lcals/PLANCKIAN.cpp
  lcals/PLANCKIAN-OMPTarget.cpp
  lcals/PLANCKIAN-Sycl.cpp
  polybench/POLYBENCH_2MM.cpp
  polybench/POLYBENCH_2MM-OMPTarget.cpp
  polybench/POLYBENCH_2MM-Sycl.cpp
  polybench/POLYBENCH_3MM.cpp
  polybench/POLYBENCH_3MM-OMPTarget.cpp
  polybench/POLYBENCH_3MM-Sycl.cpp
  polybench/POLYBENCH_ADI.cpp
  polybench/POLYBENCH_ADI-OMPTarget.cpp
  polybench/POLYBENCH_ADI-Sycl.cpp
  polybench/POLYBENCH_ATAX.cpp
  polybench/POLYBENCH_ATAX-OMPTarget.cpp
  polybench/POLYBENCH_ATAX-Sycl.cpp
  polybench/POLYBENCH_FDTD_2D.cpp
  polybench/POLYBENCH_FDTD_2D-OMPTarget.cpp
  polybench/POLYBENCH_FDTD_2D-Sycl.cpp
  polybench/POLYBENCH_FLOYD_WARSHALL.cpp
  polybench/POLYBENCH_FLOYD_WARSHALL-OMPTarget.cpp
  polybench/POLYBENCH_FLOYD_WARSHALL-Sycl.cpp
  polybench/POLYBENCH_GEMM.cpp
  polybench/POLYBENCH_GEMM-OMPTarget.cpp
  polybench/POLYBENCH_GEMM-Sycl.cpp
  polybench/POLYBENCH_GEMVER.cpp
  polybench/POLYBENCH_GEMVER-OMPTarget.cpp
  polybench/POLYBENCH_GEMVER-Sycl.cpp
  polybench/POLYBENCH_GESUMMV.cpp
  polybench/POLYBENCH_GESUMMV-OMPTarget.cpp
  polybench/POLYBENCH_GESUMMV-Sycl.cpp
  polybench/POLYBENCH_HEAT_3D.cpp
  polybench/POLYBENCH_HEAT_3D-OMPTarget.cpp
  polybench/POLYBENCH_HEAT_3D-Sycl.cpp
  polybench/POLYBENCH_JACOBI_1D.cpp
  polybench/POLYBENCH_JACOBI_1D-OMPTarget.cpp
  polybench/POLYBENCH_JACOBI_1D-Sycl.cpp
  polybench/POLYBENCH_JACOBI_2D.cpp
  polybench/POLYBENCH_JACOBI_2D-OMPTarget.cpp
  polybench/POLYBENCH_JACOBI_2D-Sycl.cpp
  polybench/POLYBENCH_MVT.cpp
  polybench/POLYBENCH_MVT-OMPTarget.cpp
  polybench/POLYBENCH_MVT-Sycl.cpp
  stream/ADD.cpp
  stream/ADD-OMPTarget.cpp
  stream/ADD-Sycl.cpp
  stream/COPY.cpp
  stream/COPY-OMPTarget.cpp
  stream/COPY-Sycl.cpp
  stream/DOT.cpp
  stream/DOT-OMPTarget.cpp
  stream/MUL.cpp
  stream/MUL-OMPTarget.cpp
  stream/MUL-Sycl.cpp
  stream/TRIAD.cpp
  stream/TRIAD-OMPTarget.cpp
  stream/TRIAD-Sycl.cpp
  common/DataUtils.cpp
  common/Executor.cpp
  common/KernelBase.cpp
  common/OutputUtils.cpp
  common/RAJAPerfSuite.cpp
  common/RPTypes.hpp
  common/RunParams.cpp
  common/SyclDataUtils.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_DEPENDS}
)

else()
blt_add_executable(
  NAME raja-perf.exe
  SOURCES RAJAPerfSuiteDriver.cpp
  DEPENDS_ON ${RAJA_PERFSUITE_EXECUTABLE_DEPENDS}
  )
endif()
