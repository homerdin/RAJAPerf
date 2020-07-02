  
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-738930
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For details about use and distribution, please read RAJAPerf/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FDTD_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_FDTD_2D_DATA_SETUP_SYCL \
  const unsigned long nx = m_nx; \
  const unsigned long ny = m_ny; \
  const unsigned long tsteps = m_tsteps; \
\
  cl::sycl::buffer<Real_type> d_fict {m_fict, m_tsteps}; \
  cl::sycl::buffer<Real_type> d_ex {m_ex, m_nx * m_ny}; \
  cl::sycl::buffer<Real_type> d_ey {m_ey, m_nx * m_ny}; \
  cl::sycl::buffer<Real_type> d_hz {m_hz, m_nx * m_ny}; \

#define POLYBENCH_FDTD_2D_TEARDOWN_SYCL

void POLYBENCH_FDTD_2D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_FDTD_2D_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          qu.submit([&] (cl::sycl::handler& h)
          {
            auto ey = d_ey.get_access<cl::sycl::access::mode::write>(h);
            auto fict = d_fict.get_access<cl::sycl::access::mode::read>(h);

            h.parallel_for<class polybenchFDTD_1>(cl::sycl::range<1> {ny},
                                                  [=] (cl::sycl::item<1> item) {
              Index_type j = item.get_id(0);

              POLYBENCH_FDTD_2D_BODY1
            });
          });

          qu.submit([&] (cl::sycl::handler& h)
          {
            auto ey = d_ey.get_access<cl::sycl::access::mode::read_write>(h);
            auto hz = d_hz.get_access<cl::sycl::access::mode::read>(h);

            h.parallel_for<class polybenchFDTD_2>(cl::sycl::range<2> {nx-1, ny},
                                                  cl::sycl::id<2> {1, 0},
                                                  [=] (cl::sycl::item<2> item) {
              Index_type i = item.get_id(0);
              Index_type j = item.get_id(1);

              POLYBENCH_FDTD_2D_BODY2
            });
          });

          qu.submit([&] (cl::sycl::handler& h)
          {
            auto ex = d_ex.get_access<cl::sycl::access::mode::read_write>(h);
            auto hz = d_hz.get_access<cl::sycl::access::mode::read>(h);

            h.parallel_for<class polybenchFDTD_3>(cl::sycl::range<2> {nx, ny-1},
                                                  cl::sycl::id<2> {0, 1},
                                                  [=] (cl::sycl::item<2> item) {
              Index_type i = item.get_id(0);
              Index_type j = item.get_id(1);

              POLYBENCH_FDTD_2D_BODY3
            });
          });

          qu.submit([&] (cl::sycl::handler& h)
          {
            auto hz = d_hz.get_access<cl::sycl::access::mode::read_write>(h);
            auto ex = d_ex.get_access<cl::sycl::access::mode::read>(h);
            auto ey = d_ey.get_access<cl::sycl::access::mode::read>(h);

            h.parallel_for<class polybenchFDTD_4>(cl::sycl::range<2> {nx-1, ny-1},
                                                  [=] (cl::sycl::item<2> item) {
              Index_type i = item.get_id(0);
              Index_type j = item.get_id(1);

              POLYBENCH_FDTD_2D_BODY4
            });
          });
        } // tstep loop

      } // run_reps
      qu.wait();
      stopTimer();
    } // Buffer scope

    POLYBENCH_FDTD_2D_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_FDTD_2D : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
