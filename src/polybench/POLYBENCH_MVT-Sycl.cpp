  
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

#include "POLYBENCH_MVT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_MVT_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_x1 {m_x1, N}; \
  cl::sycl::buffer<Real_type> d_x2 {m_x2, N}; \
  cl::sycl::buffer<Real_type> d_y1 {m_y1, N}; \
  cl::sycl::buffer<Real_type> d_y2 {m_y2, N}; \
  cl::sycl::buffer<Real_type> d_A {m_A, N*N};

#define POLYBENCH_MVT_TEARDOWN_SYCL

void POLYBENCH_MVT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const unsigned long N = m_N;

  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_MVT_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto x1 = d_x1.get_access<cl::sycl::access::mode::write>(h);
          auto y1 = d_y1.get_access<cl::sycl::access::mode::read>(h);
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybenchMVT_1>(cl::sycl::range<1> {N},
                                               [=] (cl::sycl::item<1> item) {
            Index_type i = item.get_id(0);

            POLYBENCH_MVT_BODY1;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_MVT_BODY2;
            }
            POLYBENCH_MVT_BODY3;

          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto x2 = d_x2.get_access<cl::sycl::access::mode::write>(h);
          auto y2 = d_y2.get_access<cl::sycl::access::mode::read>(h);
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybenchMVT_2>(cl::sycl::range<1> {N},
                                               [=] (cl::sycl::item<1> item) {
            Index_type i = item.get_id(0);

            POLYBENCH_MVT_BODY4;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_MVT_BODY5;
            }
            POLYBENCH_MVT_BODY6;
          });
        });
      }

      stopTimer();
    }

    POLYBENCH_MVT_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_MVT : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
