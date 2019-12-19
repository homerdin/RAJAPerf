
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

#include "POLYBENCH_ATAX.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf
{
namespace polybench
{

#define POLYBENCH_ATAX_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_tmp {m_tmp, N}; \
  cl::sycl::buffer<Real_type> d_y {m_y, N}; \
  cl::sycl::buffer<Real_type> d_x {m_x, N}; \
  cl::sycl::buffer<Real_type> d_A {m_A, N*N}; \

#define POLYBENCH_ATAX_TEARDOWN_SYCL

void POLYBENCH_ATAX::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const unsigned long N = m_N;

  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_ATAX_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto tmp = d_tmp.get_access<cl::sycl::access::mode::write>(h);
          auto y = d_y.get_access<cl::sycl::access::mode::write>(h);
          auto x = d_x.get_access<cl::sycl::access::mode::read>(h);
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybenchATAX_1>(cl::sycl::range<1> {N},
                                              [=] (cl::sycl::item<1> item) {
            int i = item.get_id(0);

            POLYBENCH_ATAX_BODY1;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_ATAX_BODY2;
            }
            POLYBENCH_ATAX_BODY3;
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto tmp = d_tmp.get_access<cl::sycl::access::mode::read>(h);
          auto y = d_y.get_access<cl::sycl::access::mode::read_write>(h);
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybenchATAX_2>(cl::sycl::range<1> {N},
                                              [=] (cl::sycl::item<1> item) {
            int j = item.get_id(0);

            POLYBENCH_ATAX_BODY4;
            for (Index_type i = 0; i < N; ++i ) {
              POLYBENCH_ATAX_BODY5;
            }
            POLYBENCH_ATAX_BODY6;
          });
        }); 
      }
    
      stopTimer();
    }

    POLYBENCH_ATAX_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_ATAX : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
