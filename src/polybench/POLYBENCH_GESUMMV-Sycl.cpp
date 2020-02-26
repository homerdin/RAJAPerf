  
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

#include "POLYBENCH_GESUMMV.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_GESUMMV_DATA_SETUP_SYCL \
  const unsigned long N = m_N; \
\
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  cl::sycl::buffer<Real_type> d_x {m_x, N}; \
  cl::sycl::buffer<Real_type> d_y {m_y, N}; \
  cl::sycl::buffer<Real_type> d_A {m_A, N*N}; \
  cl::sycl::buffer<Real_type> d_B {m_B, N*N}; \
\
  force_memcpy_real(d_x, qu); \
  force_memcpy_real(d_y, qu); \
  force_memcpy_real(d_A, qu); \
  force_memcpy_real(d_B, qu);

#define POLYBENCH_GESUMMV_TEARDOWN_SYCL

void POLYBENCH_GESUMMV::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_GESUMMV_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto x = d_x.get_access<cl::sycl::access::mode::read>(h);
          auto y = d_y.get_access<cl::sycl::access::mode::write>(h);
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);
          auto B = d_B.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybenchGESUMMV>(cl::sycl::range<1> {N},
                                                 [=] (cl::sycl::item<1> item) {

            Index_type i = item.get_id(0);

            POLYBENCH_GESUMMV_BODY1;
            for (Index_type j = 0; j < N; ++j ) {
              POLYBENCH_GESUMMV_BODY2;
            }
            POLYBENCH_GESUMMV_BODY3;

          });
        });
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    }

    POLYBENCH_GESUMMV_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_GESUMMV : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
  
