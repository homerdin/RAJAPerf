  
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

#include "POLYBENCH_GEMM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_GEMM_DATA_SETUP_SYCL \
  const unsigned long ni = m_ni; \
  const unsigned long nj = m_nj; \
  const unsigned long nk = m_nk; \
\
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  cl::sycl::buffer<Real_type> d_A {m_A, ni*nk}; \
  cl::sycl::buffer<Real_type> d_B {m_B, nk*nj}; \
  cl::sycl::buffer<Real_type> d_C {m_C, ni*nj}; \
\
  force_memcpy_real(d_A, qu); \
  force_memcpy_real(d_B, qu); \
  force_memcpy_real(d_C, qu);

#define POLYBENCH_GEMM_TEARDOWN_SYCL

void POLYBENCH_GEMM::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_GEMM_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);
          auto B = d_B.get_access<cl::sycl::access::mode::read>(h);
          auto C = d_C.get_access<cl::sycl::access::mode::write>(h);

          h.parallel_for<class polybenchGEMM>(cl::sycl::range<2> {ni, nj},
                                              [=] (cl::sycl::item<2> item) {
            Index_type i = item.get_id(0);
            Index_type j = item.get_id(1);

            POLYBENCH_GEMM_BODY1;
            for (Index_type k = 0; k < nk; ++k ) {
               POLYBENCH_GEMM_BODY2;
            }
            POLYBENCH_GEMM_BODY3;
          });
        });
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    } // buffer scope

    POLYBENCH_GEMM_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_GEMM : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
