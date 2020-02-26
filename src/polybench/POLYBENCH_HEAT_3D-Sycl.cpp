  
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

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_HEAT_3D_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_A {m_Ainit, m_N*m_N*m_N}; \
  cl::sycl::buffer<Real_type> d_B {m_Binit, m_N*m_N*m_N}; \
\
  d_A.set_final_data(m_A); \
  d_B.set_final_data(m_B); \
\
  force_memcpy_real(d_A, qu); \
  force_memcpy_real(d_B, qu);


#define POLYBENCH_HEAT_3D_TEARDOWN_SYCL

void POLYBENCH_HEAT_3D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const unsigned long N = m_N;
  const Index_type tsteps = m_tsteps;

  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_HEAT_3D_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type t = 0; t < tsteps; ++t) {
          qu.submit([&] (cl::sycl::handler& h)
          {
            auto A = d_A.get_access<cl::sycl::access::mode::read>(h);
            auto B = d_B.get_access<cl::sycl::access::mode::write>(h);

            h.parallel_for<class polybenchHEAT3D_1>(cl::sycl::range<3> {N-2,N-2,N-2},
                                                    cl::sycl::id<3> {1,1,1},
                                                    [=] (cl::sycl::item<3> item) {
              Index_type i = item.get_id(0);
              Index_type j = item.get_id(1);
              Index_type k = item.get_id(2);

              POLYBENCH_HEAT_3D_BODY1

            });
          });

          qu.submit([&] (cl::sycl::handler& h)
          {
            auto A = d_A.get_access<cl::sycl::access::mode::write>(h);
            auto B = d_B.get_access<cl::sycl::access::mode::read>(h);

            h.parallel_for<class polybenchHEAT3D_2>(cl::sycl::range<3> {N-2,N-2,N-2},
                                                    cl::sycl::id<3> {1,1,1},
                                                    [=] (cl::sycl::item<3> item) {
              Index_type i = item.get_id(0);
              Index_type j = item.get_id(1);
              Index_type k = item.get_id(2);

              POLYBENCH_HEAT_3D_BODY2

            });
          });
        }
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    }

    POLYBENCH_HEAT_3D_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_HEAT_3D : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
