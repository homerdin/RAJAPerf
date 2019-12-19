  
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

#include "POLYBENCH_ADI.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_ADI_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_U {m_U, m_n * m_n}; \
  cl::sycl::buffer<Real_type> d_V {m_V, m_n * m_n}; \
  cl::sycl::buffer<Real_type> d_P {m_P, m_n * m_n}; \
  cl::sycl::buffer<Real_type> d_Q {m_Q, m_n * m_n}; \
\
  const unsigned long n = m_n; \
  const Index_type tsteps = m_tsteps; \
\
  Real_type DX,DY,DT; \
  Real_type B1,B2; \
  Real_type mul1,mul2; \
  Real_type a,b,c,d,e,f; \

#define POLYBENCH_ADI_TEARDOWN_SYCL


void POLYBENCH_ADI::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  
  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_ADI_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       POLYBENCH_ADI_BODY1;

        for (Index_type t = 1; t <= tsteps; ++t) {
          qu.submit([&] (cl::sycl::handler& h)
          {
            auto V = d_V.get_access<cl::sycl::access::mode::read_write>(h);
            auto P = d_P.get_access<cl::sycl::access::mode::read_write>(h);
            auto Q = d_Q.get_access<cl::sycl::access::mode::read_write>(h);
            auto U = d_U.get_access<cl::sycl::access::mode::read>(h);

            h.parallel_for<class polybenchADI_1>(cl::sycl::range<1> {n-2},
                                                 cl::sycl::id<1> {1},
                                                 [=] (cl::sycl::item<1> item) {
              int i = item.get_id(0);

              POLYBENCH_ADI_BODY2;
              for (Index_type j = 1; j < n-1; ++j) {
                POLYBENCH_ADI_BODY3;
              }
              POLYBENCH_ADI_BODY4;
              for (Index_type k = n-2; k >= 1; --k) {
                POLYBENCH_ADI_BODY5;
              }
            });
          });

          qu.submit([&] (cl::sycl::handler& h)
          {
            auto V = d_V.get_access<cl::sycl::access::mode::read>(h);
            auto P = d_P.get_access<cl::sycl::access::mode::read_write>(h);
            auto Q = d_Q.get_access<cl::sycl::access::mode::read_write>(h);
            auto U = d_U.get_access<cl::sycl::access::mode::read_write>(h);

            h.parallel_for<class polybenchADI_2>(cl::sycl::range<1> {n-2},
                                                 cl::sycl::id<1> {1},
                                                 [=] (cl::sycl::item<1> item) {
              int i = item.get_id(0);

              POLYBENCH_ADI_BODY6;
              for (Index_type j = 1; j < n-1; ++j) {
                POLYBENCH_ADI_BODY7;
              }
              POLYBENCH_ADI_BODY8;
              for (Index_type k = n-2; k >= 1; --k) {
                POLYBENCH_ADI_BODY9;
              }
            });
          });
        }  // tstep loop
      }
      stopTimer();
    } // Trigger buffer destruction

    POLYBENCH_ADI_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_ADI : Unknown Sycl variant id = " << vid << std::endl;
  }
}
  
} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
  
