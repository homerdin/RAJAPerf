  
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

#include "POLYBENCH_GEMVER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_GEMVER_DATA_SETUP_SYCL \
  unsigned long n = m_n; \
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  cl::sycl::buffer<Real_type> d_A {m_A, m_n * m_n}; \
  cl::sycl::buffer<Real_type> d_u1 {m_u1, m_n}; \
  cl::sycl::buffer<Real_type> d_v1 {m_v1, m_n}; \
  cl::sycl::buffer<Real_type> d_u2 {m_u2, m_n}; \
  cl::sycl::buffer<Real_type> d_v2 {m_v2, m_n}; \
  cl::sycl::buffer<Real_type> d_w {m_w, m_n}; \
  cl::sycl::buffer<Real_type> d_x {m_x, m_n}; \
  cl::sycl::buffer<Real_type> d_y {m_y, m_n}; \
  cl::sycl::buffer<Real_type> d_z {m_z, m_n};

#define POLYBENCH_GEMVER_TEARDOWN_SYCL

void POLYBENCH_GEMVER::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  
  if ( vid == Base_SYCL ) {
    {

      POLYBENCH_GEMVER_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto A = d_A.get_access<cl::sycl::access::mode::read_write>(h);
          auto u1 = d_u1.get_access<cl::sycl::access::mode::read>(h);
          auto v1 = d_v1.get_access<cl::sycl::access::mode::read>(h);
          auto u2 = d_u2.get_access<cl::sycl::access::mode::read>(h);
          auto v2 = d_v2.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybenchGEMVER_1>(cl::sycl::range<2> {n, n},
                                                  [=] (cl::sycl::item<2> item) {
            Index_type i = item.get_id(0);
            Index_type j = item.get_id(1);

            POLYBENCH_GEMVER_BODY1
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);
          auto y = d_y.get_access<cl::sycl::access::mode::read>(h);
          auto x = d_x.get_access<cl::sycl::access::mode::read_write>(h);

          h.parallel_for<class polybenchGEMVER_2>(cl::sycl::range<1> {n},
                                                  [=] (cl::sycl::item<1> item) {
            Index_type i = item.get_id(0);

            POLYBENCH_GEMVER_BODY2
            for (Index_type j = 0; j < n; j++) {
              POLYBENCH_GEMVER_BODY3
            }
            POLYBENCH_GEMVER_BODY4
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto x = d_x.get_access<cl::sycl::access::mode::read_write>(h);
          auto z = d_z.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybenchGEMVER_3>(cl::sycl::range<1> {n},
                                                  [=] (cl::sycl::item<1> item) {
            Index_type i = item.get_id(0);

            POLYBENCH_GEMVER_BODY5
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto w = d_w.get_access<cl::sycl::access::mode::read_write>(h);
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);
          auto x = d_x.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybenchGEMVER_4>(cl::sycl::range<1> {n},
                                                  [=] (cl::sycl::item<1> item) {
            Index_type i = item.get_id(0);

            POLYBENCH_GEMVER_BODY6;
            for (Index_type j = 0; j < n; j++) {
              POLYBENCH_GEMVER_BODY7;
            }
            POLYBENCH_GEMVER_BODY8;
          });
        });
      }
      qu.wait();
      stopTimer();
    }

    POLYBENCH_GEMVER_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_GEMVER : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_Sycl
