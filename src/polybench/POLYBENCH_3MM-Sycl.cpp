  
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

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_3MM_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_A {m_A, m_ni * m_nk}; \
  cl::sycl::buffer<Real_type> d_B {m_B, m_nk * m_nj}; \
  cl::sycl::buffer<Real_type> d_C {m_C, m_nj * m_nm}; \
  cl::sycl::buffer<Real_type> d_D {m_D, m_nm * m_nl}; \
  cl::sycl::buffer<Real_type> d_E {m_E, m_ni * m_nj}; \
  cl::sycl::buffer<Real_type> d_F {m_F, m_nj * m_nl}; \
  cl::sycl::buffer<Real_type> d_G {m_G, m_ni * m_nl};

#define POLYBENCH_3MM_TEARDOWN_SYCL

void POLYBENCH_3MM::runSyclVariant(VariantID vid)
{
  const unsigned long run_reps = getRunReps();
  const unsigned long ni = m_ni;
  const unsigned long nj = m_nj;
  const unsigned long nk = m_nk;
  const unsigned long nl = m_nl;
  const unsigned long nm = m_nm;

  
  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_3MM_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);
          auto B = d_B.get_access<cl::sycl::access::mode::read>(h);
          auto E = d_E.get_access<cl::sycl::access::mode::write>(h);

          h.parallel_for<class polybench3MM_1>(cl::sycl::range<2> {ni, nj},
                                               [=] (cl::sycl::item<2> item) {
            Index_type i = item.get_id(0);
            Index_type j = item.get_id(1);

            POLYBENCH_3MM_BODY1;
            for (Index_type k=0; k < nk; ++k) {
              POLYBENCH_3MM_BODY2;
            }
            POLYBENCH_3MM_BODY3;

          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto C = d_C.get_access<cl::sycl::access::mode::read>(h);
          auto D = d_D.get_access<cl::sycl::access::mode::read>(h);
          auto F = d_F.get_access<cl::sycl::access::mode::write>(h);

          h.parallel_for<class polybench3MM_2>(cl::sycl::range<2> {nj, nl},
                                               [=] (cl::sycl::item<2> item) {
            Index_type j = item.get_id(0);
            Index_type l = item.get_id(1);

            POLYBENCH_3MM_BODY4;
            for (Index_type m=0; m < nm; ++m) {
              POLYBENCH_3MM_BODY5;
            }
            POLYBENCH_3MM_BODY6;

          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto E = d_E.get_access<cl::sycl::access::mode::read>(h);
          auto F = d_F.get_access<cl::sycl::access::mode::read>(h);
          auto G = d_G.get_access<cl::sycl::access::mode::write>(h);

          h.parallel_for<class polybench3MM_3>(cl::sycl::range<2> {ni, nl},
                                               [=] (cl::sycl::item<2> item) {
            Index_type i = item.get_id(0);
            Index_type l = item.get_id(1);

            POLYBENCH_3MM_BODY7;
            for (Index_type j=0; j < nj; ++j) {
              POLYBENCH_3MM_BODY8;
            }
            POLYBENCH_3MM_BODY9;

          });
        });
      }
      stopTimer();
    }

    POLYBENCH_3MM_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_3MM : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
  
