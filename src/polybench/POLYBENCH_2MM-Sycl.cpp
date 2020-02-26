  
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

#include "POLYBENCH_2MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>
#include <cmath>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_2MM_DATA_SETUP_SYCL \
  size_t block_size = sqrt(qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>()); \
\
  cl::sycl::buffer<Real_type> d_tmp {m_tmp, m_ni * m_nj}; \
  cl::sycl::buffer<Real_type> d_A {m_A, m_ni * m_nk}; \
  cl::sycl::buffer<Real_type> d_B {m_B, m_nk * m_nj}; \
  cl::sycl::buffer<Real_type> d_C {m_C, m_nj * m_nl}; \
  cl::sycl::buffer<Real_type> d_D {m_D, m_ni * m_nl}; \
\
  Real_type alpha = m_alpha; \
  Real_type beta = m_beta; \
\
  force_memcpy_real(d_tmp, qu); \
  force_memcpy_real(d_A, qu); \
  force_memcpy_real(d_B, qu); \
  force_memcpy_real(d_C, qu); \
  force_memcpy_real(d_D, qu);

#define POLYBENCH_2MM_TEARDOWN_SYCL

void POLYBENCH_2MM::runSyclVariant(VariantID vid)
{
  const unsigned long run_reps = getRunReps();
  const unsigned long ni = m_ni;
  const unsigned long nj = m_nj;
  const unsigned long nk = m_nk;
  const unsigned long nl = m_nl;


  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_2MM_DATA_SETUP_SYCL;

      const size_t ni_grid_size = block_size * RAJA_DIVIDE_CEILING_INT(ni, block_size);
      const size_t nj_grid_size = block_size * RAJA_DIVIDE_CEILING_INT(nj, block_size);
      const size_t nl_grid_size = block_size * RAJA_DIVIDE_CEILING_INT(nl, block_size);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto tmp = d_tmp.get_access<cl::sycl::access::mode::write>(h);
          auto A = d_A.get_access<cl::sycl::access::mode::read>(h);
          auto B = d_B.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class polybench2MM_1>(cl::sycl::nd_range<2> 
                                                 {cl::sycl::range<2> {ni_grid_size, nj_grid_size},
                                                  cl::sycl::range<2> {block_size, block_size}},
                                               [=] (cl::sycl::nd_item<2> item) {

           Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
           Index_type j = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

           if (i < ni && j < nj) {
             POLYBENCH_2MM_BODY1;
             for (Index_type k=0; k < nk; ++k) {
                POLYBENCH_2MM_BODY2;
              }
              POLYBENCH_2MM_BODY3;
            }
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto tmp = d_tmp.get_access<cl::sycl::access::mode::read>(h);
          auto C = d_C.get_access<cl::sycl::access::mode::read>(h);
          auto D = d_D.get_access<cl::sycl::access::mode::write>(h);

          h.parallel_for<class polybench2MM_2>(cl::sycl::nd_range<2>
                                                 {cl::sycl::range<2> {ni_grid_size, nl_grid_size},
                                                  cl::sycl::range<2> {block_size, block_size}},
                                               [=] (cl::sycl::nd_item<2> item) {

           Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
           Index_type l = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

           if(i < ni && l < nl) {        
             POLYBENCH_2MM_BODY4;
             for (Index_type j=0; j < nj; ++j) {
                POLYBENCH_2MM_BODY5;
              }
              POLYBENCH_2MM_BODY6;
            }
          });
        });
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    }

    POLYBENCH_2MM_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_2MM : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_Sycl
  
