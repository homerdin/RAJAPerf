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

#include "INIT3.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

#define INIT3_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_out1 {m_out1, iend}; \
  cl::sycl::buffer<Real_type> d_out2 {m_out2, iend}; \
  cl::sycl::buffer<Real_type> d_out3 {m_out3, iend}; \
  cl::sycl::buffer<Real_type> d_in1 {m_in1, iend}; \
  cl::sycl::buffer<Real_type> d_in2 {m_in2, iend}; \
\
  force_memcpy_real(d_out1, qu); \
  force_memcpy_real(d_out2, qu); \
  force_memcpy_real(d_out3, qu); \
  force_memcpy_real(d_in1, qu); \
  force_memcpy_real(d_in2, qu);

#define INIT3_DATA_TEARDOWN_SYCL

void INIT3::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  if ( vid == Base_SYCL ) {
    {
      INIT3_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto out1 = d_out1.get_access<cl::sycl::access::mode::write>(h);
          auto out2 = d_out2.get_access<cl::sycl::access::mode::write>(h);
          auto out3 = d_out3.get_access<cl::sycl::access::mode::write>(h);
          auto in1 = d_in1.get_access<cl::sycl::access::mode::read>(h);
          auto in2 = d_in2.get_access<cl::sycl::access::mode::read>(h);

          const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

          h.parallel_for<class syclInit3>(cl::sycl::nd_range<1>{grid_size, block_size},
                                          [=] (cl::sycl::nd_item<1> item ) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            if (i < iend) {
              INIT3_BODY
            }
          });
        });
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    } // Block to trigger buffer destruction

    INIT3_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  INIT3 : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
