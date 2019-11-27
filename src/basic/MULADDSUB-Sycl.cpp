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

#include "MULADDSUB.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define MULADDSUB_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_out1 {m_out1, iend}; \
  cl::sycl::buffer<Real_type> d_out2 {m_out2, iend}; \
  cl::sycl::buffer<Real_type> d_out3 {m_out3, iend}; \
  cl::sycl::buffer<Real_type> d_in1 {m_in1, iend}; \
  cl::sycl::buffer<Real_type> d_in2 {m_in2, iend};

#define MULADDSUB_DATA_TEARDOWN_SYCL

void MULADDSUB::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  if ( vid == Base_SYCL ) {
    {
      MULADDSUB_DATA_SETUP_SYCL;

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

          h.parallel_for<class syclMulAddSub>(cl::sycl::nd_range<1>{grid_size, block_size},
                                          [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            if (i < iend) {
              MULADDSUB_BODY
            }
          });
        });
      }

      stopTimer();
    } // Block to trigger buffer destruction

    MULADDSUB_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  MULADDSUB : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
