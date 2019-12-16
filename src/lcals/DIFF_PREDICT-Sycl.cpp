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

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace lcals
{

#define DIFF_PREDICT_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_px { m_px, m_array_length }; \
  cl::sycl::buffer<Real_type> d_cx { m_cx, m_array_length }; \
  const Index_type offset = m_offset; \

#define DIFF_PREDICT_DATA_TEARDOWN_SYCL

void DIFF_PREDICT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  if ( vid == Base_SYCL ) {
    {
      DIFF_PREDICT_DATA_SETUP_SYCL;

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto px = d_px.get_access<cl::sycl::access::mode::read_write>(h);
          auto cx = d_cx.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclDiffPredict>(cl::sycl::nd_range<1>{grid_size, block_size},
                                                [=] (cl::sycl::nd_item<1> item ) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            if (i < iend) {
              DIFF_PREDICT_BODY
            }
          });
        });
      }

      stopTimer();
    } // Block to trigger buffer destruction

    DIFF_PREDICT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  DIFF_PREDICT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
