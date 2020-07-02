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

#include "MUL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace stream
{

#define MUL_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_b {m_b, iend}; \
  cl::sycl::buffer<Real_type> d_c {m_c, iend}; \
\
  Real_type alpha = m_alpha; \

#define MUL_DATA_TEARDOWN_SYCL

void MUL::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_SYCL ) {
    {
      MUL_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

        qu.submit([&] (cl::sycl::handler& h) {

          auto b = d_b.get_access<cl::sycl::access::mode::write>(h);
          auto c = d_c.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class MUL>(cl::sycl::nd_range<1> {grid_size, block_size},
                                    [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

            if (i < iend) {
              MUL_BODY
            }
          });
        });
      }
      qu.wait();
      stopTimer();
    }

    MUL_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  MUL : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_Sycl
