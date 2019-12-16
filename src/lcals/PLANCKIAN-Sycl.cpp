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

#include "PLANCKIAN.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>
#include <cmath>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace lcals
{

#define PLANCKIAN_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_x {m_x, iend}; \
  cl::sycl::buffer<Real_type> d_y {m_y, iend}; \
  cl::sycl::buffer<Real_type> d_u {m_u, iend}; \
  cl::sycl::buffer<Real_type> d_v {m_v, iend}; \
  cl::sycl::buffer<Real_type> d_w {m_w, iend}; \

#define PLANCKIAN_DATA_TEARDOWN_SYCL

void PLANCKIAN::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  if ( vid == Base_SYCL ) {
    {
      PLANCKIAN_DATA_SETUP_SYCL;

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      using cl::sycl::exp;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto x = d_x.get_access<cl::sycl::access::mode::read>(h);
          auto y = d_y.get_access<cl::sycl::access::mode::read_write>(h);
          auto u = d_u.get_access<cl::sycl::access::mode::read>(h);
          auto v = d_v.get_access<cl::sycl::access::mode::read>(h);
          auto w = d_w.get_access<cl::sycl::access::mode::write>(h);

          h.parallel_for<class syclPlankian>(cl::sycl::nd_range<1> {grid_size, block_size},
                                             [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            if (i < iend) {
              PLANCKIAN_BODY
            }
          });
        });
      }

      stopTimer();
    }

    PLANCKIAN_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  PLANCKIAN : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
