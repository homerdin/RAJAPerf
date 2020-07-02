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

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <CL/sycl.hpp>

#include <iostream>

namespace rajaperf
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
RAJA_DEVICE
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/cl::sycl::sqrt(denom);
   return denom;
}


#define TRAP_INT_DATA_SETUP_SYCL \
const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  Real_type x0 = m_x0; \
  Real_type xp = m_xp; \
  Real_type y = m_y; \
  Real_type yp = m_yp; \
  Real_type h = m_h;

#define TRAP_INT_DATA_TEARDOWN_SYCL // nothing to do here...

void TRAP_INT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  if ( vid == Base_SYCL ) {

    TRAP_INT_DATA_SETUP_SYCL;

    Real_type sumx;

    const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);
    const size_t groups = RAJA_DIVIDE_CEILING_INT(iend, block_size);
    Real_type tsum[groups];

    for (int i = 0; i < groups; i++)
      tsum[i] = 0.0;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      {
        sumx = m_sumx_init;

        cl::sycl::buffer<Real_type> d_sumx(tsum, groups);

        qu.submit( [&] (cl::sycl::handler& cgh)
        {
          auto t_sumx = d_sumx.get_access<cl::sycl::access::mode::write>(cgh);

          cl::sycl::accessor<Real_type, 1, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local> psumx(block_size, cgh);

          cgh.parallel_for<class Reduce3>(cl::sycl::nd_range<1>{grid_size, block_size},
                                        [=] (cl::sycl::nd_item<1> item ) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            Index_type tid = item.get_local_id(0);

            psumx[tid] = 0.0;

            if (i < iend) {
                  Real_type x = x0 + i*h;
                  Real_type val = trap_int_func(x, y, xp, yp);
                  psumx[tid] += val;
            }

            item.barrier(cl::sycl::access::fence_space::local_space);

            for (i = item.get_local_range(0) / 2; i > 0; i /=2) {
              if (tid < i) {
                psumx[tid] += psumx[tid+i];
              }

              item.barrier(cl::sycl::access::fence_space::local_space);
            }

            if (tid == 0) {
              t_sumx[0].fetch_add(psumx[tid]);
            }
          });
        });

      } // Buffer Destruction

      // Final reduction on host, no double support for atomics on device
      for(Index_type i = 0; i < groups; i++) {
        sumx += tsum[i];
      }

      m_sumx += sumx * h;

    }
    stopTimer();

    TRAP_INT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  TRAP_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
