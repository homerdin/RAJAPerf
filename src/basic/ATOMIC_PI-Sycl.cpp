//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC_PI.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;


#define ATOMIC_PI_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(pi, m_pi, 1, qu);

#define ATOMIC_PI_DATA_TEARDOWN_SYCL \
  deallocSyclDeviceData(pi, qu);

__global__ void atomic_pi(Real_ptr pi,
                          Real_type dx, 
                          Index_type iend) 
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     double x = (double(i) + 0.5) * dx;
     RAJA::atomicAdd<RAJA::sycl_atomic>(pi, dx / (1.0 + x * x));
   }
}


void ATOMIC_PI::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  ATOMIC_PI_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    ATOMIC_PI_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initSyclDeviceData(pi, &m_pi_init, 1);
 
      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);
      qu.submit([&] (cl::sycl::handler h) {

        h.parallel_for(cl::sycl::nd_range<1>(grid_size, block_size),
                       [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
          if (i < iend) {
            RAJA::atomicAdd<RAJA::sycl_atomic>(pi, dx / (1.0 + x * x));
          }
        });
      });
      atomic_pi<<<grid_size, block_size>>>( pi, dx, iend ); 


      getSyclDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    ATOMIC_PI_DATA_TEARDOWN_SYCL;

  } else if ( vid == RAJA_SYCL ) {

    ATOMIC_PI_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initSyclDeviceData(pi, &m_pi_init, 1);

      RAJA::forall< RAJA::sycl_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          double x = (double(i) + 0.5) * dx;
          RAJA::atomicAdd<RAJA::sycl_atomic>(pi, dx / (1.0 + x * x));
      });

      getSyclDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    ATOMIC_PI_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  ATOMIC_PI : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
