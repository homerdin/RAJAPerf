 
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

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_pin {m_pin, m_N * m_N}; \
  cl::sycl::buffer<Real_type> d_pout {m_pout, m_N * m_N};

#define POLYBENCH_FLOYD_WARSHALL_TEARDOWN_SYCL

void POLYBENCH_FLOYD_WARSHALL::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const unsigned long N = m_N;

  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_FLOYD_WARSHALL_DATA_SETUP_SYCL;

      const size_t block_size = 16;

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(N, block_size);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type k = 0; k < N; ++k) {

          qu.submit([&] (cl::sycl::handler& h)
          {
            auto pin = d_pin.get_access<cl::sycl::access::mode::read>(h);
            auto pout = d_pout.get_access<cl::sycl::access::mode::write>(h);

            h.parallel_for<class polybenchFLOYDWARSHALL>(cl::sycl::nd_range<2>
                                                         {cl::sycl::range<2> {grid_size, grid_size},
                                                          cl::sycl::range<2> {block_size, block_size}},
                                                       [=] (cl::sycl::nd_item<2> item) {

              Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
              Index_type j = item.get_group(1) * item.get_local_range(1) + item.get_local_id(1);

              if (i < N && j < N) { 
                POLYBENCH_FLOYD_WARSHALL_BODY
              }
            });
          });
        }
      }
      qu.wait();
      stopTimer();
    } // buffer scope

    POLYBENCH_FLOYD_WARSHALL_TEARDOWN_SYCL;

  } else {
      std::cout << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
  
