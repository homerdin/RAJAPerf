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

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <CL/sycl.hpp>

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define REDUCE3_INT_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Int_type> d_vec {m_vec, iend}; \
\
  Int_type vsum = m_vsum_init; \
  Int_type vmin = m_vmin_init; \
  Int_type vmax = m_vmax_init;

#define REDUCE3_INT_DATA_TEARDOWN_SYCL \

void REDUCE3_INT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  if ( vid == Base_SYCL && 0 ) {
    {

      REDUCE3_INT_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        {
          cl::sycl::buffer<Int_type> d_vsum {&m_vsum_init, 1};
          cl::sycl::buffer<Int_type> d_vmin {&m_vmin_init, 1};
          cl::sycl::buffer<Int_type> d_vmax {&m_vmax_init, 1};

          d_vsum.set_final_data(&vsum);
          d_vmin.set_final_data(&vmin);
          d_vmax.set_final_data(&vmax);

          qu.submit( [&] (cl::sycl::handler& h)
          {
            auto vec = d_vec.get_access<cl::sycl::access::mode::read>(h);
            auto t_vsum =  d_vsum.get_access<cl::sycl::access::mode::atomic>(h);
            auto t_vmin =  d_vmin.get_access<cl::sycl::access::mode::atomic>(h);
            auto t_vmax =  d_vmax.get_access<cl::sycl::access::mode::atomic>(h);

            // Limits are templated.  Not allowed in kernel
            Int_type t_sum_init = m_vsum_init;
            Int_type t_min_init = m_vmin_init;
            Int_type t_max_init = m_vmax_init;

            // Local Memory
            cl::sycl::accessor<Int_type, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local> psum(block_size, h);
            cl::sycl::accessor<Int_type, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local> pmin(block_size, h);
            cl::sycl::accessor<Int_type, 1, cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local> pmax(block_size, h);

            const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

            h.parallel_for<class Reduce3>(cl::sycl::nd_range<1>{grid_size, block_size},
                                          [=] (cl::sycl::nd_item<1> item ) {

              Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
              Index_type tid = item.get_local_id(0);

              // Initialize threads local memory
              psum[tid] = t_sum_init;
              pmin[tid] = t_min_init;
              pmax[tid] = t_max_init;

              if (i < iend) {
                psum[tid] += vec[i];
                pmin[tid] = RAJA_MIN(pmin[tid], vec[i]);
                pmax[tid] = RAJA_MAX(pmax[tid], vec[i]);
              }

              item.barrier(cl::sycl::access::fence_space::local_space);

              for (i = item.get_local_range(0) / 2; i > 0; i /=2) {
                if (tid < i) {
                  psum[tid] += psum[tid + i];
                  pmin[tid] = RAJA_MIN(pmin[tid], pmin[tid + i]);
                  pmax[tid] = RAJA_MAX(pmax[tid], pmax[tid + i]);
                }

                item.barrier(cl::sycl::access::fence_space::local_space);
              }

              if (tid == 0) {
                t_vsum[0].fetch_add(psum[tid]);
                t_vmin[0].fetch_min(pmin[tid]);
                t_vmax[0].fetch_max(pmax[tid]);
              }

            });
          });

        } // buffer destruction

        m_vsum += vsum;
        m_vmin = RAJA_MIN(m_vmin, vmin);
        m_vmax = RAJA_MAX(m_vmax, vmax);

      }
      stopTimer();
    } // Original Vector Buffer Destruction

    REDUCE3_INT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  REDUCE3_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

