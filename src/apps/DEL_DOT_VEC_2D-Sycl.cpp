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

#include "DEL_DOT_VEC_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "AppsData.hpp"

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

#define DEL_DOT_VEC_2D_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_div {m_div, m_array_length}; \
  cl::sycl::buffer<Index_type> d_real_zones {m_domain->real_zones, iend}; \
  cl::sycl::buffer<Real_type> d_x {m_x, m_array_length}; \
  cl::sycl::buffer<Real_type> d_y {m_y, m_array_length}; \
  cl::sycl::buffer<Real_type> d_xdot {m_xdot, m_array_length}; \
  cl::sycl::buffer<Real_type> d_ydot {m_ydot, m_array_length}; \
\
  const Real_type ptiny = m_ptiny; \
  const Real_type half = m_half; \
\
  force_memcpy_real(d_div, qu); \
  force_memcpy_index(d_real_zones, qu); \
  force_memcpy_real(d_x, qu); \
  force_memcpy_real(d_y, qu); \
  force_memcpy_real(d_xdot, qu); \
  force_memcpy_real(d_ydot, qu);

#define DEL_DOT_VEC_2D_DATA_TEARDOWN_SYCL

void DEL_DOT_VEC_2D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = m_domain->n_real_zones;

  if ( vid == Base_SYCL ) {
    {
      ResReal_ptr x = m_x;
      ResReal_ptr y = m_y;
      ResReal_ptr xdot = m_xdot;
      ResReal_ptr ydot = m_ydot;

      Index_type v1,v2,v3,v4 ;
      DEL_DOT_VEC_2D_DATA_SETUP_SYCL;

      v4 = 0;
      v1 = v4 + 1;
      v2 = v1 + m_domain->jp;
      v3 = v4 + m_domain->jp;

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto div = d_div.get_access<cl::sycl::access::mode::write>(h);
          auto real_zones = d_real_zones.get_access<cl::sycl::access::mode::read>(h);
          auto x1 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v1);
          auto x2 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v2);
          auto x3 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v3);
          auto x4 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v4);
          auto y1 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v1);
          auto y2 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v2);
          auto y3 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v3);
          auto y4 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v4);
          auto fx1 = d_xdot.get_access<cl::sycl::access::mode::read>(h, m_array_length, v1);
          auto fx2 = d_xdot.get_access<cl::sycl::access::mode::read>(h, m_array_length, v2);
          auto fx3 = d_xdot.get_access<cl::sycl::access::mode::read>(h, m_array_length, v3);
          auto fx4 = d_xdot.get_access<cl::sycl::access::mode::read>(h, m_array_length, v4);
          auto fy1 = d_ydot.get_access<cl::sycl::access::mode::read>(h, m_array_length, v1);
          auto fy2 = d_ydot.get_access<cl::sycl::access::mode::read>(h, m_array_length, v2);
          auto fy3 = d_ydot.get_access<cl::sycl::access::mode::read>(h, m_array_length, v3);
          auto fy4 = d_ydot.get_access<cl::sycl::access::mode::read>(h, m_array_length, v4);

          h.parallel_for<class syclDelDotVec>(cl::sycl::nd_range<1> {grid_size, block_size},
                                              [=] (cl::sycl::nd_item<1> item) {

            Index_type ii = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
            if (ii < iend) {
              DEL_DOT_VEC_2D_BODY_INDEX
              DEL_DOT_VEC_2D_BODY
            }
          });
        });
      
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    } // Buffer Destruction

    DEL_DOT_VEC_2D_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  DEL_DOT_VEC_2D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
