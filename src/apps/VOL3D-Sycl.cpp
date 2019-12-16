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

#include "VOL3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "AppsData.hpp"

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace apps
{

#define VOL3D_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_x {m_x, m_array_length}; \
  cl::sycl::buffer<Real_type> d_y {m_y, m_array_length}; \
  cl::sycl::buffer<Real_type> d_z {m_z, m_array_length}; \
  cl::sycl::buffer<Real_type> d_vol {m_vol, m_array_length}; \
\
  const Real_type vnormq = m_vnormq; \

#define VOL3D_DATA_TEARDOWN_SYCL \

void VOL3D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  const Index_type iterations = iend - ibegin;

  if ( vid == Base_SYCL ) {
    {
      ResReal_ptr x = m_x;
      ResReal_ptr y = m_y;
      ResReal_ptr z = m_z;

      Index_type v0,v1,v2,v3,v4,v5,v6,v7 ;

      v0 = 0;
      v1 = v0 + 1;
      v2 = v0 + m_domain->jp;
      v3 = v1 + m_domain->jp;
      v4 = v0 + m_domain->kp;
      v5 = v1 + m_domain->kp;
      v6 = v2 + m_domain->kp;
      v7 = v3 + m_domain->kp;

      VOL3D_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
 
        const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iterations, block_size);

        qu.submit([&] (cl::sycl::handler& h) {

          auto vol = d_vol.get_access<cl::sycl::access::mode::write>(h);

          auto x0 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v0);
          auto x1 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v1);
          auto x2 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v2);
          auto x3 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v3);
          auto x4 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v4);
          auto x5 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v5);
          auto x6 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v6);
          auto x7 = d_x.get_access<cl::sycl::access::mode::read>(h, m_array_length, v7);
          auto y0 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v0);
          auto y1 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v1);
          auto y2 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v2);
          auto y3 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v3);
          auto y4 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v4);
          auto y5 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v5);
          auto y6 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v6);
          auto y7 = d_y.get_access<cl::sycl::access::mode::read>(h, m_array_length, v7);
          auto z0 = d_z.get_access<cl::sycl::access::mode::read>(h, m_array_length, v0);
          auto z1 = d_z.get_access<cl::sycl::access::mode::read>(h, m_array_length, v1);
          auto z2 = d_z.get_access<cl::sycl::access::mode::read>(h, m_array_length, v2);
          auto z3 = d_z.get_access<cl::sycl::access::mode::read>(h, m_array_length, v3);
          auto z4 = d_z.get_access<cl::sycl::access::mode::read>(h, m_array_length, v4);
          auto z5 = d_z.get_access<cl::sycl::access::mode::read>(h, m_array_length, v5);
          auto z6 = d_z.get_access<cl::sycl::access::mode::read>(h, m_array_length, v6);
          auto z7 = d_z.get_access<cl::sycl::access::mode::read>(h, m_array_length, v7);

          h.parallel_for<class VOL3D>(cl::sycl::nd_range<1> {grid_size, block_size},
                                      [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            i += ibegin;

            if(i < iend) {
              VOL3D_BODY
            }
          });
        });
      }

      stopTimer();
    } // Buffer Destruction
 
    VOL3D_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  VOL3D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
