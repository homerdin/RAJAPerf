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

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

#define PRESSURE_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_compression {m_compression, iend}; \
  cl::sycl::buffer<Real_type> d_bvc {m_bvc, iend}; \
  cl::sycl::buffer<Real_type> d_p_new {m_p_new, iend}; \
  cl::sycl::buffer<Real_type> d_e_old {m_e_old, iend}; \
  cl::sycl::buffer<Real_type> d_vnewc {m_vnewc, iend}; \
\
  const Real_type cls = m_cls; \
  const Real_type p_cut = m_p_cut; \
  const Real_type pmin = m_pmin; \
  const Real_type eosvmax = m_eosvmax; \
\
  force_memcpy_real(d_compression, qu); \
  force_memcpy_real(d_bvc, qu); \
  force_memcpy_real(d_p_new, qu); \
  force_memcpy_real(d_e_old, qu); \
  force_memcpy_real(d_vnewc, qu);

#define PRESSURE_DATA_TEARDOWN_SYCL

void PRESSURE::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const unsigned int ibegin = 0;
  const unsigned int iend = getRunSize();

  if ( vid == Base_SYCL ) {
    {
      PRESSURE_DATA_SETUP_SYCL;

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      using cl::sycl::fabs;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu.submit([&] (cl::sycl::handler& h) {

          auto compression = d_compression.get_access<cl::sycl::access::mode::read>(h);
          auto bvc = d_bvc.get_access<cl::sycl::access::mode::write>(h);

          h.parallel_for<class PRESSURE_1>(cl::sycl::nd_range<1> {grid_size, block_size},
                                           [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

            if (i < iend) {
              PRESSURE_BODY1
            }
          });
        });

        qu.submit([&] (cl::sycl::handler& h) {

          auto p_new = d_p_new.get_access<cl::sycl::access::mode::write>(h);
          auto bvc = d_bvc.get_access<cl::sycl::access::mode::read>(h);
          auto e_old = d_e_old.get_access<cl::sycl::access::mode::read>(h);
          auto vnewc = d_vnewc.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class PRESSURE_2>(cl::sycl::nd_range<1> {grid_size, block_size},
                                          [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

            if (i < iend) {
              PRESSURE_BODY2
            }
          });
        });
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    }

    PRESSURE_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  PRESSURE : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
