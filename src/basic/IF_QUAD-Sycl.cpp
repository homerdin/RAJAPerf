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

#include "IF_QUAD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

#define IF_QUAD_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  Real_ptr a; \
  Real_ptr b; \
  Real_ptr c; \
  Real_ptr x1; \
  Real_ptr x2; \
\
  allocAndInitSyclDeviceData(a, m_a, iend, qu); \
  allocAndInitSyclDeviceData(b, m_b, iend, qu); \
  allocAndInitSyclDeviceData(c, m_c, iend, qu); \
  allocAndInitSyclDeviceData(x1, m_x1, iend, qu); \
  allocAndInitSyclDeviceData(x2, m_x2, iend, qu);

#define IF_QUAD_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_x1, x1, iend, qu); \
  getSyclDeviceData(m_x2, x2, iend, qu); \
  deallocSyclDeviceData(a, qu); \
  deallocSyclDeviceData(b, qu); \
  deallocSyclDeviceData(c, qu); \
  deallocSyclDeviceData(x1, qu); \
  deallocSyclDeviceData(x2, qu);


void IF_QUAD::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  if ( vid == Base_SYCL ) {
    IF_QUAD_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      qu.submit([&] (cl::sycl::handler& h)
      {
        const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

        h.parallel_for(cl::sycl::nd_range<1>(cl::sycl::range<1>(grid_size),
                                             cl::sycl::range<1>(block_size)),
                       [=] (cl::sycl::nd_item<1> item ) {

          Index_type i = item.get_global_id(0);

          if (i < iend) {
            using cl::sycl::sqrt;
            IF_QUAD_BODY
          }
        });
      });
    }
    qu.wait(); // Wait for computation to finish before stoping timer
    stopTimer();

    IF_QUAD_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  IF_QUAD : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
