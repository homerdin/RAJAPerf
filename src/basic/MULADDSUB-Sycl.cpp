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

#include "MULADDSUB.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

#define MULADDSUB_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  Real_ptr out1; \
  Real_ptr out2; \
  Real_ptr out3; \
  Real_ptr in1; \
  Real_ptr in2; \
\
  allocAndInitSyclDeviceData(out1, m_out1, iend, qu); \
  allocAndInitSyclDeviceData(out2, m_out2, iend, qu); \
  allocAndInitSyclDeviceData(out3, m_out3, iend, qu); \
  allocAndInitSyclDeviceData(in1, m_in1, iend, qu); \
  allocAndInitSyclDeviceData(in2, m_in2, iend, qu);

#define MULADDSUB_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_out1, out1, iend, qu); \
  getSyclDeviceData(m_out2, out2, iend, qu); \
  getSyclDeviceData(m_out3, out3, iend, qu); \
  deallocSyclDeviceData(out1, qu); \
  deallocSyclDeviceData(out2, qu); \
  deallocSyclDeviceData(out3, qu); \
  deallocSyclDeviceData(in1, qu); \
  deallocSyclDeviceData(in2, qu);


void MULADDSUB::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  if ( vid == Base_SYCL ) {
    MULADDSUB_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
      qu.submit([&] (cl::sycl::handler& h)
      {
        const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

        h.parallel_for(cl::sycl::nd_range<1>{grid_size, block_size},
                       [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          if (i < iend) {
            MULADDSUB_BODY
          }
        });
      });
    }
    qu.wait(); // Wait for computation to finish before stoping timer
    stopTimer();

    MULADDSUB_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  MULADDSUB : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
