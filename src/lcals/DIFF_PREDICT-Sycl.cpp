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

#include "DIFF_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace lcals
{

#define DIFF_PREDICT_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(px, m_px, m_array_length, qu); \
  allocAndInitSyclDeviceData(cx, m_cx, m_array_length, qu);

#define DIFF_PREDICT_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_px, px, m_array_length, qu); \
  deallocSyclDeviceData(px, qu); \
  deallocSyclDeviceData(cx, qu);

void DIFF_PREDICT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  DIFF_PREDICT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    DIFF_PREDICT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class DiffPredict>(cl::sycl::range<1>(iend),
                                          [=] (cl::sycl::item<1> item) {

          Index_type i = item.get_id(0);
          DIFF_PREDICT_BODY

        });
      });
    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    DIFF_PREDICT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  DIFF_PREDICT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
