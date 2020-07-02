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

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

#define INIT_VIEW1D_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(a, m_a, iend, qu);

#define INIT_VIEW1D_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_a, a, iend, qu); \
  deallocSyclDeviceData(a, qu);

void INIT_VIEW1D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INIT_VIEW1D_DATA_SETUP;

  if ( vid == Base_SYCL ) {
    INIT_VIEW1D_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class Init3_view1d>(cl::sycl::range<1>(iend),
                                          [=] (cl::sycl::item<1> item) {

          Index_type i = item.get_id(0);
          INIT_VIEW1D_BODY

        });
      });
    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  INIT_VIEW1D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
