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

#include "NESTED_INIT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

#define NESTED_INIT_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(array, m_array, m_array_length, qu);

#define NESTED_INIT_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_array, array, m_array_length, qu); \
  deallocSyclDeviceData(array, qu);

void NESTED_INIT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  NESTED_INIT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    NESTED_INIT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class NestedInit>(cl::sycl::range<3> (nk, nj, ni),
                                         [=] (cl::sycl::item<3> item) {

          Index_type i = item.get_id(2);
          Index_type j = item.get_id(1);
          Index_type k = item.get_id(0);

          NESTED_INIT_BODY

        });
      });
    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    NESTED_INIT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  NESTED_INIT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
