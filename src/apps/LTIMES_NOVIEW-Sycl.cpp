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

#include "LTIMES_NOVIEW.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

#define LTIMES_NOVIEW_DATA_SETUP_SYCL \
\
  allocAndInitSyclDeviceData(phidat, m_phidat, m_philen, qu); \
  allocAndInitSyclDeviceData(elldat, m_elldat, m_elllen, qu); \
  allocAndInitSyclDeviceData(psidat, m_psidat, m_psilen, qu);

#define LTIMES_NOVIEW_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_phidat, phidat, m_philen, qu); \
  deallocSyclDeviceData(phidat, qu); \
  deallocSyclDeviceData(elldat, qu); \
  deallocSyclDeviceData(psidat, qu);

void LTIMES_NOVIEW::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_NOVIEW_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    LTIMES_NOVIEW_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class LTIMES_NOVIEW>(cl::sycl::range<3>(num_m, num_g, num_z),
                                            [=] (cl::sycl::item<3> item) {

          Index_type m = item.get_id(0);
          Index_type g = item.get_id(1);
          Index_type z = item.get_id(2);

          for (Index_type d = 0; d < num_d; ++d) {
            LTIMES_NOVIEW_BODY
          }

        });
      });
    }
    qu.wait(); // Wait for computation to finish before stopping timer      
    stopTimer();

    LTIMES_NOVIEW_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n LTIMES_NOVIEW : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
