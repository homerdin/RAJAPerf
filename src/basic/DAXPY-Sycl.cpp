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

#include "DAXPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for SYCL execution
  //
  const size_t block_size = 256;

#define DAXPY_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(x, m_x, iend, qu); \
  allocAndInitSyclDeviceData(y, m_y, iend, qu);

#define DAXPY_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_y, y, iend, qu); \
  deallocSyclDeviceData(x, qu); \
  deallocSyclDeviceData(y, qu);


void DAXPY::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  DAXPY_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    #include "DAXPY_Setup.hpp"

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #include "DAXPY_Kernel.hpp"

    }
    qu.wait(); // Wait for computation to finish before stopping timer

    stopTimer();

    #include "DAXPY_Teardown.hpp"
    
  } else if ( vid == RAJA_SYCL ) {

    #include "DAXPY_Setup.hpp"

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      #include "DAXPY_RAJA_Kernel.hpp"

    }
    qu.wait();
    stopTimer();

    #include "DAXPY_Teardown.hpp"

  } else {
     std::cout << "\n  DAXPY : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
