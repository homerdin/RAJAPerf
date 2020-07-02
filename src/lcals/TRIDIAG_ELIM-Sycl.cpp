//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include "common/SyclDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace lcals
{

#define TRIDIAG_ELIM_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(xout, m_xout, m_N, qu); \
  allocAndInitSyclDeviceData(xin, m_xin, m_N, qu); \
  allocAndInitSyclDeviceData(y, m_y, m_N, qu); \
  allocAndInitSyclDeviceData(z, m_z, m_N, qu);

#define TRIDIAG_ELIM_DATA_TEARDOWN_SYCL \
  getSyclDeviceData(m_xout, xout, m_N, qu); \
  deallocSyclDeviceData(xout, qu); \
  deallocSyclDeviceData(xin, qu); \
  deallocSyclDeviceData(y, qu); \
  deallocSyclDeviceData(z, qu);

void TRIDIAG_ELIM::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    TRIDIAG_ELIM_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      qu.submit([&] (cl::sycl::handler& h) {
        h.parallel_for<class TridiagElim>(cl::sycl::range<1>(iend),
                                          cl::sycl::id<1> (ibegin),
                                          [=] (cl::sycl::item<1> i) {

          TRIDIAG_ELIM_BODY;

        });
      });
    }
    qu.wait();
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  TRIDIAG_ELIM : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
