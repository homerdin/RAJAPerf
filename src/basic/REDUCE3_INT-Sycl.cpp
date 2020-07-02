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

#include "REDUCE3_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <CL/sycl.hpp>

#include <iostream>
#include "common/SyclDataUtils.hpp"

namespace rajaperf
{
namespace basic
{

const size_t block_size = 256;

#define REDUCE3_INT_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(vec, m_vec, iend, qu); \
    Int_ptr hsum; \
    allocAndInitSyclDeviceData(hsum, &m_vsum_init, 1, qu); \
    Int_ptr hmin; \
    allocAndInitSyclDeviceData(hmin, &m_vmin_init, 1, qu); \
    Int_ptr hmax; \
    allocAndInitSyclDeviceData(hmax, &m_vmax_init, 1, qu);

#define REDUCE3_INT_DATA_TEARDOWN_SYCL \
  deallocSyclDeviceData(vec, qu);

void REDUCE3_INT::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  REDUCE3_INT_DATA_SETUP;

  if ( vid == Base_SYCL ) {

    REDUCE3_INT_DATA_SETUP_SYCL;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initSyclDeviceData(hsum, &m_vsum_init, 1, qu);
      initSyclDeviceData(hmin, &m_vmin_init, 1, qu);
      initSyclDeviceData(hmax, &m_vmax_init, 1, qu);

      qu.submit( [&] (cl::sycl::handler& h) {

        auto vsum = cl::sycl::intel::detail::reducer(hsum, 0, cl::sycl::intel::plus<int>());
        auto vmin = cl::sycl::intel::detail::reducer(hmin, 0, cl::sycl::intel::minimum<int>());
        auto vmax = cl::sycl::intel::detail::reducer(hmax, 0, cl::sycl::intel::maximum<int>());

        const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

        h.parallel_for<class Reduce3>(cl::sycl::nd_range<1>(grid_size, block_size),
                                      [=] (cl::sycl::nd_item<1> item) {

          Index_type i = item.get_global_id(0);
          vsum += vec[i];
          vmin.combine(vec[i]);
          vmax.combine(vec[i]);            

        });
      });

      Int_type lsum;
      Int_ptr plsum = &lsum;
      getSyclDeviceData(plsum, hsum, 1, qu);
      m_vsum += lsum;

      Int_type lmin;
      Int_ptr plmin = &lmin;
      getSyclDeviceData(plmin, hmin, 1, qu);
      m_vmin = RAJA_MIN(m_vmin, lmin);

      Int_type lmax;
      Int_ptr plmax = &lmax;
      getSyclDeviceData(plmax, hmax, 1, qu);
      m_vmax = RAJA_MAX(m_vmax, lmax);

    }
    qu.wait(); // Wait for computation to finish before stopping timer
    stopTimer();

    REDUCE3_INT_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  REDUCE3_INT : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL

