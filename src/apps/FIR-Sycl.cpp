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

#include "FIR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <algorithm>
#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace apps
{

#define FIR_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_in {m_in, getRunSize()}; \
  cl::sycl::buffer<Real_type> d_out {m_out, getRunSize()}; \
  cl::sycl::buffer<Real_type> d_coeff {coeff_array, FIR_COEFFLEN}; \
\
  const Index_type coefflen = m_coefflen;

#define FIR_DATA_TEARDOWN_SYCL

void FIR::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize() - m_coefflen;

  if ( vid == Base_SYCL ) {
    {

      FIR_COEFF;

      FIR_DATA_SETUP_SYCL;
      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto in = d_in.get_access<cl::sycl::access::mode::read>(h);
          auto out = d_out.get_access<cl::sycl::access::mode::write>(h);
          auto coeff = d_coeff.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclFir>(cl::sycl::nd_range<1> {grid_size, block_size},
                                        [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

            if (i < iend) {
              FIR_BODY
            }
          });
        });
      }

      stopTimer();
    }

    FIR_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  FIR : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
