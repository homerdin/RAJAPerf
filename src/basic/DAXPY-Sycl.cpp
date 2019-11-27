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

//#include "common/SyclDataUtils.hpp"

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf
{
namespace basic
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;

#define DAXPY_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_x { m_x, getRunSize() }; \
  cl::sycl::buffer<Real_type> d_y { m_y, getRunSize() }; \
  Real_type a = m_a;

#define DAXPY_DATA_TEARDOWN_SYCL

void DAXPY::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const unsigned long iend = getRunSize();

  if ( vid == Base_SYCL ) {
    {
      DAXPY_DATA_SETUP_SYCL

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        {
          auto x = d_x.get_access<cl::sycl::access::mode::read>(h);
          auto y = d_y.get_access<cl::sycl::access::mode::read_write>(h);

          const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

          h.parallel_for<class syclDAXPY>(cl::sycl::nd_range<1>{grid_size, block_size},
                                          [=] (cl::sycl::nd_item<1> item ) {

            Index_type i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
            if (i < iend) {
              DAXPY_BODY
            }
          });
        });
      }
      stopTimer();
    } // Block to make sure host memory is written back

    DAXPY_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  DAXPY : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
