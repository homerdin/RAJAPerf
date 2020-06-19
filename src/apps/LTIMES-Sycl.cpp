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

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace apps
{

#define LTIMES_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_phidat {m_phidat, m_philen}; \
  cl::sycl::buffer<Real_type> d_elldat {m_elldat, m_elllen}; \
  cl::sycl::buffer<Real_type> d_psidat {m_psidat, m_psilen}; \
\
  unsigned long num_d = m_num_d; \
  unsigned long num_z = m_num_z; \
  unsigned long num_g = m_num_g; \
  unsigned long num_m = m_num_m; \
\
  force_allocate<cl::sycl::access::target::global_buffer>(d_phidat, qu); \
  force_allocate<cl::sycl::access::target::global_buffer>(d_elldat, qu); \
  force_allocate<cl::sycl::access::target::global_buffer>(d_psidat, qu);

#define LTIMES_DATA_TEARDOWN_SYCL

void LTIMES::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_SYCL ) {
    {
      LTIMES_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu.submit([&] (cl::sycl::handler& h) {

          auto phidat = d_phidat.get_access<cl::sycl::access::mode::read_write>(h);
          auto elldat = d_elldat.get_access<cl::sycl::access::mode::read>(h);
          auto psidat = d_psidat.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class LTIMES>(cl::sycl::range<3> {num_z, num_g, num_m},
                                       [=] (cl::sycl::item<3> item) {

            Index_type z = item.get_id(0);
            Index_type g = item.get_id(1);
            Index_type m = item.get_id(2);

            for (Index_type d = 0; d < num_d; ++d) {
              LTIMES_BODY
            }
          });
        });
      }
      qu.wait(); // Wait for computation to finish before stopping timer      
      stopTimer();
    }

    LTIMES_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n LTIMES : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
