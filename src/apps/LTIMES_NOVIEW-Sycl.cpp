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

namespace rajaperf 
{
namespace apps
{


#define LTIMES_NOVIEW_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_phidat {m_phidat, m_philen}; \
  cl::sycl::buffer<Real_type> d_elldat {m_elldat, m_elllen}; \
  cl::sycl::buffer<Real_type> d_psidat {m_psidat, m_psilen}; \
\
  unsigned long num_d = m_num_d; \
  unsigned long num_z = m_num_z; \
  unsigned long num_g = m_num_g; \
  unsigned long num_m = m_num_m; \

#define LTIMES_NOVIEW_DATA_TEARDOWN_SYCL

void LTIMES_NOVIEW::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  if ( vid == Base_SYCL ) {
    {
      LTIMES_NOVIEW_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu.submit([&] (cl::sycl::handler& h) {

          auto phidat = d_phidat.get_access<cl::sycl::access::mode::read_write>(h);
          auto elldat = d_elldat.get_access<cl::sycl::access::mode::read>(h);
          auto psidat = d_psidat.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class LTIMES_NOVIEW>(cl::sycl::range<3> {num_z, num_g, num_m},
                                             [=] (cl::sycl::item<3> item) {

            Index_type z = item.get_id(0);
            Index_type g = item.get_id(1);
            Index_type m = item.get_id(2);

            for (Index_type d = 0; d < num_d; ++d) {
              LTIMES_NOVIEW_BODY
            }
          });
        });
      }
      
      stopTimer();
    }

    LTIMES_NOVIEW_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n LTIMES_NOVIEW : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
