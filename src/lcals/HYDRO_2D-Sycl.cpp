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

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace lcals
{

#define HYDRO_2D_DATA_SETUP_SYCL \
  cl::sycl::buffer<Real_type> d_za { m_za, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zb { m_zb, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zm { m_zm, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zp { m_zp, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zq { m_zq, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zr { m_zr, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zu { m_zu, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zv { m_zv, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zz { m_zz, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zrout { m_zrout, m_array_length }; \
  cl::sycl::buffer<Real_type> d_zzout { m_zzout, m_array_length }; \
\
  const Real_type s = m_s; \
  const Real_type t = m_t; \
\
  const unsigned long jn = m_jn; \
  const unsigned long kn = m_kn;


#define HYDRO_2D_DATA_TEARDOWN_SYCL

void HYDRO_2D::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  if ( vid == Base_SYCL ) {
    {
      HYDRO_2D_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
        qu.submit([&] (cl::sycl::handler& h)
        { 
          auto zadat = d_za.get_access<cl::sycl::access::mode::write>(h);
          auto zbdat = d_zb.get_access<cl::sycl::access::mode::write>(h);
          auto zpdat = d_zp.get_access<cl::sycl::access::mode::read>(h);
          auto zqdat = d_zq.get_access<cl::sycl::access::mode::read>(h);
          auto zrdat = d_zr.get_access<cl::sycl::access::mode::read>(h);
          auto zmdat = d_zr.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclHydro2dBody1>(cl::sycl::range<2>{kn-2, jn-2},
                                                 cl::sycl::id<2>{1, 1}, // offset to start a idx 1
                                                 [=] (cl::sycl::item<2> item ) {
            int j = item.get_id(1);
            int k = item.get_id(0);
//            ++j; ++k;
 
            HYDRO_2D_BODY1
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        { 
          auto zudat = d_zu.get_access<cl::sycl::access::mode::write>(h);
          auto zvdat = d_zv.get_access<cl::sycl::access::mode::write>(h);
          auto zadat = d_za.get_access<cl::sycl::access::mode::read>(h);
          auto zzdat = d_zz.get_access<cl::sycl::access::mode::read>(h);
          auto zbdat = d_zb.get_access<cl::sycl::access::mode::read>(h);
          auto zrdat = d_zr.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclHydro2dBody2>(cl::sycl::range<2>{kn-2, jn-2},
                                                 cl::sycl::id<2>{1, 1}, // offset to start a idx 1
                                                 [=] (cl::sycl::item<2> item ) {
            int j = item.get_id(1);
            int k = item.get_id(0);
//            ++j; ++k;

            HYDRO_2D_BODY2
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        { 
          auto zroutdat = d_zrout.get_access<cl::sycl::access::mode::write>(h);
          auto zzoutdat = d_zzout.get_access<cl::sycl::access::mode::write>(h);
          auto zrdat = d_zr.get_access<cl::sycl::access::mode::read>(h);
          auto zudat = d_zu.get_access<cl::sycl::access::mode::read>(h);
          auto zzdat = d_zz.get_access<cl::sycl::access::mode::read>(h);
          auto zvdat = d_zv.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclHydro2dBody3>(cl::sycl::range<2>{kn-2, jn-2},
                                                 cl::sycl::id<2>{1, 1}, // offset to start a idx 1
                                                 [=] (cl::sycl::item<2> item ) {
            int j = item.get_id(1);
            int k = item.get_id(0);
//            ++j; ++k;

            HYDRO_2D_BODY3
          });
        });
      }

      stopTimer();

    }
    HYDRO_2D_DATA_TEARDOWN_SYCL;

  } else { 
     std::cout << "\n  HYDRO_2D : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
