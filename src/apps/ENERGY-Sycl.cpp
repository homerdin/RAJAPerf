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

#include "ENERGY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>

namespace rajaperf 
{
namespace apps
{

#define ENERGY_DATA_SETUP_SYCL \
  const size_t block_size = qu.get_device().get_info<cl::sycl::info::device::max_work_group_size>(); \
\
  cl::sycl::buffer<Real_type> d_e_new {m_e_new, iend}; \
  cl::sycl::buffer<Real_type> d_e_old {m_e_old, iend}; \
  cl::sycl::buffer<Real_type> d_delvc {m_delvc, iend}; \
  cl::sycl::buffer<Real_type> d_p_new {m_p_new, iend}; \
  cl::sycl::buffer<Real_type> d_p_old {m_p_old, iend}; \
  cl::sycl::buffer<Real_type> d_q_new {m_q_new, iend}; \
  cl::sycl::buffer<Real_type> d_q_old {m_q_old, iend}; \
  cl::sycl::buffer<Real_type> d_work {m_work, iend}; \
  cl::sycl::buffer<Real_type> d_compHalfStep {m_compHalfStep, iend}; \
  cl::sycl::buffer<Real_type> d_pHalfStep {m_pHalfStep, iend}; \
  cl::sycl::buffer<Real_type> d_bvc {m_bvc, iend}; \
  cl::sycl::buffer<Real_type> d_pbvc {m_pbvc, iend}; \
  cl::sycl::buffer<Real_type> d_ql_old {m_ql_old, iend}; \
  cl::sycl::buffer<Real_type> d_qq_old {m_qq_old, iend}; \
  cl::sycl::buffer<Real_type> d_vnewc {m_vnewc, iend}; \
  const Real_type rho0 = m_rho0; \
  const Real_type e_cut = m_e_cut; \
  const Real_type emin = m_emin; \
  const Real_type q_cut = m_q_cut; \

#define ENERGY_DATA_TEARDOWN_SYCL

void ENERGY::runSyclVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const unsigned int ibegin = 0;
  const unsigned int iend = getRunSize();

  if ( vid == Base_SYCL ) {
    {
      ENERGY_DATA_SETUP_SYCL;

      const size_t grid_size = block_size * RAJA_DIVIDE_CEILING_INT(iend, block_size);

      using cl::sycl::sqrt;
      using cl::sycl::fabs;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto e_new = d_e_new.get_access<cl::sycl::access::mode::write>(h);
          auto e_old = d_e_old.get_access<cl::sycl::access::mode::read>(h);
          auto delvc = d_delvc.get_access<cl::sycl::access::mode::read>(h);
          auto p_old = d_p_old.get_access<cl::sycl::access::mode::read>(h);
          auto q_old = d_q_old.get_access<cl::sycl::access::mode::read>(h);
          auto work = d_work.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclEnergy_1>(cl::sycl::nd_range<1> {grid_size, block_size},
                                             [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

            if(i < iend) {
              ENERGY_BODY1
            }
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto delvc = d_delvc.get_access<cl::sycl::access::mode::read>(h);
          auto q_new = d_q_new.get_access<cl::sycl::access::mode::write>(h);
          auto compHalfStep = d_compHalfStep.get_access<cl::sycl::access::mode::read>(h);
          auto pbvc = d_pbvc.get_access<cl::sycl::access::mode::read>(h);
          auto e_new = d_e_new.get_access<cl::sycl::access::mode::read>(h);
          auto bvc = d_bvc.get_access<cl::sycl::access::mode::read>(h);
          auto pHalfStep = d_pHalfStep.get_access<cl::sycl::access::mode::read>(h);
          auto ql_old = d_ql_old.get_access<cl::sycl::access::mode::read>(h);
          auto qq_old = d_qq_old.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclEnergy_2>(cl::sycl::nd_range<1> {grid_size, block_size},
                                             [=] (cl::sycl::nd_item<1> item) {
            
            Index_type i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);
            
            if(i < iend) {
              ENERGY_BODY2
            }
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto e_new = d_e_new.get_access<cl::sycl::access::mode::read_write>(h);
          auto delvc = d_delvc.get_access<cl::sycl::access::mode::read>(h);
          auto p_old = d_p_old.get_access<cl::sycl::access::mode::read>(h);
          auto q_old = d_q_old.get_access<cl::sycl::access::mode::read>(h);
          auto pHalfStep = d_pHalfStep.get_access<cl::sycl::access::mode::read>(h);
          auto q_new = d_q_new.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclEnergy_3>(cl::sycl::nd_range<1> {grid_size, block_size},
                                             [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

            if(i < iend) {
              ENERGY_BODY3
            }
          });
        });


        qu.submit([&] (cl::sycl::handler& h)
        {
          auto e_new = d_e_new.get_access<cl::sycl::access::mode::read_write>(h);
          auto work = d_work.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclEnergy_4>(cl::sycl::nd_range<1> {grid_size, block_size},
                                             [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

            if(i < iend) {
              ENERGY_BODY4
            }
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto delvc = d_delvc.get_access<cl::sycl::access::mode::read>(h);
          auto pbvc = d_pbvc.get_access<cl::sycl::access::mode::read>(h);
          auto e_new = d_e_new.get_access<cl::sycl::access::mode::read_write>(h);
          auto vnewc = d_vnewc.get_access<cl::sycl::access::mode::read>(h);
          auto bvc = d_bvc.get_access<cl::sycl::access::mode::read>(h);
          auto p_new = d_p_new.get_access<cl::sycl::access::mode::read>(h);
          auto ql_old = d_ql_old.get_access<cl::sycl::access::mode::read>(h);
          auto qq_old = d_qq_old.get_access<cl::sycl::access::mode::read>(h);
          auto p_old = d_p_old.get_access<cl::sycl::access::mode::read>(h);
          auto q_old = d_q_old.get_access<cl::sycl::access::mode::read>(h);
          auto pHalfStep = d_pHalfStep.get_access<cl::sycl::access::mode::read>(h);
          auto q_new = d_q_new.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclEnergy_5>(cl::sycl::nd_range<1> {grid_size, block_size},
                                             [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

            if(i < iend) {
              ENERGY_BODY5
            }
          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {
          auto delvc = d_delvc.get_access<cl::sycl::access::mode::read>(h);
          auto pbvc = d_pbvc.get_access<cl::sycl::access::mode::read>(h);
          auto e_new = d_e_new.get_access<cl::sycl::access::mode::read>(h);
          auto vnewc = d_vnewc.get_access<cl::sycl::access::mode::read>(h);
          auto bvc = d_bvc.get_access<cl::sycl::access::mode::read>(h);
          auto p_new = d_p_new.get_access<cl::sycl::access::mode::read>(h);
          auto q_new = d_p_new.get_access<cl::sycl::access::mode::read_write>(h);
          auto ql_old = d_ql_old.get_access<cl::sycl::access::mode::read>(h);
          auto qq_old = d_qq_old.get_access<cl::sycl::access::mode::read>(h);

          h.parallel_for<class syclEnergy_6>(cl::sycl::nd_range<1> {grid_size, block_size},
                                             [=] (cl::sycl::nd_item<1> item) {

            Index_type i = item.get_group(0) * item.get_local_range().get(0) + item.get_local_id(0);

            if(i < iend) {
              ENERGY_BODY6
            }
          });
        });
      }

      stopTimer();
    } // Buffer Destruction

    ENERGY_DATA_TEARDOWN_SYCL;

  } else {
     std::cout << "\n  ENERGY : Unknown Sycl variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
