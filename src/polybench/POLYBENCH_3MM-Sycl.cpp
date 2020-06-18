  
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

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <iostream>

#include <CL/sycl.hpp>
#include "common/SyclDataUtils.hpp"

namespace rajaperf 
{
namespace polybench
{

#define POLYBENCH_3MM_DATA_SETUP_SYCL \
  allocAndInitSyclDeviceData(A, m_A, m_ni * m_nk, qu); \
  allocAndInitSyclDeviceData(B, m_B, m_nk * m_nj, qu); \
  allocAndInitSyclDeviceData(C, m_C, m_nj * m_nm, qu); \
  allocAndInitSyclDeviceData(D, m_D, m_nm * m_nl, qu); \
  allocAndInitSyclDeviceData(E, m_E, m_ni * m_nj, qu); \
  allocAndInitSyclDeviceData(F, m_F, m_nj * m_nl, qu); \
  allocAndInitSyclDeviceData(G, m_G, m_ni * m_nl, qu); 

#define POLYBENCH_3MM_TEARDOWN_SYCL \
  getsyclDeviceData(m_G, G, m_ni * m_nl, qu); \
  deallocSyclDeviceData(A, qu); \
  deallocSyclDeviceData(B, qu); \
  deallocSyclDeviceData(C, qu); \
  deallocSyclDeviceData(D, qu); \
  deallocSyclDeviceData(E, qu); \
  deallocSyclDeviceData(F, qu); \
  deallocSyclDeviceData(G, qu);

void POLYBENCH_3MM::runSyclVariant(VariantID vid)
{
  const unsigned long run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;
 
  if ( vid == Base_SYCL ) {
    {
      POLYBENCH_3MM_DATA_SETUP_SYCL;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        qu.submit([&] (cl::sycl::handler& h)
        {

          h.parallel_for<class polybench3MM_1>(cl::sycl::range<2> {ni, nj},
                                               [=] (cl::sycl::item<2> item) {
            Index_type i = item.get_id(0);
            Index_type j = item.get_id(1);

            POLYBENCH_3MM_BODY1;
            for (Index_type k=0; k < nk; ++k) {
              POLYBENCH_3MM_BODY2;
            }
            POLYBENCH_3MM_BODY3;

          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {

          h.parallel_for<class polybench3MM_2>(cl::sycl::range<2> {nj, nl},
                                               [=] (cl::sycl::item<2> item) {
            Index_type j = item.get_id(0);
            Index_type l = item.get_id(1);

            POLYBENCH_3MM_BODY4;
            for (Index_type m=0; m < nm; ++m) {
              POLYBENCH_3MM_BODY5;
            }
            POLYBENCH_3MM_BODY6;

          });
        });

        qu.submit([&] (cl::sycl::handler& h)
        {

          h.parallel_for<class polybench3MM_3>(cl::sycl::range<2> {ni, nl},
                                               [=] (cl::sycl::item<2> item) {
            Index_type i = item.get_id(0);
            Index_type l = item.get_id(1);

            POLYBENCH_3MM_BODY7;
            for (Index_type j=0; j < nj; ++j) {
              POLYBENCH_3MM_BODY8;
            }
            POLYBENCH_3MM_BODY9;

          });
        });
      }
      qu.wait(); // Wait for computation to finish before stopping timer
      stopTimer();
    }

    POLYBENCH_3MM_TEARDOWN_SYCL;

  } else if (vid == RAJA_SYCL) {

    POLYBENCH_3MM_DATA_SETUP_SYCL;

    POLYBENCH_3MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::SyclKernel<
          RAJA::statement::For<0, RAJA::sycl_global_1<1>,
            RAJA::statement::For<1, RAJA::sycl_global_2<256>,
              RAJA::statement::Lambda<0>,
              RAJA::statement::For<2, RAJA::seq_exec,
                RAJA::statement::Lambda<1>
              >,
              RAJA::statement::Lambda<2>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nk}),
        RAJA::make_tuple(static_cast<Real_type>(0.0)),

        [=] (Index_type /*i*/, Index_type /*j*/, Index_type /*k*/,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY1_RAJA;
        },
        [=] (Index_type i, Index_type j, Index_type k,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY2_RAJA;
        },
        [=] (Index_type i, Index_type j, Index_type /*k*/,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY3_RAJA;
        }

      );
      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nm}),
        RAJA::make_tuple(static_cast<Real_type>(0.0)),

        [=]  (Index_type /*j*/, Index_type /*l*/, Index_type /*m*/,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY4_RAJA;
        },
        [=] (Index_type j, Index_type l, Index_type m,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY5_RAJA;
        },
        [=]  (Index_type j, Index_type l, Index_type /*m*/,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY6_RAJA;
        }

      );

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nj}),
        RAJA::make_tuple(static_cast<Real_type>(0.0)),

        [=] (Index_type /*i*/, Index_type /*l*/, Index_type /*j*/,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY7_RAJA;
        },
        [=] (Index_type i, Index_type l, Index_type j,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY8_RAJA;
        },
        [=] (Index_type i, Index_type l, Index_type /*j*/,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY9_RAJA;
        }

      );

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_SYCL;


  } else {
      std::cout << "\n  POLYBENCH_3MM : Unknown Sycl variant id = " << vid << std::endl;
  }

}

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_SYCL
  
