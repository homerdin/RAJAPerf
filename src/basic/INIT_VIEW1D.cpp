//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


#define INIT_VIEW1D_DATA_SETUP_CPU \
  Real_ptr a = m_a; \
  const Real_type v = m_val; \
\
  using ViewType = RAJA::View<Real_type, RAJA::Layout<1, Index_type, 0> >; \
  const RAJA::Layout<1> my_layout(iend); \
  ViewType view(a, my_layout);


INIT_VIEW1D::INIT_VIEW1D(const RunParams& params)
  : KernelBase(rajaperf::Basic_INIT_VIEW1D, params)
{
   setDefaultSize(500000);
   setDefaultReps(5000);
}

INIT_VIEW1D::~INIT_VIEW1D() 
{
}

void INIT_VIEW1D::setUp(VariantID vid)
{
  allocAndInitDataConst(m_a, getRunSize(), 0.0, vid);
  m_val = 0.123;
}

void INIT_VIEW1D::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  INIT_VIEW1D_DATA_SETUP_CPU;

  auto initview1d_lam = [=](Index_type i) {
                          INIT_VIEW1D_BODY_RAJA;
                        };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_BODY;
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)     
    case RAJA_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::simd_exec>(
          RAJA::RangeSegment(ibegin, iend), initview1d_lam);

      }
      stopTimer();

      break;
    }
#endif // RUN_RAJA_SEQ

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)                        
    case Base_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend),  initview1d_lam);

      }
      stopTimer();

      break;
    }
#endif
                       
#if defined(RAJA_ENABLE_TARGET_OPENMP)
    case Base_OpenMPTarget :
    case RAJA_OpenMPTarget :
    {
      runOpenMPTargetVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_CUDA)
    case Base_CUDA :
    case RAJA_CUDA :
    {
      runCudaVariant(vid);
      break;
    }
#endif

#if defined(RAJA_ENABLE_SYCL)
    case Base_SYCL :
    {
      runSyclVariant(vid);
      break;
    }
#endif

    default : {
      std::cout << "\n  INIT_VIEW1D : Unknown variant id = " << vid << std::endl;
    }

  }

}

void INIT_VIEW1D::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunSize());
}

void INIT_VIEW1D::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
}

} // end namespace basic
} // end namespace rajaperf
