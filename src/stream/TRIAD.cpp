//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD.hpp"

#include "RAJA/RAJA.hpp"

#include "common/DataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace stream
{


#define TRIAD_DATA_SETUP_CPU \
  ResReal_ptr a = m_a; \
  ResReal_ptr b = m_b; \
  ResReal_ptr c = m_c; \
  Real_type alpha = m_alpha;


TRIAD::TRIAD(const RunParams& params)
  : KernelBase(rajaperf::Stream_TRIAD, params)
{
   setDefaultSize(1000000);
   setDefaultReps(1000);
}

TRIAD::~TRIAD() 
{
}

void TRIAD::setUp(VariantID vid)
{
  allocAndInitDataConst(m_a, getRunSize(), 0.0, vid);
  allocAndInitData(m_b, getRunSize(), vid);
  allocAndInitData(m_c, getRunSize(), vid);
  initData(m_alpha, vid);
}

void TRIAD::runKernel(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  TRIAD_DATA_SETUP_CPU;

  auto triad_lam = [=](Index_type i) {
                     TRIAD_BODY;
                   };

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        for (Index_type i = ibegin; i < iend; ++i ) {
          TRIAD_BODY;
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
          RAJA::RangeSegment(ibegin, iend), triad_lam);

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
          TRIAD_BODY;
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::forall<RAJA::omp_parallel_for_exec>(
          RAJA::RangeSegment(ibegin, iend), triad_lam);

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
      std::cout << "\n  TRIAD : Unknown variant id = " << vid << std::endl;
    }

  }

}

void TRIAD::updateChecksum(VariantID vid)
{
  checksum[vid] += calcChecksum(m_a, getRunSize());
}

void TRIAD::tearDown(VariantID vid)
{
  (void) vid;
  deallocData(m_a);
  deallocData(m_b);
  deallocData(m_c);
}

} // end namespace stream
} // end namespace rajaperf
