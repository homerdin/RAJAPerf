//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// TRIAD kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   a[i] = b[i] + alpha * c[i] ;
/// }
///

#ifndef RAJAPerf_Stream_TRIAD_HPP
#define RAJAPerf_Stream_TRIAD_HPP


#define TRIAD_BODY  \
  a[i] = b[i] + alpha * c[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class TRIAD : public KernelBase
{
public:

  TRIAD(const RunParams& params);

  ~TRIAD();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runSyclVariant(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_b;
  Real_ptr m_c;
  Real_type m_alpha;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
