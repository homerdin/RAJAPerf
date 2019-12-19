//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// COPY kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   c[i] = a[i] ;
/// }
///

#ifndef RAJAPerf_Stream_COPY_HPP
#define RAJAPerf_Stream_COPY_HPP


#define COPY_BODY  \
  c[i] = a[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace stream
{

class COPY : public KernelBase
{
public:

  COPY(const RunParams& params);

  ~COPY();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runSyclVariant(VariantID vid);

private:
  Real_ptr m_a;
  Real_ptr m_c;
};

} // end namespace stream
} // end namespace rajaperf

#endif // closing endif for header file include guard
