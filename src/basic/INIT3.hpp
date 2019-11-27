//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// INIT3 kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;
/// }
///

#ifndef RAJAPerf_Basic_INIT3_HPP
#define RAJAPerf_Basic_INIT3_HPP


#define INIT3_BODY  \
  out1[i] = out2[i] = out3[i] = - in1[i] - in2[i] ;


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace basic
{

class INIT3 : public KernelBase
{
public:

  INIT3(const RunParams& params);

  ~INIT3();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runSyclVariant(VariantID vid);

private:
  Real_ptr m_out1;
  Real_ptr m_out2;
  Real_ptr m_out3;
  Real_ptr m_in1;
  Real_ptr m_in2; 
};

} // end namespace basic
} // end namespace rajaperf

#endif // closing endif for header file include guard
