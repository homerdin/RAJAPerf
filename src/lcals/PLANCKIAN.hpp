//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// PLANCKIAN kernel reference implementation:
///
/// for (Index_type i = ibegin; i < iend; ++i ) {
///   y[i] = u[i] / v[i];
///   w[i] = x[i] / ( exp( y[i] ) - 1.0 );
/// }
///

#ifndef RAJAPerf_Basic_PLANCKIAN_HPP
#define RAJAPerf_Basic_PLANCKIAN_HPP


#define PLANCKIAN_BODY  \
  y[i] = u[i] / v[i]; \
  w[i] = x[i] / ( exp( y[i] ) - 1.0 );


#include "common/KernelBase.hpp"

namespace rajaperf 
{
class RunParams;

namespace lcals
{

class PLANCKIAN : public KernelBase
{
public:

  PLANCKIAN(const RunParams& params);

  ~PLANCKIAN();

  void setUp(VariantID vid);
  void runKernel(VariantID vid); 
  void updateChecksum(VariantID vid);
  void tearDown(VariantID vid);

  void runCudaVariant(VariantID vid);
  void runOpenMPTargetVariant(VariantID vid);
  void runSyclVariant(VariantID vid);

private:
  Real_ptr m_x;
  Real_ptr m_y;
  Real_ptr m_u;
  Real_ptr m_v;
  Real_ptr m_w;
};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
