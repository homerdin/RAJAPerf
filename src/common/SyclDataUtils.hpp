//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-19, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Method to trigger data movement for SYCL 
///


#ifndef RAJAPerf_SyclDataUtils_HPP
#define RAJAPerf_SyclDataUtils_HPP

#include "RPTypes.hpp"

#if defined(RAJA_ENABLE_SYCL)

#include <CL/sycl.hpp>

namespace rajaperf
{

void force_memcpy_index(cl::sycl::buffer<Index_type, 1> buf, cl::sycl::queue q);
void force_memcpy_real(cl::sycl::buffer<Real_type, 1> buf, cl::sycl::queue q);
void force_memcpy_int(cl::sycl::buffer<Int_type, 1> buf, cl::sycl::queue q);

}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_SYCL

#endif  // closing endif for header file include guard

