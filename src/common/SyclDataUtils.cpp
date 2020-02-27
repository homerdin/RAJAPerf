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



#include "RPTypes.hpp"
#include "common/SyclDataUtils.hpp"

#if defined(RAJA_ENABLE_SYCL)

namespace rajaperf
{

void force_memcpy_index(cl::sycl::buffer<Index_type, 1> buf, cl::sycl::queue q) {

  q.submit([&] (cl::sycl::handler &h) {
    sycl::accessor<Index_type, 1, cl::sycl::access::mode::read_write> acc(
        buf, h, buf.get_size());
    h.single_task<class forceMemcpy_Index_t>([=]() {acc[0];});
  });

  q.wait();
}

void force_memcpy_real(cl::sycl::buffer<Real_type, 1> buf, cl::sycl::queue q) {

  q.submit([&] (cl::sycl::handler &h) {
    sycl::accessor<Real_type, 1, cl::sycl::access::mode::read_write> acc(
        buf, h, buf.get_size());
    h.single_task<class forceMemcpy_Real_t>([=]() {acc[0];});
  });

  q.wait();

}

void force_memcpy_int(cl::sycl::buffer<Int_type, 1> buf, cl::sycl::queue q) {

  q.submit([&] (cl::sycl::handler &h) {
    sycl::accessor<Int_type, 1, cl::sycl::access::mode::read_write> acc(
        buf, h, buf.get_size());
    h.single_task<class forceMemcpy_Int_t>([=]() {acc[0];});
  });

  q.wait();
}

}  // closing brace for rajaperf namespace

#endif // RAJA_ENABLE_SYCL
