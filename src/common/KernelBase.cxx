/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for kernel base class.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-xxxxxx
//
// All rights reserved.
//
// This file is part of the RAJA Performance Suite.
//
// For additional details, please read the file LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


#include "KernelBase.hxx"

#include "RunParams.hxx"

namespace rajaperf {

KernelBase::KernelBase(KernelID kid, const RunParams& params) 
  : kernel_id(kid),
    name( getKernelName(kernel_id) ),
    run_size(0),
    run_samples(0),
    default_size(0),
    default_samples(0),
    run_params(params)
{
  for (size_t ivar = 0; ivar < NumVariants; ++ivar) {
     min_time[ivar] = 0.0;
     max_time[ivar] = 0.0;
     tot_time[ivar] = 0.0;
     checksum[ivar] = 0.0;
  }
}

 
KernelBase::~KernelBase()
{
}


void KernelBase::setDefaultSize(int size)
{
  default_size = size;
  run_size = static_cast<int>( size*run_params.getSizeFraction() );
}

void KernelBase::setDefaultSamples(int nsamp)
{
  default_samples = nsamp;
  run_samples = static_cast<int>( nsamp*run_params.getSampleFraction() );
}


void KernelBase::execute(VariantID vid) 
{
  this->setUp(vid);
  
  this->runKernel(vid); 

  this->computeChecksum(vid); 

  this->tearDown(vid);
}


void KernelBase::startTimer()
{
  //start_time = ...;
}

void KernelBase::stopTimer()
{
  //stop_time = ...;
  recordExecTime();
}

void KernelBase::recordExecTime()
{
#if 0 // RDH
  Duration time = stop_time - start_time;

  min_time = std::min(min_time, time.count());
  max_time = std::max(min_time, time.count());
  tot_time += time.count();
#endif
}

}  // closing brace for rajaperf namespace