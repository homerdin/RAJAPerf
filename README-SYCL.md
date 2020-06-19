# SYCL

This branch contains a WIP port of RAJAPerf to SYCL using the Intel
USM extension.

It is built against the branch of RAJA at:
https://github.com/homerdin/RAJA dpcpp branch
and CAMP at:
https://github.com/homerdin/camp sycl branch

The Polybench group has been disabled temporarily due to an issue.

Not all kernels have been ported, and some ported kernels use
different execution models.  This will be corrected to properly
measure overhead of the RAJA abstraction.

Some of the changes in RAJA and CAMP are to allow for trivially
copyable classes/structs in the RAJA-SYCL kernels.  Views are 
still not fully trivially copyable (INIT_VIEW1d, etc disabled)

There is a build script for the Intel SYCL compiler at
scripts/alcf-builds/sycl.sh
