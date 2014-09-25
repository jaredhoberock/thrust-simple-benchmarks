#pragma once

#include <thrust/execution_policy.h>


template<class Function>
__global__ void apply_kernel(Function f)
{
  f();
}


template<class Function>
void device_apply(Function f)
{
  apply_kernel<<<1,1>>>(f);
}


__global__ void check_device_thrust_enabled(bool* result)
{
  *result = __BULK_HAS_CUDART__;
}


bool device_thrust_enabled()
{
  thrust::device_vector<bool> result(1, false);

  check_device_thrust_enabled<<<1,1>>>(raw_pointer_cast(result.data()));

  return result[0];
}


template<class Function>
double time_function(Function f, size_t num_trials = 100)
{
  if(device_thrust_enabled())
  {
    // warm up
    for(int i = 0; i < num_trials; ++i)
    {
      device_apply(f);
    }

    return time_invocation_cuda(num_trials, device_apply<Function>, f);
  }

  // warm up
  for(int i = 0; i < num_trials; ++i)
  {
    f();
  }

  return time_invocation_cuda(num_trials, f);
}

