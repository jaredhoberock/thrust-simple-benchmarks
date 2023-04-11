#pragma once

template<class Function>
double time_function(Function f, size_t num_trials = 100)
{
  // warm up
  for(int i = 0; i < num_trials; ++i)
  {
    f();
  }

  return time_invocation_cuda(num_trials, f);
}

