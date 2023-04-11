#pragma once

#include <concepts>
#include <limits>
#include <random>
#include <thrust/device_vector.h>
#include <vector>

template<std::integral T>
thrust::device_vector<T> random_device_vector(std::size_t n)
{
  std::vector<T> h_vec(n);

  std::default_random_engine e;
  std::uniform_int_distribution<T> rng(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  for(auto& x: h_vec)
  {
    x = rng(e);
  } 

  return thrust::device_vector<T>(h_vec.begin(), h_vec.end());
}

template<std::floating_point T>
thrust::device_vector<T> random_device_vector(std::size_t n)
{
  std::vector<T> h_vec(n);

  std::default_random_engine e;
  std::uniform_real_distribution<T> rng;
  for(auto& x: h_vec)
  {
    x = rng(e);
  } 

  return thrust::device_vector<T>(h_vec.begin(), h_vec.end());
}

