#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "time_invocation.hpp"

template<class T>
double time(size_t n)
{
  thrust::device_vector<T> vec(n);

  auto us = time_invocation_in_microseconds(100, [&]
  {
    thrust::reduce(vec.begin(), vec.end());
    cudaDeviceSynchronize();
  });

  return static_cast<double>(us) / 1000000;
}

int main(int argc, char** argv)
{
  double (*call_me)(size_t) = time<int>;
  std::string type = "int";

  if(argc >= 2)
  {
    type = argv[1];
  }

  size_t n = 1 << 20;

  if(argc >= 3)
  {
    n = atoi(argv[2]);
  }

  if(type == "int")
  {
    call_me = time<int>;
  }
  else if(type == "long")
  {
    call_me = time<uint64_t>;
  }
  else if(type == "float")
  {
    call_me = time<float>;
  }
  else if(type == "double")
  {
    call_me = time<double>;
  }
  else
  {
    throw std::runtime_error("Unrecognized type");
  }

  double seconds = call_me(n);

  std::cout << n << ", " << seconds;

  return 0;
}

