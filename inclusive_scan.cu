#include "time_invocation.hpp"
#include <thrust/device_vector.h>
#include <thrust/scan.h>

template<class Vector>
struct inclusive_scan_functor
{
  typename Vector::iterator first, last, result;

  __host__ __device__
  inclusive_scan_functor(Vector& vec)
    : first(vec.begin()), last(vec.end()), result(vec.begin())
  {}

  __host__ __device__
  void operator()()
  {
    thrust::inclusive_scan(thrust::cuda::par, first, last, result);
  }
};

template<class Vector>
inclusive_scan_functor<Vector> make_inclusive_scan_functor(Vector& vec)
{
  return inclusive_scan_functor<Vector>(vec);
}

template<class T>
double time(size_t n)
{
  thrust::device_vector<T> input(n);
  thrust::device_vector<T> result(n);

  auto us = time_invocation_in_microseconds(100, [&]
  {
    thrust::inclusive_scan(input.begin(), input.end(), result.begin());
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

