#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <typeinfo>
#include "time_invocation_cuda.hpp"
#include "utility.hpp"

template<class Vector>
struct radix_sort_functor
{
  typename Vector::iterator first, last;

  __host__ __device__
  radix_sort_functor(Vector& vec)
    : first(vec.begin()), last(vec.end())
  {}

  __host__ __device__
  void operator()()
  {
    thrust::stable_sort(thrust::cuda::par, first, last);
  }
};

template<class Vector>
radix_sort_functor<Vector> make_radix_sort_functor(Vector& vec)
{
  return radix_sort_functor<Vector>(vec);
}

template<class T>
double time(size_t n)
{
  thrust::device_vector<T> vec(n);

  auto f = make_radix_sort_functor(vec);
  
  return time_function(f);
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
    call_me = time<double>;
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

  std::clog << "T: " << type << std::endl;
  std::clog << "n: " << n << std::endl;

  double ms = std::numeric_limits<double>::infinity();

//  try
//  {
    ms = call_me(n);
//  }
//  catch(...)
//  {
//    std::cout << "bad: " << ms << std::endl;
//  }

  std::clog << "ms: " << ms << std::endl;

  std::cout << ms;

  return 0;
}

