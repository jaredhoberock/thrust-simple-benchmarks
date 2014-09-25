#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/system/cuda/execution_policy.h>
#include <typeinfo>
#include "time_invocation_cuda.hpp"
#include "utility.hpp"
#include <functional>

template<class Vector>
struct saxpy_functor
{
  typename Vector::iterator first1, last1, first2, result;

  __host__ __device__
  saxpy_functor(typename Vector::iterator first1_,
                typename Vector::iterator last1_,
                typename Vector::iterator first2_,
                typename Vector::iterator result_)
    : first1(first1_), last1(last1_),
      first2(first2_),
      result(result_)
  {}

  using value_type = typename Vector::value_type;

  struct f
  {
    value_type a;

    __host__ __device__
    value_type operator()(const value_type& x, const value_type& y)
    {
      return a * x + y;
    }
  };

  __host__ __device__
  void operator()()
  {
    thrust::transform(thrust::cuda::par, first1, last1, first2, result, f());
  }
};

template<class Vector>
saxpy_functor<Vector> make_saxpy_functor(size_t n, Vector& vec1, Vector& vec2, Vector& vec3)
{
  return saxpy_functor<Vector>(vec1.begin(), vec1.begin() + n,
                               vec2.begin(),
                               vec3.begin());
}


template<class T>
void time(size_t n, size_t step)
{
  thrust::device_vector<T> vec1(n);
  thrust::device_vector<T> vec2(n);
  thrust::device_vector<T> vec3(n);

  std::cout << "data size, ms" << std::endl;

  for(size_t m = 0; m <= n; m += step)
  {
    auto f = make_saxpy_functor(m, vec1, vec2, vec3);

    double ms = 1000000;

    try
    {
      ms = time_function(f, 1);
    }
    catch(...)
    {
    }

    std::cout << m << ", " << ms << std::endl;
  }
}


int main(int argc, char** argv)
{
  std::string type = "int";

  if(argc >= 2)
  {
    type = argv[1];
  }

  std::function<void(size_t,size_t)> call_me = time<int>;

  if(type == "int")
  {
    call_me = time<int>;
  }
  else if(type == "float")
  {
    call_me = time<float>;
  }
  else if(type == "int64")
  {
    call_me = time<long>;
  }
  else if(type == "double")
  {
    call_me = time<double>;
  }

  size_t n = 16 << 20;

  if(argc >= 3)
  {
    n = atoi(argv[2]);
  }

  size_t step = 5000;

  if(argc >= 4)
  {
    step = atoi(argv[3]);
  }

  std::cout << type << std::endl;
  time<int>(n, step);
  std::cout << std::endl;

  return 0;
}

