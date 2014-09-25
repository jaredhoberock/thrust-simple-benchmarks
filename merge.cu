#include <thrust/device_vector.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/system/cuda/execution_policy.h>
#include <typeinfo>
#include "time_invocation_cuda.hpp"
#include "utility.hpp"

template<class Vector>
struct merge_functor
{
  typename Vector::iterator first1, last1, first2, last2, result;

  __host__ __device__
  merge_functor(Vector& vec1, Vector& vec2, Vector& vec3)
    : first1(vec1.begin()), last1(vec1.end()),
      first2(vec2.begin()), last2(vec2.end()),
      result(vec3.begin())
  {}

  __host__ __device__
  void operator()()
  {
    thrust::merge(thrust::cuda::par,
                  first1, last1,
                  first2, last2,
                  result);
  }
};

template<class Vector>
merge_functor<Vector> make_merge_functor(Vector& vec1, Vector& vec2, Vector& vec3)
{
  return merge_functor<Vector>(vec1, vec2, vec3);
}


struct hash
{
  unsigned int init_;

  __host__ __device__
  hash(unsigned int init)
    : init_(init)
  {}

  __host__ __device__
  unsigned int operator()(unsigned int a)
  {
    a += init_;

    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
  }
};


template<class T>
double time(size_t n)
{
  thrust::device_vector<T> vec1(n);
  thrust::device_vector<T> vec2(n);
  thrust::device_vector<T> vec3(2 * n);

  thrust::tabulate(vec1.begin(), vec1.end(), hash(0));
  thrust::tabulate(vec2.begin(), vec2.end(), hash(n));

  thrust::sort(vec1.begin(), vec1.end());
  thrust::sort(vec2.begin(), vec2.end());

  auto f = make_merge_functor(vec1, vec2, vec3);
  
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

  double ms = 1000000;

  try
  {
    ms = call_me(n);
  }
  catch(...)
  {
  }

  std::clog << "ms: " << ms << std::endl;

  std::cout << ms;

  return 0;
}

