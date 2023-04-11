#include "time_invocation.hpp"
#include <thrust/device_vector.h>
#include <thrust/transform.h>


template<class T>
struct saxpy_functor
{
  T a;

  __host__ __device__
  T operator()(const T& x, const T& y) const
  {
    return a * x + y;
  }
};


template<class T>
double time(size_t n)
{
  thrust::device_vector<T> vec1(n);
  thrust::device_vector<T> vec2(n);
  thrust::device_vector<T> vec3(n);

  auto us = time_invocation_in_microseconds(100, [&]
  {
    thrust::transform(vec1.begin(), vec1.end(), vec2.begin(), vec3.begin(), saxpy_functor<T>(13));

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

  std::clog << "T: " << type << std::endl;
  std::clog << "n: " << n << std::endl;

  double seconds = call_me(n);

  std::clog << "s: " << seconds << std::endl;

  std::cout << seconds;

  return 0;
}

