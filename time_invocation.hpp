#pragma once

#include <chrono>
#include <cstddef>
#include <utility>

template<class Duration, class Clock, class Function>
Duration time_invocation_in(const Clock& clock, std::size_t num_trials, Function f)
{
  // warm up
  for(std::size_t i = 0; i < num_trials; ++i)
  {
    f();
  }

  auto start = clock.now();
  for(std::size_t i = 0;
      i < num_trials;
      ++i)
  {
    f();
  }
  auto end = clock.now();

  // return mean duration
  return std::chrono::duration_cast<Duration>(end - start) / num_trials;
}

template<class Function>
std::size_t time_invocation_in_nanoseconds(std::size_t num_trials, Function f)
{
  return ::time_invocation_in<std::chrono::nanoseconds>(std::chrono::high_resolution_clock(), num_trials, f).count();
}

template<class Function>
std::size_t time_invocation_in_microseconds(std::size_t num_trials, Function f)
{
  return ::time_invocation_in<std::chrono::microseconds>(std::chrono::high_resolution_clock(), num_trials, f).count();
}

template<class Function>
std::size_t time_invocation_in_milliseconds(std::size_t num_trials, Function f)
{
  return ::time_invocation_in<std::chrono::milliseconds>(std::chrono::high_resolution_clock(), num_trials, f).count();
}

template<class Function>
std::size_t time_invocation_in_seconds(std::size_t num_trials, Function f)
{
  return ::time_invocation_in<std::chrono::seconds>(std::chrono::system_clock(), num_trials, f).count();
}


template<class Duration, class Clock, class Function1, class Function2>
Duration time_invocation_with_reset_in(const Clock& clock, std::size_t num_trials, Function1 f, Function2 reset)
{
  // warm up
  for(std::size_t i = 0; i < num_trials; ++i)
  {
    reset();
    f();
  }

  typename Clock::duration result = 0;

  for(std::size_t i = 0; i < num_trials; ++i)
  {
    reset();

    auto start = clock.now();
    f();
    auto end = clock.now();

    result += (end - start);
  }

  // return mean duration
  return std::chrono::duration_cast<Duration>(result) / num_trials;
}

template<class Function1, class Function2>
std::size_t time_invocation_with_reset_in_nanoseconds(std::size_t num_trials, Function1 f, Function2 reset)
{
  return time_invocation_with_reset_in<std::chrono::nanoseconds>(std::chrono::high_resolution_clock(), num_trials, f).count();
}

template<class Function1, class Function2>
std::size_t time_invocation_with_reset_in_microseconds(std::size_t num_trials, Function1 f, Function2 reset)
{
  return time_invocation_with_reset_in<std::chrono::microseconds>(std::chrono::high_resolution_clock(), num_trials, f).count();
}

template<class Function1, class Function2>
std::size_t time_invocation_with_reset_in_milliseconds(std::size_t num_trials, Function1 f, Function2 reset)
{
  return time_invocation_with_reset_in<std::chrono::milliseconds>(std::chrono::high_resolution_clock(), num_trials, f).count();
}

template<class Function1, class Function2>
std::size_t time_invocation_with_reset_in_seconds(std::size_t num_trials, Function1 f, Function2 reset)
{
  return time_invocation_with_reset_in<std::chrono::seconds>(std::chrono::system_clock(), num_trials, f).count();
}

