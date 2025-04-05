// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>

//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#if defined(EIGEN_USE_SYCL) && !defined(EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H)
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H
#include <unordered_set>

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace TensorSycl {
namespace internal {

/// Cache all the device information needed
struct SyclDeviceInfo {
  SyclDeviceInfo(cl::sycl::queue queue)
      : local_mem_type(queue.get_device().template get_info<cl::sycl::info::device::local_mem_type>()),
        max_work_item_sizes(queue.get_device().template get_info<cl::sycl::info::device::max_work_item_sizes<3>>()),
        max_mem_alloc_size(queue.get_device().template get_info<cl::sycl::info::device::max_mem_alloc_size>()),
        max_compute_units(queue.get_device().template get_info<cl::sycl::info::device::max_compute_units>()),
        max_work_group_size(queue.get_device().template get_info<cl::sycl::info::device::max_work_group_size>()),
        local_mem_size(queue.get_device().template get_info<cl::sycl::info::device::local_mem_size>()),
        platform_name(queue.get_device().get_platform().template get_info<cl::sycl::info::platform::name>()),
        device_name(queue.get_device().template get_info<cl::sycl::info::device::name>()),
        device_vendor(queue.get_device().template get_info<cl::sycl::info::device::vendor>()) {}

  cl::sycl::info::local_mem_type local_mem_type;
  cl::sycl::id<3> max_work_item_sizes;
  unsigned long max_mem_alloc_size;
  unsigned long max_compute_units;
  unsigned long max_work_group_size;
  size_t local_mem_size;
  std::string platform_name;
  std::string device_name;
  std::string device_vendor;
};

}  // end namespace internal
}  // end namespace TensorSycl

// All devices (even AMD CPU with intel OpenCL runtime) that support OpenCL and
// can consume SPIR or SPIRV can use the Eigen SYCL backend and consequently
// TensorFlow via the Eigen SYCL Backend.
EIGEN_STRONG_INLINE auto get_sycl_supported_devices() -> decltype(cl::sycl::device::get_devices()) {
#ifdef EIGEN_SYCL_USE_DEFAULT_SELECTOR
  return {cl::sycl::device(cl::sycl::default_selector())};
#else
  std::vector<cl::sycl::device> supported_devices;
  auto platform_list = cl::sycl::platform::get_platforms();
  for (const auto &platform : platform_list) {
    auto device_list = platform.get_devices();
    auto platform_name = platform.template get_info<cl::sycl::info::platform::name>();
    std::transform(platform_name.begin(), platform_name.end(), platform_name.begin(), ::tolower);
    for (const auto &device : device_list) {
      auto vendor = device.template get_info<cl::sycl::info::device::vendor>();
      std::transform(vendor.begin(), vendor.end(), vendor.begin(), ::tolower);
      bool unsupported_condition = (device.is_cpu() && platform_name.find("amd") != std::string::npos &&
                                    vendor.find("apu") == std::string::npos) ||
                                   (platform_name.find("experimental") != std::string::npos) || device.is_host();
      if (!unsupported_condition) {
        supported_devices.push_back(device);
      }
    }
  }
  return supported_devices;
#endif
}

class QueueInterface {
 public:
  /// Creating device by using cl::sycl::selector or cl::sycl::device.
  template <typename DeviceOrSelector>
  explicit QueueInterface(const DeviceOrSelector &dev_or_sel, cl::sycl::async_handler handler,
                          unsigned num_threads = std::thread::hardware_concurrency())
      : m_queue{dev_or_sel, handler, {sycl::property::queue::in_order()}},
        m_thread_pool(num_threads),
        m_device_info(m_queue) {}

  template <typename DeviceOrSelector>
  explicit QueueInterface(const DeviceOrSelector &dev_or_sel,
                          unsigned num_threads = std::thread::hardware_concurrency())
      : QueueInterface(
            dev_or_sel, [this](cl::sycl::exception_list l) { this->exception_caught_ = this->sycl_async_handler(l); },
            num_threads) {}

  explicit QueueInterface(const cl::sycl::queue &q, unsigned num_threads = std::thread::hardware_concurrency())
      : m_queue(q), m_thread_pool(num_threads), m_device_info(m_queue) {}

  EIGEN_STRONG_INLINE void *allocate(size_t num_bytes) const {
#if EIGEN_MAX_ALIGN_BYTES > 0
    return (void *)cl::sycl::aligned_alloc_device(EIGEN_MAX_ALIGN_BYTES, num_bytes, m_queue);
#else
    return (void *)cl::sycl::malloc_device(num_bytes, m_queue);
#endif
  }

  EIGEN_STRONG_INLINE void *allocate_temp(size_t num_bytes) const {
    return (void *)cl::sycl::malloc_device<uint8_t>(num_bytes, m_queue);
  }

  template <typename data_t>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE data_t *get(data_t *data) const {
    return data;
  }

  EIGEN_STRONG_INLINE void deallocate_temp(void *p) const { deallocate(p); }

  EIGEN_STRONG_INLINE void deallocate_temp(const void *p) const { deallocate_temp(const_cast<void *>(p)); }

  EIGEN_STRONG_INLINE void deallocate(void *p) const { cl::sycl::free(p, m_queue); }

  /// The memcpyHostToDevice is used to copy the data from host to device
  /// The destination pointer could be deleted before the copy happened which is
  /// why a callback function is needed. By default if none is provided, the
  /// function is blocking.
  EIGEN_STRONG_INLINE void memcpyHostToDevice(void *dst, const void *src, size_t n,
                                              std::function<void()> callback) const {
    auto e = m_queue.memcpy(dst, src, n);
    synchronize_and_callback(e, callback);
  }

  /// The memcpyDeviceToHost is used to copy the data from device to host.
  /// The source pointer could be deleted before the copy happened which is
  /// why a callback function is needed. By default if none is provided, the
  /// function is blocking.
  EIGEN_STRONG_INLINE void memcpyDeviceToHost(void *dst, const void *src, size_t n,
                                              std::function<void()> callback) const {
    if (n == 0) {
      if (callback) callback();
      return;
    }
    auto e = m_queue.memcpy(dst, src, n);
    synchronize_and_callback(e, callback);
  }

  /// The memcpy function.
  /// No callback is required here as both arguments are on the device
  /// and SYCL can handle the dependency.
  EIGEN_STRONG_INLINE void memcpy(void *dst, const void *src, size_t n) const {
    if (n == 0) {
      return;
    }
    m_queue.memcpy(dst, src, n).wait();
  }

  /// the memset function.
  /// No callback is required here as both arguments are on the device
  /// and SYCL can handle the dependency.
  EIGEN_STRONG_INLINE void memset(void *data, int c, size_t n) const {
    if (n == 0) {
      return;
    }
    m_queue.memset(data, c, n).wait();
  }

  template <typename T>
  EIGEN_STRONG_INLINE void fill(T *begin, T *end, const T &value) const {
    if (begin == end) {
      return;
    }
    const size_t count = end - begin;
    m_queue.fill(begin, value, count).wait();
  }

  template <typename OutScalar, typename sycl_kernel, typename Lhs, typename Rhs, typename OutPtr, typename Range,
            typename Index, typename... T>
  EIGEN_ALWAYS_INLINE cl::sycl::event binary_kernel_launcher(const Lhs &lhs, const Rhs &rhs, OutPtr outptr,
                                                             Range thread_range, Index scratchSize, T... var) const {
    auto kernel_functor = [=](cl::sycl::handler &cgh) {
      typedef cl::sycl::accessor<OutScalar, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
          LocalAccessor;

      LocalAccessor scratch(cl::sycl::range<1>(scratchSize), cgh);
      cgh.parallel_for(thread_range, sycl_kernel(scratch, lhs, rhs, outptr, var...));
    };

    return m_queue.submit(kernel_functor);
  }

  template <typename OutScalar, typename sycl_kernel, typename InPtr, typename OutPtr, typename Range, typename Index,
            typename... T>
  EIGEN_ALWAYS_INLINE cl::sycl::event unary_kernel_launcher(const InPtr &inptr, OutPtr &outptr, Range thread_range,
                                                            Index scratchSize, T... var) const {
    auto kernel_functor = [=](cl::sycl::handler &cgh) {
      typedef cl::sycl::accessor<OutScalar, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
          LocalAccessor;

      LocalAccessor scratch(cl::sycl::range<1>(scratchSize), cgh);
      cgh.parallel_for(thread_range, sycl_kernel(scratch, inptr, outptr, var...));
    };
    return m_queue.submit(kernel_functor);
  }

  template <typename OutScalar, typename sycl_kernel, typename InPtr, typename Range, typename Index, typename... T>
  EIGEN_ALWAYS_INLINE cl::sycl::event nullary_kernel_launcher(const InPtr &inptr, Range thread_range, Index scratchSize,
                                                              T... var) const {
    auto kernel_functor = [=](cl::sycl::handler &cgh) {
      typedef cl::sycl::accessor<OutScalar, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
          LocalAccessor;

      LocalAccessor scratch(cl::sycl::range<1>(scratchSize), cgh);
      cgh.parallel_for(thread_range, sycl_kernel(scratch, inptr, var...));
    };

    return m_queue.submit(kernel_functor);
  }

  EIGEN_STRONG_INLINE void synchronize() const {
#ifdef EIGEN_EXCEPTIONS
    m_queue.wait_and_throw();
#else
    m_queue.wait();
#endif
  }

  template <typename Index>
  EIGEN_STRONG_INLINE void parallel_for_setup(Index n, Index &tileSize, Index &rng, Index &GRange) const {
    tileSize = static_cast<Index>(getNearestPowerOfTwoWorkGroupSize());
    tileSize = std::min(static_cast<Index>(EIGEN_SYCL_LOCAL_THREAD_DIM0 * EIGEN_SYCL_LOCAL_THREAD_DIM1),
                        static_cast<Index>(tileSize));
    rng = n;
    if (rng == 0) rng = static_cast<Index>(1);
    GRange = rng;
    if (tileSize > GRange)
      tileSize = GRange;
    else if (GRange > tileSize) {
      Index xMode = static_cast<Index>(GRange % tileSize);
      if (xMode != 0) GRange += static_cast<Index>(tileSize - xMode);
    }
  }

  /// This is used to prepare the number of threads and also the number of
  /// threads per block for sycl kernels
  template <typename Index>
  EIGEN_STRONG_INLINE void parallel_for_setup(const std::array<Index, 2> &input_dim, cl::sycl::range<2> &global_range,
                                              cl::sycl::range<2> &local_range) const {
    std::array<Index, 2> input_range = input_dim;
    Index max_workgroup_Size = static_cast<Index>(getNearestPowerOfTwoWorkGroupSize());
    max_workgroup_Size = std::min(static_cast<Index>(EIGEN_SYCL_LOCAL_THREAD_DIM0 * EIGEN_SYCL_LOCAL_THREAD_DIM1),
                                  static_cast<Index>(max_workgroup_Size));
    Index pow_of_2 = static_cast<Index>(std::log2(max_workgroup_Size));
    local_range[1] = static_cast<Index>(std::pow(2, static_cast<Index>(pow_of_2 / 2)));
    input_range[1] = input_dim[1];
    if (input_range[1] == 0) input_range[1] = static_cast<Index>(1);
    global_range[1] = input_range[1];
    if (local_range[1] > global_range[1])
      local_range[1] = global_range[1];
    else if (global_range[1] > local_range[1]) {
      Index xMode = static_cast<Index>(global_range[1] % local_range[1]);
      if (xMode != 0) global_range[1] += static_cast<Index>(local_range[1] - xMode);
    }
    local_range[0] = static_cast<Index>(max_workgroup_Size / local_range[1]);
    input_range[0] = input_dim[0];
    if (input_range[0] == 0) input_range[0] = static_cast<Index>(1);
    global_range[0] = input_range[0];
    if (local_range[0] > global_range[0])
      local_range[0] = global_range[0];
    else if (global_range[0] > local_range[0]) {
      Index xMode = static_cast<Index>(global_range[0] % local_range[0]);
      if (xMode != 0) global_range[0] += static_cast<Index>(local_range[0] - xMode);
    }
  }

  /// This is used to prepare the number of threads and also the number of
  /// threads per block for sycl kernels
  template <typename Index>
  EIGEN_STRONG_INLINE void parallel_for_setup(const std::array<Index, 3> &input_dim, cl::sycl::range<3> &global_range,
                                              cl::sycl::range<3> &local_range) const {
    std::array<Index, 3> input_range = input_dim;
    Index max_workgroup_Size = static_cast<Index>(getNearestPowerOfTwoWorkGroupSize());
    max_workgroup_Size = std::min(static_cast<Index>(EIGEN_SYCL_LOCAL_THREAD_DIM0 * EIGEN_SYCL_LOCAL_THREAD_DIM1),
                                  static_cast<Index>(max_workgroup_Size));
    Index pow_of_2 = static_cast<Index>(std::log2(max_workgroup_Size));
    local_range[2] = static_cast<Index>(std::pow(2, static_cast<Index>(pow_of_2 / 3)));
    input_range[2] = input_dim[2];
    if (input_range[2] == 0) input_range[1] = static_cast<Index>(1);
    global_range[2] = input_range[2];
    if (local_range[2] > global_range[2])
      local_range[2] = global_range[2];
    else if (global_range[2] > local_range[2]) {
      Index xMode = static_cast<Index>(global_range[2] % local_range[2]);
      if (xMode != 0) global_range[2] += static_cast<Index>(local_range[2] - xMode);
    }
    pow_of_2 = static_cast<Index>(std::log2(static_cast<Index>(max_workgroup_Size / local_range[2])));
    local_range[1] = static_cast<Index>(std::pow(2, static_cast<Index>(pow_of_2 / 2)));
    input_range[1] = input_dim[1];
    if (input_range[1] == 0) input_range[1] = static_cast<Index>(1);
    global_range[1] = input_range[1];
    if (local_range[1] > global_range[1])
      local_range[1] = global_range[1];
    else if (global_range[1] > local_range[1]) {
      Index xMode = static_cast<Index>(global_range[1] % local_range[1]);
      if (xMode != 0) global_range[1] += static_cast<Index>(local_range[1] - xMode);
    }
    local_range[0] = static_cast<Index>(max_workgroup_Size / (local_range[1] * local_range[2]));
    input_range[0] = input_dim[0];
    if (input_range[0] == 0) input_range[0] = static_cast<Index>(1);
    global_range[0] = input_range[0];
    if (local_range[0] > global_range[0])
      local_range[0] = global_range[0];
    else if (global_range[0] > local_range[0]) {
      Index xMode = static_cast<Index>(global_range[0] % local_range[0]);
      if (xMode != 0) global_range[0] += static_cast<Index>(local_range[0] - xMode);
    }
  }

  EIGEN_STRONG_INLINE bool has_local_memory() const {
#if !defined(EIGEN_SYCL_LOCAL_MEM) && defined(EIGEN_SYCL_NO_LOCAL_MEM)
    return false;
#elif defined(EIGEN_SYCL_LOCAL_MEM) && !defined(EIGEN_SYCL_NO_LOCAL_MEM)
    return true;
#else
    return m_device_info.local_mem_type == cl::sycl::info::local_mem_type::local;
#endif
  }

  EIGEN_STRONG_INLINE unsigned long max_buffer_size() const { return m_device_info.max_mem_alloc_size; }

  EIGEN_STRONG_INLINE unsigned long getNumSyclMultiProcessors() const { return m_device_info.max_compute_units; }

  EIGEN_STRONG_INLINE unsigned long maxSyclThreadsPerBlock() const { return m_device_info.max_work_group_size; }

  EIGEN_STRONG_INLINE cl::sycl::id<3> maxWorkItemSizes() const { return m_device_info.max_work_item_sizes; }

  /// No need for sycl it should act the same as CPU version
  EIGEN_STRONG_INLINE int majorDeviceVersion() const { return 1; }

  EIGEN_STRONG_INLINE unsigned long maxSyclThreadsPerMultiProcessor() const {
    // OpenCL does not have such a concept
    return 2;
  }

  EIGEN_STRONG_INLINE size_t sharedMemPerBlock() const { return m_device_info.local_mem_size; }

  // This function returns the nearest power of 2 Work-group size which is <=
  // maximum device workgroup size.
  EIGEN_STRONG_INLINE size_t getNearestPowerOfTwoWorkGroupSize() const {
    return getPowerOfTwo(m_device_info.max_work_group_size, false);
  }

  EIGEN_STRONG_INLINE std::string getPlatformName() const { return m_device_info.platform_name; }

  EIGEN_STRONG_INLINE std::string getDeviceName() const { return m_device_info.device_name; }

  EIGEN_STRONG_INLINE std::string getDeviceVendor() const { return m_device_info.device_vendor; }

  // This function returns the nearest power of 2
  // if roundup is true returns result>=wgsize
  // else it return result <= wgsize
  EIGEN_STRONG_INLINE size_t getPowerOfTwo(size_t wGSize, bool roundUp) const {
    if (roundUp) --wGSize;
    wGSize |= (wGSize >> 1);
    wGSize |= (wGSize >> 2);
    wGSize |= (wGSize >> 4);
    wGSize |= (wGSize >> 8);
    wGSize |= (wGSize >> 16);
#if EIGEN_ARCH_x86_64 || EIGEN_ARCH_ARM64 || EIGEN_OS_WIN64
    wGSize |= (wGSize >> 32);
#endif
    return ((!roundUp) ? (wGSize - (wGSize >> 1)) : ++wGSize);
  }

  EIGEN_STRONG_INLINE cl::sycl::queue &sycl_queue() const { return m_queue; }

  // This function checks if the runtime recorded an error for the
  // underlying stream device.
  EIGEN_STRONG_INLINE bool ok() const {
    if (!exception_caught_) {
      synchronize();
    }
    return !exception_caught_;
  }

 protected:
  void synchronize_and_callback(cl::sycl::event e, const std::function<void()> &callback) const {
    if (callback) {
      auto callback_ = [=]() {
#ifdef EIGEN_EXCEPTIONS
        cl::sycl::event(e).wait_and_throw();
#else
        cl::sycl::event(e).wait();
#endif
        callback();
      };
      m_thread_pool.Schedule(std::move(callback_));
    } else {
#ifdef EIGEN_EXCEPTIONS
      m_queue.wait_and_throw();
#else
      m_queue.wait();
#endif
    }
  }

  bool sycl_async_handler(cl::sycl::exception_list exceptions) const {
    bool exception_caught = false;
    for (const auto &e : exceptions) {
      if (e) {
        exception_caught = true;
        EIGEN_THROW_X(e);
      }
    }
    return exception_caught;
  }

  /// class members:
  bool exception_caught_ = false;
  /// sycl queue
  mutable cl::sycl::queue m_queue;
  /// The thread pool is used to wait on events and call callbacks
  /// asynchronously
  mutable Eigen::ThreadPool m_thread_pool;

  const TensorSycl::internal::SyclDeviceInfo m_device_info;
};

struct SyclDeviceBase {
  /// QueueInterface is not owned. it is the caller's responsibility to destroy
  /// it
  const QueueInterface *m_queue_stream;
  explicit SyclDeviceBase(const QueueInterface *queue_stream) : m_queue_stream(queue_stream) {}
  EIGEN_STRONG_INLINE const QueueInterface *queue_stream() const { return m_queue_stream; }
};

// Here is a sycl device struct which accept the sycl queue interface
// as an input
struct SyclDevice : public SyclDeviceBase {
  explicit SyclDevice(const QueueInterface *queue_stream) : SyclDeviceBase(queue_stream) {}

  /// This is used to prepare the number of threads and also the number of
  /// threads per block for sycl kernels
  template <typename Index>
  EIGEN_STRONG_INLINE void parallel_for_setup(Index n, Index &tileSize, Index &rng, Index &GRange) const {
    queue_stream()->parallel_for_setup(n, tileSize, rng, GRange);
  }

  /// This is used to prepare the number of threads and also the number of
  /// threads per block for sycl kernels
  template <typename Index>
  EIGEN_STRONG_INLINE void parallel_for_setup(const std::array<Index, 2> &input_dim, cl::sycl::range<2> &global_range,
                                              cl::sycl::range<2> &local_range) const {
    queue_stream()->parallel_for_setup(input_dim, global_range, local_range);
  }

  /// This is used to prepare the number of threads and also the number of
  /// threads per block for sycl kernels
  template <typename Index>
  EIGEN_STRONG_INLINE void parallel_for_setup(const std::array<Index, 3> &input_dim, cl::sycl::range<3> &global_range,
                                              cl::sycl::range<3> &local_range) const {
    queue_stream()->parallel_for_setup(input_dim, global_range, local_range);
  }

  /// allocate device memory
  EIGEN_STRONG_INLINE void *allocate(size_t num_bytes) const { return queue_stream()->allocate(num_bytes); }

  EIGEN_STRONG_INLINE void *allocate_temp(size_t num_bytes) const { return queue_stream()->allocate_temp(num_bytes); }

  /// deallocate device memory
  EIGEN_STRONG_INLINE void deallocate(void *p) const { queue_stream()->deallocate(p); }

  EIGEN_STRONG_INLINE void deallocate_temp(void *buffer) const { queue_stream()->deallocate_temp(buffer); }

  EIGEN_STRONG_INLINE void deallocate_temp(const void *buffer) const { queue_stream()->deallocate_temp(buffer); }

  template <typename data_t>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE data_t *get(data_t *data) const {
    return data;
  }

  // some runtime conditions that can be applied here
  EIGEN_STRONG_INLINE bool isDeviceSuitable() const { return true; }

  /// memcpyHostToDevice
  template <typename Index>
  EIGEN_STRONG_INLINE void memcpyHostToDevice(Index *dst, const Index *src, size_t n,
                                              std::function<void()> callback = {}) const {
    queue_stream()->memcpyHostToDevice(dst, src, n, callback);
  }
  /// memcpyDeviceToHost
  template <typename Index>
  EIGEN_STRONG_INLINE void memcpyDeviceToHost(void *dst, const Index *src, size_t n,
                                              std::function<void()> callback = {}) const {
    queue_stream()->memcpyDeviceToHost(dst, src, n, callback);
  }
  /// the memcpy function
  template <typename Index>
  EIGEN_STRONG_INLINE void memcpy(void *dst, const Index *src, size_t n) const {
    queue_stream()->memcpy(dst, src, n);
  }
  /// the memset function
  EIGEN_STRONG_INLINE void memset(void *data, int c, size_t n) const { queue_stream()->memset(data, c, n); }
  /// the fill function
  template <typename T>
  EIGEN_STRONG_INLINE void fill(T *begin, T *end, const T &value) const {
    queue_stream()->fill(begin, end, value);
  }
  /// returning the sycl queue
  EIGEN_STRONG_INLINE cl::sycl::queue &sycl_queue() const { return queue_stream()->sycl_queue(); }

  EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const { return 48 * 1024; }

  EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
    // We won't try to take advantage of the l2 cache for the time being, and
    // there is no l3 cache on sycl devices.
    return firstLevelCacheSize();
  }
  EIGEN_STRONG_INLINE unsigned long getNumSyclMultiProcessors() const {
    return queue_stream()->getNumSyclMultiProcessors();
  }
  EIGEN_STRONG_INLINE unsigned long maxSyclThreadsPerBlock() const { return queue_stream()->maxSyclThreadsPerBlock(); }
  EIGEN_STRONG_INLINE cl::sycl::id<3> maxWorkItemSizes() const { return queue_stream()->maxWorkItemSizes(); }
  EIGEN_STRONG_INLINE unsigned long maxSyclThreadsPerMultiProcessor() const {
    // OpenCL does not have such a concept
    return queue_stream()->maxSyclThreadsPerMultiProcessor();
  }
  EIGEN_STRONG_INLINE size_t sharedMemPerBlock() const { return queue_stream()->sharedMemPerBlock(); }
  EIGEN_STRONG_INLINE size_t getNearestPowerOfTwoWorkGroupSize() const {
    return queue_stream()->getNearestPowerOfTwoWorkGroupSize();
  }

  EIGEN_STRONG_INLINE size_t getPowerOfTwo(size_t val, bool roundUp) const {
    return queue_stream()->getPowerOfTwo(val, roundUp);
  }
  /// No need for sycl it should act the same as CPU version
  EIGEN_STRONG_INLINE int majorDeviceVersion() const { return queue_stream()->majorDeviceVersion(); }

  EIGEN_STRONG_INLINE void synchronize() const { queue_stream()->synchronize(); }

  // This function checks if the runtime recorded an error for the
  // underlying stream device.
  EIGEN_STRONG_INLINE bool ok() const { return queue_stream()->ok(); }

  EIGEN_STRONG_INLINE bool has_local_memory() const { return queue_stream()->has_local_memory(); }
  EIGEN_STRONG_INLINE long max_buffer_size() const { return queue_stream()->max_buffer_size(); }
  EIGEN_STRONG_INLINE std::string getPlatformName() const { return queue_stream()->getPlatformName(); }
  EIGEN_STRONG_INLINE std::string getDeviceName() const { return queue_stream()->getDeviceName(); }
  EIGEN_STRONG_INLINE std::string getDeviceVendor() const { return queue_stream()->getDeviceVendor(); }
  template <typename OutScalar, typename KernelType, typename... T>
  EIGEN_ALWAYS_INLINE cl::sycl::event binary_kernel_launcher(T... var) const {
    return queue_stream()->template binary_kernel_launcher<OutScalar, KernelType>(var...);
  }
  template <typename OutScalar, typename KernelType, typename... T>
  EIGEN_ALWAYS_INLINE cl::sycl::event unary_kernel_launcher(T... var) const {
    return queue_stream()->template unary_kernel_launcher<OutScalar, KernelType>(var...);
  }

  template <typename OutScalar, typename KernelType, typename... T>
  EIGEN_ALWAYS_INLINE cl::sycl::event nullary_kernel_launcher(T... var) const {
    return queue_stream()->template nullary_kernel_launcher<OutScalar, KernelType>(var...);
  }
};
}  // end namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_SYCL_H
