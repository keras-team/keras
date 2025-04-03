/*
 * Copyright 2020 The TensorFlow Runtime Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// device abstraction
//
// Device is a low level abstraction that represents a physical compute device
// (e.g. CPU, GPU, TPU), to be used in both op-by-op and graph execution.
#ifndef TFRT_HOST_CONTEXT_DEVICE_H_
#define TFRT_HOST_CONTEXT_DEVICE_H_

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "tfrt/support/forward_decls.h"
#include "tfrt/support/mutex.h"
#include "tfrt/support/ref_count.h"
#include "tfrt/support/thread_annotations.h"

namespace tfrt {

class HostContext;
class DeviceTypeRegistry;

// A thin wrapper of device type string like "gpu", "cpu", etc. We don't use
// enum because it requires a central place that lists all the supported value
// which is not very extensible.
class DeviceType {
 public:
  string_view name() const { return name_; }

  bool operator==(const DeviceType& other) const {
    // Safe to compare by their names because all instances are uniquely
    // registered to DeviceTypeRegistry.
    return name_ == other.name_;
  }

  static const DeviceType& kUnknownDeviceType;

 private:
  friend DeviceTypeRegistry;

  // It is hidden because all instances are managed by DeviceTypeRegistry.
  explicit DeviceType(string_view name) : name_(name) {}

  const std::string name_;
};

// Represents a device that a tensor can be placed on. E.g. "gpu:0", "tpu:1".
class Device : public ReferenceCounted<Device> {
 public:
  Device(const DeviceType& type, string_view name) : type_(type), name_(name) {
    assert(!(type == DeviceType::kUnknownDeviceType) &&
           "device type can not be Unknown");
    assert(!name_.empty() && "Cannot create a Device with empty device name");
  }

  virtual ~Device() {}

  // This class is not copyable or assignable.
  Device(const Device& other) = delete;
  Device& operator=(const Device&) = delete;

  const DeviceType& type() const { return type_; }
  string_view name() const { return name_; }

  bool IsDeviceType(const DeviceType& type) const {
    return this->type() == type;
  }

 private:
  const DeviceType& type_;
  const std::string name_;
};

// A central place to manage all active devices. Thread-safe.
class DeviceManager {
 public:
  DeviceManager() = default;
  ~DeviceManager() = default;
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;

  // Add a new device if it doesn't exist. If it doesn't exist, return the newly
  // added device, otherwise, return the existing device.
  template <typename T,
            std::enable_if_t<std::is_base_of<Device, T>::value, int> = 0>
  RCReference<T> MaybeAddDevice(RCReference<T> device) {
    mutex_lock l(mu_);
    auto it = device_map_.try_emplace(device->name(), device);
    // TODO(fishx): Change the static_cast to dyn_cast to check the type after
    // introducing classof method into Device.
    return FormRef(static_cast<T*>(it.first->second.get()));
  }

  // Lookup a device by its name. Return an empty RCReference if not found.
  template <typename T,
            std::enable_if_t<std::is_base_of<Device, T>::value, int> = 0>
  RCReference<T> GetDeviceRef(string_view device_name) const {
    mutex_lock l(mu_);
    auto it = device_map_.find(device_name);

    // TODO(fishx): Change the static_cast to dyn_cast to check the type after
    // introducing classof method into Device.
    return it == device_map_.end() ? RCReference<T>()
                                   : FormRef(static_cast<T*>(it->second.get()));
  }

  // Return a default device for a Device type. Return an empty RCReference if
  // not found.
  template <typename T,
            std::enable_if_t<std::is_base_of<Device, T>::value, int> = 0>
  RCReference<T> GetDeviceRef(const DeviceType& type) const {
    mutex_lock l(mu_);
    for (auto& it : device_map_) {
      // TODO(fishx): Change the static_cast to dyn_cast to check the type after
      // introducing classof method into Device.
      if (it.second->type() == type)
        return FormRef(static_cast<T*>(it.second.get()));
    }
    return RCReference<T>();
  }

  // Return a list of devices.
  template <typename T,
            std::enable_if_t<std::is_base_of<Device, T>::value, int> = 0>
  llvm::SmallVector<RCReference<T>, 4> ListDevices() const {
    llvm::SmallVector<RCReference<T>, 4> devices;
    devices.reserve(device_map_.size());
    mutex_lock l(mu_);
    for (auto& it : device_map_) {
      devices.push_back(it.second);
    }
    return devices;
  }

  // Return a string that lists all the devices in the device manager.
  std::string DebugString() {
    std::string out;
    mutex_lock l(mu_);
    for (const auto& it : device_map_) {
      out += it.first().str() + "\n";
    }
    return out;
  }

 private:
  mutable mutex mu_;
  llvm::StringMap<RCReference<Device>> device_map_ TFRT_GUARDED_BY(mu_);
};

// Contains all the DeviceType that are supported.
class DeviceTypeRegistry {
 public:
  // Each process should only have one DeviceTypeRegistry.
  static DeviceTypeRegistry* GetStaticDeviceTypeRegistry();

  ~DeviceTypeRegistry();

  const DeviceType& RegisterDeviceType(string_view type);
  const DeviceType& GetDeviceType(string_view type) const;

 private:
  // We don't use map because we don't expected to have many device types. And
  // it is not on the performance critical path. However we need to avoid
  // reallocations as Device stores the reference.
  llvm::SmallVector<std::unique_ptr<DeviceType>, 4> types_;
};

// Register and return a new DeviceType.
const DeviceType& RegisterDeviceType(string_view type);

const DeviceType& GetStaticDeviceType(string_view type);

// DeviceTraits register DeviceType for Derived class. Each sub Device needs to
// inherit DeviceTraits.
// Example usage:
// class MyDevice : public Device, public DeviceTraits<MyDevice> {};
template <class Derived>
class DeviceTraits {
 public:
  static const DeviceType& kDeviceType;

  static bool classof(const Device* t) { return t->IsDeviceType(kDeviceType); }
};

template <class Derived>
const tfrt::DeviceType& DeviceTraits<Derived>::kDeviceType =
    RegisterDeviceType(Derived::type_name());

class CpuDevice : public Device, public DeviceTraits<CpuDevice> {
 public:
  static const char* type_name() {
    static constexpr char kName[] = "cpu";
    return kName;
  }

  explicit CpuDevice(string_view name) : Device(kDeviceType, name) {}

  ~CpuDevice() override {}
};

}  // namespace tfrt

#endif  // TFRT_HOST_CONTEXT_DEVICE_H_
