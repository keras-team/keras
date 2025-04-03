// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware specific properties for OpenVINO runtime devices
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file openvino/runtime/properties.hpp
 */
#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <iomanip>
#include <istream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {

/**
 * @defgroup ov_runtime_cpp_prop_api Device properties
 * @ingroup ov_runtime_cpp_api
 */

/**
 * @brief Enum to define property value mutability
 */
enum class PropertyMutability {
    RO,  //!< Read-only property values can not be passed as input parameter
    RW,  //!< Read/Write property key may change readability in runtime
    WO,  //!< Write-only property can not be read
};

/** @cond INTERNAL */
namespace util {
struct PropertyTag {};

template <typename... Args>
struct StringAny;

template <typename T, typename... Args>
struct StringAny<T, Args...> {
    constexpr static const bool value =
        std::is_convertible<T, std::pair<std::string, ov::Any>>::value && StringAny<Args...>::value;
};

template <typename T>
struct StringAny<T> {
    constexpr static const bool value = std::is_convertible<T, std::pair<std::string, ov::Any>>::value;
};

template <typename T, typename... Args>
using EnableIfAllStringAny = typename std::enable_if<StringAny<Args...>::value, T>::type;

/**
 * @brief This class is used to bind property name with property type
 * @tparam T type of value used to pass or get property
 */
template <typename T, PropertyMutability mutability_ = PropertyMutability::RW>
struct BaseProperty : public PropertyTag {
    using value_type = T;                                  //!< Property type
    constexpr static const auto mutability = mutability_;  //!< Property readability

    /**
     * @brief Constructs property access variable
     * @param str_ property name
     */
    constexpr BaseProperty(const char* name_) : _name{name_} {}

    /**
     * @brief return property name
     * @return Pointer to const string key representation
     */
    const char* name() const {
        return _name;
    }

    /**
     * @brief compares property name
     * @return true if string is the same
     */
    bool operator==(const std::string& str) const {
        return _name == str;
    }

    /**
     * @brief compares property name
     * @return true if string is the same
     */
    friend bool operator==(const std::string& str, const BaseProperty<T, mutability_>& property) {
        return property == str;
    }

private:
    const char* _name = nullptr;
};

template <typename T, PropertyMutability M>
inline std::ostream& operator<<(std::ostream& os, const BaseProperty<T, M>& property) {
    return os << property.name();
}
}  // namespace util
/** @endcond */

/**
 * @brief This class is used to bind property name with value type
 * @tparam T type of value used to set or get property
 */
template <typename T, PropertyMutability mutability_ = PropertyMutability::RW>
class Property : public util::BaseProperty<T, mutability_> {
    template <typename V>
    struct Forward {
        template <typename U,
                  typename std::enable_if<std::is_same<typename std::decay<U>::type, std::string>::value &&
                                              std::is_convertible<V, std::string>::value,
                                          bool>::type = true>
        explicit operator U() {
            return value;
        }

        template <typename U,
                  typename std::enable_if<std::is_same<typename std::decay<U>::type, std::string>::value &&
                                              !std::is_convertible<V, std::string>::value,
                                          bool>::type = true>
        explicit operator U() {
            return Any{value}.as<U>();
        }

        template <typename U,
                  typename std::enable_if<!std::is_same<typename std::decay<U>::type, std::string>::value &&
                                              std::is_convertible<V, std::string>::value,
                                          bool>::type = true>
        explicit operator U() {
            return Any{value}.as<U>();
        }

        template <typename U,
                  typename std::enable_if<!std::is_same<typename std::decay<U>::type, std::string>::value &&
                                              !std::is_convertible<V, std::string>::value,
                                          bool>::type = true>
        explicit operator U() {
            return value;
        }

        V&& value;
    };

public:
    using util::BaseProperty<T, mutability_>::BaseProperty;
    /**
     * @brief Constructs property
     * @tparam Args property constructor arguments types
     * @param args property constructor arguments
     * @return Pair of name and type erased value.
     */
    template <typename... Args>
    inline std::pair<std::string, Any> operator()(Args&&... args) const {
        return {this->name(), Any::make<T>(Forward<Args>{std::forward<Args>(args)}...)};
    }
};

/**
 * @brief This class is used to return property name and its mutability attribute
 */
struct PropertyName : public std::string {
    using std::string::string;

    /**
     * @brief Constructs property name object
     * @param str property name
     * @param mutability property mutability
     */
    PropertyName(const std::string& str, PropertyMutability mutability = PropertyMutability::RW)
        : std::string{str},
          _mutability{mutability} {}

    template <class T, typename std::enable_if<std::is_base_of<ov::util::PropertyTag, T>::value, bool>::type = true>
    PropertyName(const T& property) : std::string{property.name()},
                                      _mutability{property.mutability} {}

    /**
     * @brief check property mutability
     * @return true if property is mutable
     */
    bool is_mutable() const {
        return _mutability == PropertyMutability::RW;
    }

private:
    PropertyMutability _mutability = PropertyMutability::RW;
};

/**
 * @brief This class is used to bind read-only property name with value type
 * @tparam T type of value used to pass or get property
 */
template <typename T>
struct Property<T, PropertyMutability::RO> : public util::BaseProperty<T, PropertyMutability::RO> {
    using util::BaseProperty<T, PropertyMutability::RO>::BaseProperty;
};

/**
 * @brief Read-only property to get a std::vector<PropertyName> of supported read-only properties.
 * This can be used as a compiled model property as well.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::vector<PropertyName>, PropertyMutability::RO> supported_properties{
    "SUPPORTED_PROPERTIES"};

/**
 * @brief Read-only property to get a std::vector<std::string> of available device IDs
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RO> available_devices{"AVAILABLE_DEVICES"};

/**
 * @brief Read-only property to get a name of name of a model
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string, PropertyMutability::RO> model_name{"NETWORK_NAME"};

/**
 * @brief Read-only property to get an unsigned integer value of optimal number of compiled model infer requests.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<uint32_t, PropertyMutability::RO> optimal_number_of_infer_requests{
    "OPTIMAL_NUMBER_OF_INFER_REQUESTS"};

/**
 * @brief Namespace with hint properties
 */
namespace hint {

/**
 * @brief Hint for device to use specified precision for inference
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<element::Type, PropertyMutability::RW> inference_precision{"INFERENCE_PRECISION_HINT"};

/**
 * @brief Enum to define possible priorities hints
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class Priority {
    LOW = 0,           //!<  Low priority
    MEDIUM = 1,        //!<  Medium priority
    HIGH = 2,          //!<  High priority
    DEFAULT = MEDIUM,  //!<  Default priority is MEDIUM
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Priority& priority) {
    switch (priority) {
    case Priority::LOW:
        return os << "LOW";
    case Priority::MEDIUM:
        return os << "MEDIUM";
    case Priority::HIGH:
        return os << "HIGH";
    default:
        OPENVINO_THROW("Unsupported model priority value");
    }
}

inline std::istream& operator>>(std::istream& is, Priority& priority) {
    std::string str;
    is >> str;
    if (str == "LOW") {
        priority = Priority::LOW;
    } else if (str == "MEDIUM") {
        priority = Priority::MEDIUM;
    } else if (str == "HIGH") {
        priority = Priority::HIGH;
    } else if (str == "DEFAULT") {
        priority = Priority::DEFAULT;
    } else {
        OPENVINO_THROW("Unsupported model priority: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief High-level OpenVINO model priority hint
 * Defines what model should be provided with more performant bounded resource first
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<Priority> model_priority{"MODEL_PRIORITY"};

/**
 * @brief Enum to define possible performance mode hints
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class PerformanceMode {
    LATENCY = 1,                //!<  Optimize for latency
    THROUGHPUT = 2,             //!<  Optimize for throughput
    CUMULATIVE_THROUGHPUT = 3,  //!<  Optimize for cumulative throughput
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const PerformanceMode& performance_mode) {
    switch (performance_mode) {
    case PerformanceMode::LATENCY:
        return os << "LATENCY";
    case PerformanceMode::THROUGHPUT:
        return os << "THROUGHPUT";
    case PerformanceMode::CUMULATIVE_THROUGHPUT:
        return os << "CUMULATIVE_THROUGHPUT";
    default:
        OPENVINO_THROW("Unsupported performance mode hint");
    }
}

inline std::istream& operator>>(std::istream& is, PerformanceMode& performance_mode) {
    std::string str;
    is >> str;
    if (str == "LATENCY") {
        performance_mode = PerformanceMode::LATENCY;
    } else if (str == "THROUGHPUT") {
        performance_mode = PerformanceMode::THROUGHPUT;
    } else if (str == "CUMULATIVE_THROUGHPUT") {
        performance_mode = PerformanceMode::CUMULATIVE_THROUGHPUT;
    } else {
        OPENVINO_THROW("Unsupported performance mode: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief High-level OpenVINO Performance Hints
 * unlike low-level properties that are individual (per-device), the hints are something that every device accepts
 * and turns into device-specific settings
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<PerformanceMode> performance_mode{"PERFORMANCE_HINT"};

/**
 * @enum       SchedulingCoreType
 * @brief      This enum contains definition of core type can be used for CPU tasks on different devices.
 */
enum class SchedulingCoreType {
    ANY_CORE = 0,    //!<  Any processors can be used.
    PCORE_ONLY = 1,  //!<  Only processors of performance-cores can be used.
    ECORE_ONLY = 2,  //!<  Only processors of efficient-cores can be used.
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SchedulingCoreType& core_type) {
    switch (core_type) {
    case SchedulingCoreType::ANY_CORE:
        return os << "ANY_CORE";
    case SchedulingCoreType::PCORE_ONLY:
        return os << "PCORE_ONLY";
    case SchedulingCoreType::ECORE_ONLY:
        return os << "ECORE_ONLY";
    default:
        OPENVINO_THROW("Unsupported core type!");
    }
}

inline std::istream& operator>>(std::istream& is, SchedulingCoreType& core_type) {
    std::string str;
    is >> str;
    if (str == "ANY_CORE") {
        core_type = SchedulingCoreType::ANY_CORE;
    } else if (str == "PCORE_ONLY") {
        core_type = SchedulingCoreType::PCORE_ONLY;
    } else if (str == "ECORE_ONLY") {
        core_type = SchedulingCoreType::ECORE_ONLY;
    } else {
        OPENVINO_THROW("Unsupported core type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief This property defines CPU core type which can be used during inference.
 * @ingroup ov_runtime_cpp_prop_api
 *
 * Developer can use this property to select specific CPU cores for inference. Please refer SchedulingCoreType for
 * all definition of core type.
 *
 * The following code is an example to only use efficient-cores for inference on hybrid CPU. If user sets this
 * configuration on a platform with only performance-cores, CPU inference will still run on the performance-cores.
 *
 * @code
 * ie.set_property(ov::hint::scheduling_core_type(ov::hint::SchedulingCoreType::ECORE_ONLY));
 * @endcode
 */
static constexpr Property<SchedulingCoreType> scheduling_core_type{"SCHEDULING_CORE_TYPE"};

enum class ModelDistributionPolicy {
    TENSOR_PARALLEL = 0,    // Distribute tensor to multiple sockets/devices during model compilation. At inference
                            // time, sockets/devices process individual tensor in parallel.
    PIPELINE_PARALLEL = 1,  // Distribute tensor to multiple sockets/devices during model compilation. At inference
                            // time, sockets/devices process individual tensor one by one. And each socket/device
                            // processes a portion of a different tensor in parallel.
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ModelDistributionPolicy& stream_mode) {
    switch (stream_mode) {
    case ModelDistributionPolicy::TENSOR_PARALLEL:
        return os << "TENSOR_PARALLEL";
    case ModelDistributionPolicy::PIPELINE_PARALLEL:
        return os << "PIPELINE_PARALLEL";
    default:
        OPENVINO_THROW("Unsupported model distribution policy!");
    }
}

inline std::istream& operator>>(std::istream& is, ModelDistributionPolicy& stream_mode) {
    std::string str;
    is >> str;
    if (str == "TENSOR_PARALLEL") {
        stream_mode = ModelDistributionPolicy::TENSOR_PARALLEL;
    } else if (str == "PIPELINE_PARALLEL") {
        stream_mode = ModelDistributionPolicy::PIPELINE_PARALLEL;
    } else {
        OPENVINO_THROW("Unsupported model distribution policy: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief This property defines model distribution policy for inference with multiple sockets/devices.
 * @ingroup ov_runtime_cpp_prop_api
 * This property can be used to select model distribution policy between execution units (e.g. between CPU sockets/NUMA
 * nodes or between different GPUs).
 * -- TENSOR_PARALLEL   : Distribute tensor to multiple sockets/devices during model compilation. At inference time,
 *                        sockets/devices process individual tensor in parallel.
 * -- PIPELINE_PARALLEL : Distribute tensor to multiple sockets/devices during model compilation. At inference time,
 *                        sockets/devices process individual tensor one by one. And each socket/device processes a
 *                        portion of a different tensor in parallel.
 *
 * The following code is an example how TENSOR_PARALLEL or PIPELINE_PARALLEL model distribution policy might be enabled.
 *
 * @code
 * ie.set_property(ov::hint::model_distribution_policy({ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL}));
 * ie.set_property(ov::hint::model_distribution_policy({ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL}));
 * @endcode
 */
static constexpr Property<std::set<ModelDistributionPolicy>> model_distribution_policy{"MODEL_DISTRIBUTION_POLICY"};

/**
 * @brief This property allows CPU pinning during inference.
 * @ingroup ov_runtime_cpp_prop_api
 *
 * Developer can use this property to enable or disable CPU pinning during inference on Windows and Linux. MacOS
 * does not support CPU pinning, and this property is always disabled. If user does not explicitly set value for
 * this property, OpenVINO may choose any desired value based on internal logic.
 *
 * The following is an example of CPU fixed behavior on a hybrid CPU (8 performance cores and 16 efficiency cores).
 * For stream with 4 threads on performance cores, if CPU pinning is enabled, each thread is bound to a specific
 * performance core. If CPU pinning is disabled, OS will schedule 4 threads on performance cores only.
 * For stream with 24 threads on all cores, if CPU pinning is enabled, each thread is bound to a specific
 * performance core. If CPU pinning is disabled, OS will schedule 24 threads on both performance cores and efficiency
 * cores.
 *
 * The following code is example to use this property.
 *
 * @code
 * ie.set_property(ov::hint::enable_cpu_pinning(true));
 * ie.set_property(ov::hint::enable_cpu_pinning(false));
 * @endcode
 */
static constexpr Property<bool> enable_cpu_pinning{"ENABLE_CPU_PINNING"};

/**
 * @brief This property define if using hyper threading during inference.
 * @ingroup ov_runtime_cpp_prop_api
 *
 * Developer can use this property to use or not use CPU pinning during inference. If user does not explicitly set
 * value for this property, OpenVINO may choose any desired value based on internal logic.
 *
 * The following code is example to use this property.
 *
 * @code
 * ie.set_property(ov::hint::enable_hyper_threading(true));
 * ie.set_property(ov::hint::enable_hyper_threading(false));
 * @endcode
 */
static constexpr Property<bool> enable_hyper_threading{"ENABLE_HYPER_THREADING"};

/**
 * @brief (Optional) property that backs the (above) Performance Hints
 * by giving additional information on how many inference requests the application will be keeping in flight
 * usually this value comes from the actual use-case (e.g. number of video-cameras, or other sources of inputs)
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<uint32_t> num_requests{"PERFORMANCE_HINT_NUM_REQUESTS"};

/**
 * @brief This key identifies shared pointer to the ov::Model, required for some properties (ov::max_batch_size and
 * ov::optimal_batch_size)
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::shared_ptr<ov::Model>> model{"MODEL_PTR"};

/**
 * @brief Special key for auto batching feature configuration. Enabled by default
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<bool, PropertyMutability::RW> allow_auto_batching{"ALLOW_AUTO_BATCHING"};

/**
 * @brief Enum to define possible execution mode hints
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class ExecutionMode {
    PERFORMANCE = 1,  //!<  Optimize for max performance, may apply properties which slightly affect accuracy
    ACCURACY = 2,     //!<  Optimize for max accuracy
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ExecutionMode& mode) {
    switch (mode) {
    case ExecutionMode::PERFORMANCE:
        return os << "PERFORMANCE";
    case ExecutionMode::ACCURACY:
        return os << "ACCURACY";
    default:
        OPENVINO_THROW("Unsupported execution mode hint");
    }
}

inline std::istream& operator>>(std::istream& is, ExecutionMode& mode) {
    std::string str;
    is >> str;
    if (str == "PERFORMANCE") {
        mode = ExecutionMode::PERFORMANCE;
    } else if (str == "ACCURACY") {
        mode = ExecutionMode::ACCURACY;
    } else {
        OPENVINO_THROW("Unsupported execution mode: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief High-level OpenVINO Execution hint
 * unlike low-level properties that are individual (per-device), the hints are something that every device accepts
 * and turns into device-specific settings
 * Execution mode hint controls preferred optimization targets (performance or accuracy) for given model
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<ExecutionMode> execution_mode{"EXECUTION_MODE_HINT"};

/**
 * @brief This property defines group size for dynamic quantization optimization
 * @ingroup ov_runtime_cpp_prop_api
 *
 * Dynamic quantization optimization provides an ability to get performance benefit from int8 compute.
 * In contrast with static quantization dynamic approach assumes activations are quantized during inference.
 * Despite the fact dynamic quantization has some runtime overheads, it might provide better accuracy metrics.
 * This property defines granularity (aka block size) for dynamic quantization algorithms. Lower group size values
 * might result in better accuracy, but the drawback is worse performance. Group size equal 0 means dynamic
 * quantization optimization is disabled.
 */
static constexpr Property<uint64_t, PropertyMutability::RW> dynamic_quantization_group_size{
    "DYNAMIC_QUANTIZATION_GROUP_SIZE"};

/**
 * @brief Hint for device to use specified precision for kv cache compression
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<element::Type, PropertyMutability::RW> kv_cache_precision{"KV_CACHE_PRECISION"};

/**
 * @brief This property scales down activations to prevent overflows when inference precision is f16.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<float, PropertyMutability::RW> activations_scale_factor{"ACTIVATIONS_SCALE_FACTOR"};

}  // namespace hint

/**
 * @brief The name for setting performance counters option.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<bool> enable_profiling{"PERF_COUNT"};

/**
 * @brief Namespace with log level property and its possible values
 */
namespace log {

#ifdef DEBUG
#    define _OV_WAS_DEBUG DEBUG
#    undef DEBUG
#endif

/**
 * @brief Enum to define possible log levels
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class Level {
    NO = -1,      //!< disable any logging
    ERR = 0,      //!< error events that might still allow the application to continue running
    WARNING = 1,  //!< potentially harmful situations which may further lead to ERROR
    INFO = 2,     //!< informational messages that display the progress of the application at coarse-grained level
    DEBUG = 3,    //!< fine-grained events that are most useful to debug an application.
    TRACE = 4,    //!< finer-grained informational events than the DEBUG
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Level& level) {
    switch (level) {
    case Level::NO:
        return os << "LOG_NONE";
    case Level::ERR:
        return os << "LOG_ERROR";
    case Level::WARNING:
        return os << "LOG_WARNING";
    case Level::INFO:
        return os << "LOG_INFO";
    case Level::DEBUG:
        return os << "LOG_DEBUG";
    case Level::TRACE:
        return os << "LOG_TRACE";
    default:
        OPENVINO_THROW("Unsupported log level");
    }
}

inline std::istream& operator>>(std::istream& is, Level& level) {
    std::string str;
    is >> str;
    if (str == "LOG_NONE") {
        level = Level::NO;
    } else if (str == "LOG_ERROR") {
        level = Level::ERR;
    } else if (str == "LOG_WARNING") {
        level = Level::WARNING;
    } else if (str == "LOG_INFO") {
        level = Level::INFO;
    } else if (str == "LOG_DEBUG") {
        level = Level::DEBUG;
    } else if (str == "LOG_TRACE") {
        level = Level::TRACE;
    } else {
        OPENVINO_THROW("Unsupported log level: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief the property for setting desirable log level.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<Level> level{"LOG_LEVEL"};

#ifdef _OV_WAS_DEBUG
#    define DEBUG _OV_WAS_DEBUG
#    undef _OV_WAS_DEBUG
#endif
}  // namespace log

/**
 * @brief This property defines the directory which will be used to store any data cached by plugins.
 * @ingroup ov_runtime_cpp_prop_api
 *
 * The underlying cache structure is not defined and might differ between OpenVINO releases
 * Cached data might be platform / device specific and might be invalid after OpenVINO version change
 * If this property is not specified or value is empty string, then caching is disabled.
 * The property might enable caching for the plugin using the following code:
 *
 * @code
 * ie.set_property("GPU", ov::cache_dir("cache/")); // enables cache for GPU plugin
 * @endcode
 *
 * The following code enables caching of compiled network blobs for devices where import/export is supported
 *
 * @code
 * ie.set_property(ov::cache_dir("cache/")); // enables models cache
 * @endcode
 */
static constexpr Property<std::string> cache_dir{"CACHE_DIR"};

/**
 * @brief Read-only property to notify user that compiled model was loaded from the cache
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<bool, PropertyMutability::RO> loaded_from_cache{"LOADED_FROM_CACHE"};

/**
 * @brief Enum to define possible workload types
 *
 * Workload type represents the execution priority for an inference.
 *
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class WorkloadType {
    DEFAULT = 0,    // Default execution priority
    EFFICIENT = 1,  // Lower execution priority
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const WorkloadType& mode) {
    switch (mode) {
    case WorkloadType::DEFAULT:
        return os << "DEFAULT";
    case WorkloadType::EFFICIENT:
        return os << "EFFICIENT";
    default:
        OPENVINO_THROW("Unsupported workload type");
    }
}

inline std::istream& operator>>(std::istream& is, WorkloadType& mode) {
    std::string str;
    is >> str;
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    if (str == "default") {
        mode = WorkloadType::DEFAULT;
    } else if (str == "efficient") {
        mode = WorkloadType::EFFICIENT;
    } else {
        OPENVINO_THROW("Unsupported workload type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief Read-write property to select in which mode the workload will be executed
 * This is only supported by NPU.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<WorkloadType, PropertyMutability::RW> workload_type{"WORKLOAD_TYPE"};

/**
 * @brief Enum to define possible cache mode
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class CacheMode {
    OPTIMIZE_SIZE = 0,   //!< smaller cache size
    OPTIMIZE_SPEED = 1,  //!< faster loading time
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const CacheMode& mode) {
    switch (mode) {
    case CacheMode::OPTIMIZE_SIZE:
        return os << "optimize_size";
    case CacheMode::OPTIMIZE_SPEED:
        return os << "optimize_speed";
    default:
        OPENVINO_THROW("Unsupported cache mode");
    }
}

inline std::istream& operator>>(std::istream& is, CacheMode& mode) {
    std::string str;
    is >> str;
    if (str == "OPTIMIZE_SIZE" || str == "optimize_size") {
        mode = CacheMode::OPTIMIZE_SIZE;
    } else if (str == "OPTIMIZE_SPEED" || str == "optimize_speed") {
        mode = CacheMode::OPTIMIZE_SPEED;
    } else {
        OPENVINO_THROW("Unsupported cache mode: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief Read-write property to select the cache mode between optimize_size and optimize_speed.
 * If optimize_speed is selected(default), loading time will decrease but the cache file size will increase.
 * If optimize_size is selected, smaller cache files will be created.
 * This is only supported from GPU.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<CacheMode, PropertyMutability::RW> cache_mode{"CACHE_MODE"};

struct EncryptionCallbacks {
    std::function<std::string(const std::string&)> encrypt;
    std::function<std::string(const std::string&)> decrypt;
};

/**
 * @brief Write-only property to set encryption/decryption function for saving/loading model cache.
 * If cache_encryption_callbacks is set, the model topology will be encrypted when saving to the cache and decrypted
 * when loading from the cache. This property is set in core.compile_model only.
 * - First value of the struct is encryption function.
 * - Second value of the struct is decryption function.
 * @note GPU Plugin: encrypts whole blob, not only model structure. Only used when ov::cache_mode property is set to
 * "OPTIMIZE_SIZE".
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<EncryptionCallbacks, PropertyMutability::WO> cache_encryption_callbacks{
    "CACHE_ENCRYPTION_CALLBACKS"};

/**
 * @brief Read-only property to provide information about a range for streams on platforms where streams are supported.
 * @ingroup ov_runtime_cpp_prop_api
 *
 * Property returns a value of std::tuple<unsigned int, unsigned int> type, where:
 *  - First value is bottom bound.
 *  - Second value is upper bound.
 */
static constexpr Property<std::tuple<unsigned int, unsigned int>, PropertyMutability::RO> range_for_streams{
    "RANGE_FOR_STREAMS"};

/**
 * @brief Read-only property to query information optimal batch size for the given device and the network
 * @ingroup ov_runtime_cpp_prop_api
 *
 * Property returns a value of unsigned int type,
 * Returns optimal batch size for a given network on the given device. The returned value is aligned to power of 2.
 * Also, ov::hint::model is the required option for this metric since the optimal batch size depends on the model,
 * so if the ov::hint::model is not given, the result of the metric is always 1.
 * For the GPU the metric is queried automatically whenever the OpenVINO performance hint for the throughput is used,
 * so that the result (>1) governs the automatic batching (transparently to the application).
 * The automatic batching can be disabled with ALLOW_AUTO_BATCHING set to NO
 */
static constexpr Property<unsigned int, PropertyMutability::RO> optimal_batch_size{"OPTIMAL_BATCH_SIZE"};

/**
 * @brief Read-only property to get maximum batch size which does not cause performance degradation due to memory swap
 * impact.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<uint32_t, PropertyMutability::RO> max_batch_size{"MAX_BATCH_SIZE"};

/**
 * @brief Read-write property to set the timeout used to collect the inputs for the auto-batching
 * impact.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<uint32_t, PropertyMutability::RW> auto_batch_timeout{"AUTO_BATCH_TIMEOUT"};

/**
 * @brief Read-only property to provide a hint for a range for number of async infer requests. If device supports
 * streams, the metric provides range for number of IRs per stream.
 * @ingroup ov_runtime_cpp_prop_api
 *
 * Property returns a value of std::tuple<unsigned int, unsigned int, unsigned int> type, where:
 *  - First value is bottom bound.
 *  - Second value is upper bound.
 *  - Third value is step inside this range.
 */
static constexpr Property<std::tuple<unsigned int, unsigned int, unsigned int>, PropertyMutability::RO>
    range_for_async_infer_requests{"RANGE_FOR_ASYNC_INFER_REQUESTS"};

/**
 * @brief Read-write property to set whether force terminate tbb when ov core destruction
 * value type: boolean
 *   - True explicitly terminate tbb when ov core destruction
 *   - False will not involve additional tbb operations when core destruction
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<bool, PropertyMutability::RW> force_tbb_terminate{"FORCE_TBB_TERMINATE"};

/**
 * @brief Read-write property to configure `mmap()` use for model read. Enabled by default.
 * For the moment only IR Frontend supports the property.
 *
 * value type: boolean
 *   - True enable `mmap()` use and map model
 *   - False disable `mmap()` use and read model
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<bool, PropertyMutability::RW> enable_mmap{"ENABLE_MMAP"};

/**
 * @brief Namespace with device properties
 */
namespace device {
/**
 * @brief the property for setting of required device to execute on
 * values: device id starts from "0" - first device, "1" - second device, etc
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string> id{"DEVICE_ID"};

/**
 * @brief Type for device Priorities config option, with comma-separated devices listed in the desired priority
 * @ingroup ov_runtime_cpp_prop_api
 */
struct Priorities : public Property<std::string> {
private:
    template <typename H, typename... T>
    static inline std::string concat(const H& head, T&&... tail) {
        return head + std::string{','} + concat(std::forward<T>(tail)...);
    }

    template <typename H>
    static inline std::string concat(const H& head) {
        return head;
    }

public:
    using Property<std::string>::Property;

    /**
     * @brief Constructs device priorities
     * @tparam Args property constructor arguments types
     * @param args property constructor arguments
     * @return Pair of name and type erased value.
     */
    template <typename... Args>
    inline std::pair<std::string, Any> operator()(Args&&... args) const {
        return {name(), Any{concat(std::forward<Args>(args)...)}};
    }
};

/**
 * @brief Device Priorities config option, with comma-separated devices listed in the desired priority
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Priorities priorities{"MULTI_DEVICE_PRIORITIES"};

/**
 * @brief Type for property to pass set of properties to specified device
 * @ingroup ov_runtime_cpp_prop_api
 */

struct Properties : public Property<std::map<std::string, std::map<std::string, Any>>> {
    using Property<std::map<std::string, std::map<std::string, Any>>>::Property;

    /**
     * @brief Constructs property
     * @param configs set of property values with names
     * @return Pair of string key representation and type erased property value.
     */
    inline std::pair<std::string, Any> operator()(const AnyMap& config) const {
        return {name(), config};
    }

    /**
     * @brief Constructs property
     * @param device_name device plugin alias
     * @param config set of property values with names
     * @return Pair of string key representation and type erased property value.
     */
    inline std::pair<std::string, Any> operator()(const std::string& device_name, const AnyMap& config) const {
        return {name() + std::string("_") + device_name, config};
    }

    /**
     * @brief Constructs property
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param device_name device plugin alias
     * @param configs Optional pack of pairs: (config parameter name, config parameter value)
     * @return Pair of string key representation and type erased property value.
     */
    template <typename... Properties>
    inline util::EnableIfAllStringAny<std::pair<std::string, Any>, Properties...> operator()(
        const std::string& device_name,
        Properties&&... configs) const {
        return {name() + std::string("_") + device_name,
                AnyMap{std::pair<std::string, Any>{std::forward<Properties>(configs)}...}};
    }
};

/**
 * @brief Property to pass set of property values to specified device
 * @ingroup ov_runtime_cpp_prop_api
 * Usage Example:
 * @code
 * core.compile_model("HETERO"
 *     ov::device::priorities("GPU", "CPU"),
 *     ov::device::properties("CPU", ov::enable_profiling(true)),
 *     ov::device::properties("GPU", ov::enable_profiling(false)));
 * @endcode
 */
static constexpr Properties properties{"DEVICE_PROPERTIES"};

/**
 * @brief Read-only property to get a std::string value representing a full device name.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string, PropertyMutability::RO> full_name{"FULL_DEVICE_NAME"};

/**
 * @brief Read-only property which defines the device architecture.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string, PropertyMutability::RO> architecture{"DEVICE_ARCHITECTURE"};

/**
 * @brief Structure which defines format of UUID.
 * @ingroup ov_runtime_cpp_prop_api
 */
struct UUID {
    static const uint64_t MAX_UUID_SIZE = 16;  //!< Max size of uuid array (128 bits)
    std::array<uint8_t, MAX_UUID_SIZE> uuid;   //!< Array with uuid for a device
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const UUID& device_uuid) {
    std::stringstream s;
    for (auto& c : device_uuid.uuid) {
        s << std::hex << std::setw(2) << std::setfill('0') << +c;
    }
    return os << s.str();
}

inline std::istream& operator>>(std::istream& is, UUID& device_uuid) {
    std::string s;
    auto flags = is.flags();
    for (size_t i = 0; i < UUID::MAX_UUID_SIZE; i++) {
        is >> std::setw(2) >> s;
        std::istringstream ss2(s);
        int val;
        ss2 >> std::hex >> val;
        device_uuid.uuid[i] = static_cast<uint8_t>(val);
    }
    is.flags(flags);
    return is;
}
/** @endcond */

/**
 * @brief Read-only property which defines the UUID of the device.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<UUID, PropertyMutability::RO> uuid{"DEVICE_UUID"};

/**
 * @brief Structure which defines format of LUID.
 * @ingroup ov_runtime_cpp_prop_api
 */
struct LUID {
    static const uint64_t MAX_LUID_SIZE = 8;  //!< Max size of luid array (64 bits)
    std::array<uint8_t, MAX_LUID_SIZE> luid;  //!< Array with luid for a device
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const LUID& device_luid) {
    std::stringstream s;
    for (auto& c : device_luid.luid) {
        s << std::hex << std::setw(2) << std::setfill('0') << +c;
    }
    return os << s.str();
}

inline std::istream& operator>>(std::istream& is, LUID& device_luid) {
    std::string s;
    auto flags = is.flags();
    for (size_t i = 0; i < LUID::MAX_LUID_SIZE; i++) {
        is >> std::setw(2) >> s;
        std::istringstream ss2(s);
        int val;
        ss2 >> std::hex >> val;
        device_luid.luid[i] = static_cast<uint8_t>(val);
    }
    is.flags(flags);
    return is;
}
/** @endcond */

/**
 * @brief Read-only property which defines the LUID of the device.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<LUID, PropertyMutability::RO> luid{"DEVICE_LUID"};

/**
 * @brief Enum to define possible device types
 * @ingroup ov_runtime_cpp_prop_api
 */
enum class Type {
    INTEGRATED = 0,  //!<  Device is integrated into host system
    DISCRETE = 1,    //!<  Device is not integrated into host system
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Type& device_type) {
    switch (device_type) {
    case Type::DISCRETE:
        return os << "discrete";
    case Type::INTEGRATED:
        return os << "integrated";
    default:
        OPENVINO_THROW("Unsupported device type");
    }
}

inline std::istream& operator>>(std::istream& is, Type& device_type) {
    std::string str;
    is >> str;
    if (str == "discrete") {
        device_type = Type::DISCRETE;
    } else if (str == "integrated") {
        device_type = Type::INTEGRATED;
    } else {
        OPENVINO_THROW("Unsupported device type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief Read-only property to get a type of device. See Type enum definition for possible return values
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<Type, PropertyMutability::RO> type{"DEVICE_TYPE"};

/**
 * @brief Read-only property which defines Giga OPS per second count (GFLOPS or GIOPS) for a set of precisions
 * supported by specified device
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::map<element::Type, float>, PropertyMutability::RO> gops{"DEVICE_GOPS"};

/**
 * @brief Structure to store PCI bus information of device (Domain/Bus/Device/Function)
 * @ingroup ov_runtime_cpp_prop_api
 */
struct PCIInfo {
    /**
     * @brief PCI domain ID
     */
    uint32_t domain;
    /**
     * @brief PCI bus ID
     */
    uint32_t bus;
    /**
     * @brief PCI device ID
     */
    uint32_t device;
    /**
     * @brief PCI function ID
     */
    uint32_t function;
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const PCIInfo& pci_info) {
    return os << "{domain: " << pci_info.domain << " bus: " << pci_info.bus << " device: 0x" << std::hex
              << pci_info.device << " function: " << std::dec << pci_info.function << "}";
}

inline std::istream& operator>>(std::istream& is, PCIInfo& pci_info) {
    std::string delim;
    if (!(is >> delim >> pci_info.domain >> delim >> pci_info.bus >> delim >> std::hex >> pci_info.device >> delim >>
          std::dec >> pci_info.function)) {
        OPENVINO_THROW("Could not deserialize PCIInfo. Invalid format!");
    }
    return is;
}
/** @endcond */

/**
 * @brief Read-only property to get PCI bus information of device. See PCIInfo struct definition for details
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<PCIInfo, PropertyMutability::RO> pci_info{"DEVICE_PCI_INFO"};

/**
 * @brief Read-only property to get a float of device thermal
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<float, PropertyMutability::RO> thermal{"DEVICE_THERMAL"};

/**
 * @brief Read-only property to get a std::vector<std::string> of capabilities options per device.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RO> capabilities{"OPTIMIZATION_CAPABILITIES"};

/**
 * @brief Namespace with possible values for ov::device::capabilities property
 */
namespace capability {
/**
 * @addtogroup ov_runtime_cpp_prop_api
 * @{
 */
constexpr static const auto FP32 = "FP32";                    //!< Device supports fp32 inference
constexpr static const auto BF16 = "BF16";                    //!< Device supports bf16 inference
constexpr static const auto FP16 = "FP16";                    //!< Device supports fp16 inference
constexpr static const auto INT8 = "INT8";                    //!< Device supports int8 inference
constexpr static const auto INT16 = "INT16";                  //!< Device supports int16 inference
constexpr static const auto BIN = "BIN";                      //!< Device supports binary inference
constexpr static const auto WINOGRAD = "WINOGRAD";            //!< Device supports winograd optimization
constexpr static const auto EXPORT_IMPORT = "EXPORT_IMPORT";  //!< Device supports compiled model export and import
/** @}*/
}  // namespace capability

}  // namespace device

/**
 * @brief Namespace for streams in streams executor
 */
namespace streams {
/**
 * @brief Class to represent number of streams in streams executor
 * @ingroup ov_runtime_cpp_prop_api
 */
struct Num {
    using Base = std::tuple<int32_t>;  //!< NumStreams is representable as int32_t

    constexpr Num() : num{-1} {};

    constexpr Num(const int32_t num_) : num{num_} {}

    constexpr operator int32_t() const {
        return num;
    }

    int32_t num = 0;
};

/**
 * @brief The number of executor logical partitions
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<Num, PropertyMutability::RW> num{"NUM_STREAMS"};

/**
 * @brief Creates bare minimum of streams to improve the performance
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Num AUTO{-1};

/**
 * @brief Creates as many streams as needed to accommodate NUMA and avoid associated penalties
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Num NUMA{-2};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const Num& num_val) {
    switch (num_val) {
    case AUTO:
        return os << "AUTO";
    case NUMA:
        return os << "NUMA";
    default:
        return os << num_val.num;
    }
}

inline std::istream& operator>>(std::istream& is, Num& num_val) {
    std::string str;
    is >> str;
    if (str == "AUTO") {
        num_val = AUTO;
    } else if (str == "NUMA") {
        num_val = NUMA;
    } else {
        try {
            num_val = {std::stoi(str)};
        } catch (const std::exception& e) {
            OPENVINO_THROW("Could not read number of streams from str: ", str, "; ", e.what());
        }
    }
    return is;
}
/** @endcond */
}  // namespace streams

/**
 * @brief The number of executor logical partitions
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<streams::Num, PropertyMutability::RW> num_streams{"NUM_STREAMS"};

/**
 * @brief Maximum number of threads that can be used for inference tasks
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<int32_t, PropertyMutability::RW> inference_num_threads{"INFERENCE_NUM_THREADS"};

/**
 * @brief Maximum number of threads that can be used for compilation tasks
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<int32_t, PropertyMutability::RW> compilation_num_threads{"COMPILATION_NUM_THREADS"};

/**
 * @brief The devices that the inference task been executed.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RO> execution_devices{"EXECUTION_DEVICES"};

/**
 * @brief Path to the file with model's weights.
 *
 * @note This property is used for weightless caching. Only used when ov::CacheMode Property is set to "OPTIMIZE_SIZE".
 */
static constexpr Property<std::string, PropertyMutability::RW> weights_path{"WEIGHTS_PATH"};

/**
 * @brief The precision of key cache compression
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<element::Type, PropertyMutability::RW> key_cache_precision{"KEY_CACHE_PRECISION"};

/**
 * @brief The precision of value cache compression
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<element::Type, PropertyMutability::RW> value_cache_precision{"VALUE_CACHE_PRECISION"};

/**
 * @brief The group_size of key cache compression
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<uint64_t, PropertyMutability::RW> key_cache_group_size{"KEY_CACHE_GROUP_SIZE"};

/**
 * @brief The group_size of value cache compression
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<uint64_t, PropertyMutability::RW> value_cache_group_size{"VALUE_CACHE_GROUP_SIZE"};
}  // namespace ov
