// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the OpenVINO Runtime Core class C++ API.
 *
 * @file openvino/runtime/core.hpp
 */
#pragma once

#include <istream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/extension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/op_extension.hpp"
#include "openvino/core/version.hpp"
#include "openvino/op/op.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/tensor.hpp"

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
#    include <filesystem>
#endif

namespace ov {

/**
 * @brief This class represents an OpenVINO runtime Core entity.
 * @ingroup ov_runtime_cpp_api
 * User applications can create several Core class instances, but in this case the underlying plugins
 * are created multiple times and not shared between several Core instances. The recommended way is to have
 * a single Core instance per application.
 */
class OPENVINO_RUNTIME_API Core {
    class Impl;
    std::shared_ptr<Impl> _impl;

public:
    /** @brief Constructs an OpenVINO Core instance with devices
     * and their plugins description.
     *
     * There are two ways how to configure device plugins:
     * 1. (default) Use XML configuration file in case of dynamic libraries build;
     * 2. Use strictly defined configuration in case of static libraries build.
     *
     * @param xml_config_file Path to the .xml file with plugins to load from. If path contains only file name
     * with extension, file will be searched in a folder with OpenVINO runtime shared library.
     * If the XML configuration file is not specified, default OpenVINO Runtime plugins are loaded from:
     * 1. (dynamic build) default `plugins.xml` file located in the same folder as OpenVINO runtime shared library;
     * 2. (static build) statically defined configuration. In this case path to the .xml file is ignored.
     */
    explicit Core(const std::string& xml_config_file = {});

    /**
     * @brief Returns device plugins version information.
     * Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
     * in this case, std::map contains multiple entries, each per device.
     *
     * @param device_name Device name to identify a plugin.
     * @return A vector of versions.
     */
    std::map<std::string, Version> get_versions(const std::string& device_name) const;

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Reads models from IR / ONNX / PDPD / TF / TFLite file formats.
     * @param model_path Path to a model.
     * @param bin_path Path to a data file.
     * For IR format (*.bin):
     *  * if `bin_path` is empty, will try to read a bin file with the same name as xml and
     *  * if the bin file with the same name is not found, will load IR without weights.
     * For the following file formats the `bin_path` parameter is not used:
     *  * ONNX format (*.onnx)
     *  * PDPD (*.pdmodel)
     *  * TF (*.pb, *.meta, SavedModel directory)
     *  * TFLite (*.tflite)
     * @param properties Optional map of pairs: (property name, property value) relevant only for this read operation.
     * @return A model.
     */
    std::shared_ptr<ov::Model> read_model(const std::wstring& model_path,
                                          const std::wstring& bin_path = {},
                                          const ov::AnyMap& properties = {}) const;
#endif

    /**
     * @brief Reads models from IR / ONNX / PDPD / TF / TFLite file formats.
     * @param model_path Path to a model.
     * @param bin_path Path to a data file.
     * For IR format (*.bin):
     *  * if `bin_path` is empty, will try to read a bin file with the same name as xml and
     *  * if the bin file with the same name is not found, will load IR without weights.
     * For the following file formats the `bin_path` parameter is not used:
     *  * ONNX format (*.onnx)
     *  * PDPD (*.pdmodel)
     *  * TF (*.pb, *.meta, SavedModel directory)
     *  * TFLite (*.tflite)
     * @param properties Optional map of pairs: (property name, property value) relevant only for this read operation.
     * @return A model.
     * @{
     */
    std::shared_ptr<ov::Model> read_model(const std::string& model_path,
                                          const std::string& bin_path = {},
                                          const ov::AnyMap& properties = {}) const;

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    template <class Path, std::enable_if_t<std::is_same_v<Path, std::filesystem::path>>* = nullptr>
    auto read_model(const Path& model_path, const Path& bin_path = {}, const ov::AnyMap& properties = {}) const {
        return read_model(model_path.string(), bin_path.string(), properties);
    }
#endif
    /// @}

    /**
     * @brief Reads models from IR / ONNX / PDPD / TF / TFLite file formats.
     *
     * @param model_path Path to a model.
     * @param bin_path Path to a data file.
     * For IR format (*.bin):
     *  * if `bin_path` is empty, will try to read a bin file with the same name as xml and
     *  * if the bin file with the same name is not found, will load IR without weights.
     * For the following file formats the `bin_path` parameter is not used:
     *  * ONNX format (*.onnx)
     *  * PDPD (*.pdmodel)
     *  * TF (*.pb, *.meta, SavedModel directory)
     *  * TFLite (*.tflite)
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this read operation.
     * @return A model.
     * @{
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> read_model(const std::string& model_path,
                                                                        const std::string& bin_path,
                                                                        Properties&&... properties) const {
        return read_model(model_path, bin_path, AnyMap{std::forward<Properties>(properties)...});
    }

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    template <class Path,
              class... Properties,
              std::enable_if_t<std::is_same_v<Path, std::filesystem::path> && (sizeof...(Properties) > 0)>* = nullptr>
    auto read_model(const Path& model_path, const Path& bin_path, Properties&&... properties) const {
        return read_model(model_path.string(), bin_path.string(), std::forward<Properties>(properties)...);
    }
#endif
    /// @}

    /**
     * @brief Reads models from IR / ONNX / PDPD / TF / TFLite formats.
     * @param model String with a model in IR / ONNX / PDPD / TF / TFLite format.
     * @param weights Shared pointer to a constant tensor with weights.
     * Reading ONNX / PDPD / TF / TFLite models does not support loading weights from the @p weights tensors.
     * @note Created model object shares the weights with the @p weights object.
     * Thus, do not create @p weights on temporary data that can be freed later, since the model
     * constant data will point to an invalid memory.
     * @return A model.
     */
    std::shared_ptr<ov::Model> read_model(const std::string& model, const Tensor& weights) const;

    /**
     * @brief Creates and loads a compiled model from a source model to the default OpenVINO device selected by the AUTO
     * plugin.
     *
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources).
     *
     * @param model Model object acquired from Core::read_model.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this load
     * operation.
     * @return A compiled model.
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model, const AnyMap& properties = {});

    /**
     * @brief Creates and loads a compiled model from a source model to the default OpenVINO device selected by AUTO
     * plugin.
     *
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources)
     *
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model Model object acquired from Core::read_model
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this
     * load operation
     *
     * @return A compiled model
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> compile_model(
        const std::shared_ptr<const ov::Model>& model,
        Properties&&... properties) {
        return compile_model(model, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Creates a compiled model from a source model object.
     *
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources).
     *
     * @param model Model object acquired from Core::read_model.
     * @param device_name Name of a device to load a model to.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this load
     * operation.
     * @return A compiled model.
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model,
                                const std::string& device_name,
                                const AnyMap& properties = {});

    /**
     * @brief Creates a compiled model from a source model object.
     *
     * Users can create as many compiled models as they need and use
     * them simultaneously (up to the limitation of the hardware resources)
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model Model object acquired from Core::read_model
     * @param device_name Name of device to load model to
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this
     * load operation
     * @return A compiled model
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> compile_model(
        const std::shared_ptr<const ov::Model>& model,
        const std::string& device_name,
        Properties&&... properties) {
        return compile_model(model, device_name, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Reads and loads a compiled model from the IR/ONNX/PDPD file to the default OpenVINO device selected by the
     * AUTO plugin.
     *
     * This can be more efficient than using the Core::read_model + Core::compile_model(model_in_memory_object) flow,
     * especially for cases when caching is enabled and a cached model is available.
     *
     * @param model_path Path to a model.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this load
     * operation.
     *
     * @return A compiled model.
     * @{
     */
    CompiledModel compile_model(const std::string& model_path, const AnyMap& properties = {});

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    template <class Path, std::enable_if_t<std::is_same_v<Path, std::filesystem::path>>* = nullptr>
    auto compile_model(const Path& model_path, const AnyMap& properties = {}) const {
        return compile_model(model_path.string(), properties);
    }
#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    CompiledModel compile_model(const std::wstring& model_path, const AnyMap& properties = {});
#endif
    /// @}

    /**
     * @brief Reads and loads a compiled model from IR / ONNX / PDPD file to the default OpenVINO device selected by
     * AUTO plugin.
     *
     * This can be more efficient than using read_model + compile_model(Model) flow
     * especially for cases when caching is enabled and cached model is available
     *
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model_path path to model with string or wstring
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this
     * load operation
     *
     * @return A compiled model
     * @{
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> compile_model(const std::string& model_path,
                                                                           Properties&&... properties) {
        return compile_model(model_path, AnyMap{std::forward<Properties>(properties)...});
    }

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    template <class Path, class... Properties, std::enable_if_t<std::is_same_v<Path, std::filesystem::path>>* = nullptr>
    auto compile_model(const Path& model_path, Properties&&... properties) {
        return compile_model(model_path.string(), std::forward<Properties>(properties)...);
    }
#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> compile_model(const std::wstring& model_path,
                                                                           Properties&&... properties) {
        return compile_model(model_path, AnyMap{std::forward<Properties>(properties)...});
    }
#endif
    /// @}

    /**
     * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
     *
     * This can be more efficient than using the Core::read_model + Core::compile_model(model_in_memory_object) flow,
     * especially for cases when caching is enabled and a cached model is available.
     *
     * @param model_path Path to a model.
     * @param device_name Name of a device to load a model to.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this load
     * operation.
     *
     * @return A compiled model.
     * @{
     */
    CompiledModel compile_model(const std::string& model_path,
                                const std::string& device_name,
                                const AnyMap& properties = {});

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    template <class Path, std::enable_if_t<std::is_same_v<Path, std::filesystem::path>>* = nullptr>
    auto compile_model(const Path& model_path, const std::string& device_name, const AnyMap& properties = {}) {
        return compile_model(model_path.string(), device_name, properties);
    }
#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    CompiledModel compile_model(const std::wstring& model_path,
                                const std::string& device_name,
                                const AnyMap& properties = {});
#endif
    /// @}

    /**
     * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
     *
     * This can be more efficient than using read_model + compile_model(Model) flow
     * especially for cases when caching is enabled and cached model is available.
     *
     * @tparam Properties Should be a pack of `std::pair<std::string, ov::Any>` types.
     * @param model_path Path to a model.
     * @param device_name Name of a device to load a model to.
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this
     * load operation.
     *
     * @return A compiled model.
     * @{
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> compile_model(const std::string& model_path,
                                                                           const std::string& device_name,
                                                                           Properties&&... properties) {
        return compile_model(model_path, device_name, AnyMap{std::forward<Properties>(properties)...});
    }

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    template <class Path, class... Properties, std::enable_if_t<std::is_same_v<Path, std::filesystem::path>>* = nullptr>
    auto compile_model(const Path& model_path, const std::string& device_name, Properties&&... properties) {
        return compile_model(model_path.string(), device_name, std::forward<Properties>(properties)...);
    }
#endif

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> compile_model(const std::wstring& model_path,
                                                                           const std::string& device_name,
                                                                           Properties&&... properties) {
        return compile_model(model_path, device_name, AnyMap{std::forward<Properties>(properties)...});
    }
#endif
    /// @}

    /**
     * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD memory.
     * @param model String with a model in IR/ONNX/PDPD format.
     * @param weights Shared pointer to a constant tensor with weights.
     * Reading ONNX/PDPD models does not support loading weights from the @p weights tensors.
     * @param device_name Name of a device to load a model to.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this load
     * operation.
     * @note Created model object shares the weights with the @p weights object.
     * Thus, do not create @p weights on temporary data that can be freed later, since the model
     * constant data will point to an invalid memory.
     * @return A compiled model.
     */
    CompiledModel compile_model(const std::string& model,
                                const ov::Tensor& weights,
                                const std::string& device_name,
                                const AnyMap& properties = {});

    /**
     * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD memory.
     * @param model String with a model in IR/ONNX/PDPD format.
     * @param weights Shared pointer to a constant tensor with weights.
     * Reading ONNX/PDPD models does not support loading weights from the @p weights tensors.
     * @param device_name Name of a device to load a model to.
     * @tparam Properties Should be a pack of `std::pair<std::string, ov::Any>` types.
     * @note Created model object shares the weights with the @p weights object.
     * Thus, do not create @p weights on temporary data that can be freed later, since the model
     * constant data will point to an invalid memory.
     * @return A compiled model.
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> compile_model(const std::string& model,
                                                                           const ov::Tensor& weights,
                                                                           const std::string& device_name,
                                                                           Properties&&... properties) {
        return compile_model(model, weights, device_name, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Creates a compiled model from a source model within a specified remote context.
     * @param model Model object acquired from Core::read_model.
     * @param context A reference to a RemoteContext object.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this load
     * operation.
     * @return A compiled model object.
     */
    CompiledModel compile_model(const std::shared_ptr<const ov::Model>& model,
                                const RemoteContext& context,
                                const AnyMap& properties = {});

    /**
     * @brief Creates a compiled model from a source model within a specified remote context.
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param model Model object acquired from Core::read_model
     * @param context Pointer to RemoteContext object
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this
     * load operation
     * @return A compiled model object
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> compile_model(
        const std::shared_ptr<const ov::Model>& model,
        const RemoteContext& context,
        Properties&&... properties) {
        return compile_model(model, context, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Registers an extension to a Core object.
     * @param library_path Path to the library with ov::Extension.
     * @{
     */
    void add_extension(const std::string& library_path);

#ifdef OPENVINO_CPP_VER_AT_LEAST_17
    template <class Path, std::enable_if_t<std::is_same_v<Path, std::filesystem::path>>* = nullptr>
    void add_extension(const Path& model_path) {
        add_extension(model_path.string());
    }
#endif
    /// @}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    /**
     * @brief Registers an extension to a Core object.
     * @param library_path Unicode path to the library with ov::Extension.
     */
    void add_extension(const std::wstring& library_path);
#endif

    /**
     * @brief Registers an extension to a Core object.
     * @param extension Pointer to the extension.
     */
    void add_extension(const std::shared_ptr<ov::Extension>& extension);

    /**
     * @brief Registers extensions to a Core object.
     * @param extensions Vector of loaded extensions.
     */
    void add_extension(const std::vector<std::shared_ptr<ov::Extension>>& extensions);

    /**
     * @brief Registers an extension to a Core object.
     * @param extension Extension class that is inherited from the ov::Extension class.
     */
    template <class T, typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
    }

    /**
     * @brief Registers extensions to a Core object.
     * @param extension Extension class that is inherited from the ov::Extension class.
     * @param args A list of extensions.
     */
    template <class T,
              class... Targs,
              typename std::enable_if<std::is_base_of<ov::Extension, T>::value, bool>::type = true>
    void add_extension(const T& extension, Targs... args) {
        std::shared_ptr<ov::Extension> ext = std::make_shared<T>(extension);
        add_extension(ext);
        add_extension(args...);
    }

    /**
     * @brief Registers a custom operation inherited from ov::op::Op.
     */
    template <class T, typename std::enable_if<std::is_base_of<ov::op::Op, T>::value, bool>::type = true>
    void add_extension() {
        std::shared_ptr<ov::Extension> ext = std::make_shared<ov::OpExtension<T>>();
        add_extension(ext);
    }

    /**
     * @brief Registers custom operations inherited from ov::op::Op.
     */
    template <class T,
              class... Targs,
              typename std::enable_if<std::is_base_of<ov::op::Op, T>::value && sizeof...(Targs), bool>::type = true>
    void add_extension() {
        std::shared_ptr<ov::Extension> ext = std::make_shared<ov::OpExtension<T>>();
        add_extension(ext);
        if (sizeof...(Targs) > 0)
            add_extension<Targs...>();
    }

    /**
     * @brief Imports a compiled model from the previously exported one.
     * @param model_stream std::istream input stream containing a model previously exported using the
     * ov::CompiledModel::export_model method.
     * @param device_name Name of a device to import a compiled model for. Note, if @p device_name device was not used
     * to compile the original mode, an exception is thrown.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this load
     * operation.
     * @return A compiled model.
     */
    CompiledModel import_model(std::istream& model_stream,
                               const std::string& device_name,
                               const AnyMap& properties = {});

    /**
     * @brief Imports a compiled model from the previously exported one.
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types.
     * @param model_stream Model stream.
     * @param device_name Name of a device to import a compiled model for. Note, if @p device_name device was not used
     * to compile the original mode, an exception is thrown.
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this
     * load operation.
     * @return A compiled model.
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> import_model(std::istream& model_stream,
                                                                          const std::string& device_name,
                                                                          Properties&&... properties) {
        return import_model(model_stream, device_name, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Imports a compiled model from the previously exported one with the specified remote context.
     * @param model_stream std::istream input stream containing a model previously exported from
     * ov::CompiledModel::export_model
     * @param context A reference to a RemoteContext object. Note, if the device from @p context was not used to compile
     * the original mode, an exception is thrown.
     * @param properties Optional map of pairs: (property name, property value) relevant only for this load
     * operation.
     * @return A compiled model.
     */
    CompiledModel import_model(std::istream& model_stream, const RemoteContext& context, const AnyMap& properties = {});

    /**
     * @brief Imports a compiled model from the previously exported one with the specified remote context.
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types.
     * @param model_stream Model stream.
     * @param context Pointer to a RemoteContext object.
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this
     * load operation.
     * @return A compiled model.
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<CompiledModel, Properties...> import_model(std::istream& model_stream,
                                                                          const RemoteContext& context,
                                                                          Properties&&... properties) {
        return import_model(model_stream, context, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Query device if it supports the specified model with specified properties.
     *
     * @param device_name Name of a device to query.
     * @param model Model object to query.
     * @param properties Optional map of pairs: (property name, property value).
     * @return An object containing a map of pairs an operation name -> a device name supporting this operation.
     */
    SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                const std::string& device_name,
                                const AnyMap& properties = {}) const;

    /**
     * @brief Queries a device if it supports the specified model with specified properties.
     *
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types.
     * @param device_name Name of a device to query.
     * @param model Model object to query.
     * @param properties Optional pack of pairs: (property name, property value) relevant only for this
     * query operation.
     * @return An object containing a map of pairs an operation name -> a device name supporting this operation.
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<SupportedOpsMap, Properties...> query_model(
        const std::shared_ptr<const ov::Model>& model,
        const std::string& device_name,
        Properties&&... properties) const {
        return query_model(model, device_name, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Sets properties for all the
     * registered devices, acceptable keys can be found in openvino/runtime/properties.hpp.
     *
     * @param properties Map of pairs: (property name, property value).
     */
    void set_property(const AnyMap& properties);

    /**
     * @brief Sets properties for all the
     * registered devices, acceptable keys can be found in openvino/runtime/properties.hpp.
     *
     * @tparam Properties Should be a pack of `std::pair<std::string, ov::Any>` types.
     * @param properties Optional pack of pairs: property name, property value.
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(Properties&&... properties) {
        set_property(AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Sets properties for a device, acceptable keys can be found in openvino/runtime/properties.hpp.
     *
     * @param device_name Name of a device.
     *
     * @param properties Map of pairs: (property name, property value).
     */
    void set_property(const std::string& device_name, const AnyMap& properties);

    /**
     * @brief Sets properties for a device, acceptable keys can be found in openvino/runtime/properties.hpp.
     *
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types.
     * @param device_name Name of a device.
     * @param properties Optional pack of pairs: (property name, property value).
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> set_property(const std::string& device_name,
                                                                 Properties&&... properties) {
        set_property(device_name, AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Gets properties related to device behaviour.
     *
     * The method extracts information that can be set via the set_property method.
     *
     * @param device_name  Name of a device to get a property value.
     * @param name  Property name.
     * @return Value of a property corresponding to the property name.
     */
    Any get_property(const std::string& device_name, const std::string& name) const;

    /**
     * @brief Gets properties related to device behaviour.
     *
     * The method extracts information that can be set via the set_property method.
     *
     * @param device_name  Name of a device to get a property value.
     * @param name  Property name.
     * @param arguments  Additional arguments to get a property.
     * @return Value of a property corresponding to the property name.
     */
    Any get_property(const std::string& device_name, const std::string& name, const AnyMap& arguments) const;

    /**
     * @brief Gets properties related to core behaviour.
     *
     * The method extracts information that can be set via the set_property method.
     *
     * @param name  Property name.
     * @return Value of a property corresponding to the property name.
     */
    Any get_property(const std::string& name) const {
        return get_property(std::string(), name);
    }

    /**
     * @brief Gets properties related to device behaviour.
     *
     * The method is needed to request common device or system properties.
     * It can be device name, temperature, and other devices-specific values.
     *
     * @tparam T Type of a returned value.
     * @tparam M Property mutability.
     * @param device_name  Name of a device to get a property value.
     * @param property  Property object.
     * @return Property value.
     */
    template <typename T, PropertyMutability M>
    T get_property(const std::string& device_name, const ov::Property<T, M>& property) const {
        return get_property(device_name, property.name(), {}).template as<T>();
    }

    /**
     * @brief Gets properties related to device behaviour.
     *
     * The method is needed to request common device or system properties.
     * It can be device name, temperature, other devices-specific values.
     *
     * @tparam T Type of a returned value.
     * @tparam M Property mutability.
     * @param device_name  Name of a device to get a property value.
     * @param property  Property object.
     * @param arguments  Additional arguments to get a property.
     * @return Property value.
     */
    template <typename T, PropertyMutability M>
    T get_property(const std::string& device_name, const ov::Property<T, M>& property, const AnyMap& arguments) const {
        return get_property(device_name, property.name(), arguments).template as<T>();
    }

    /**
     * @brief Gets properties related to device behaviour.
     *
     * The method is needed to request common device or system properties.
     * It can be device name, temperature, other devices-specific values.
     *
     * @tparam T Type of a returned value.
     * @tparam M Property mutability.
     * @tparam Args Set of additional arguments ended with property object variable.
     * @param device_name  Name of a device to get a property value.
     * @param property  Property object.
     * @param args Optional pack of pairs: (argument name, argument value) ended with property object.
     * @return Property value.
     */
    template <typename T, PropertyMutability M, typename... Args>
    util::EnableIfAllStringAny<T, Args...> get_property(const std::string& device_name,
                                                        const ov::Property<T, M>& property,
                                                        Args&&... args) const {
        return get_property(device_name, property.name(), AnyMap{std::forward<Args>(args)...}).template as<T>();
    }

    /**
     * @brief Returns devices available for inference.
     * Core objects go over all registered plugins and ask about available devices.
     *
     * @return A vector of devices. The devices are returned as { CPU, GPU.0, GPU.1, NPU }.
     * If there is more than one device of a specific type, they are enumerated with the .# suffix.
     * Such enumerated device can later be used as a device name in all Core methods like Core::compile_model,
     * Core::query_model, Core::set_property and so on.
     */
    std::vector<std::string> get_available_devices() const;

    /**
     * @brief Register a new device and plugin that enables this device inside OpenVINO Runtime.
     *
     * @param plugin Path (absolute or relative) or name of a plugin. Depending on platform, `plugin` is wrapped with
     * shared library suffix and prefix to identify library full name.
     * For example, on Linux platform, plugin name specified as `plugin_name` will be wrapped as `libplugin_name.so`.
     * Plugin search algorithm:
     * - If `plugin` points to an exact library path (absolute or relative), it will be used.
     * - If `plugin` specifies file name (`libplugin_name.so`) or plugin name (`plugin_name`), it will be searched by
     *   file name (`libplugin_name.so`) in CWD or in paths pointed by PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH
     *   environment variables depending on the platform.
     * @note For security purposes it suggested to specify absolute path to register plugin.
     *
     * @param device_name Device name to register a plugin for.
     * @param config Plugin configuration options
     */
    void register_plugin(const std::string& plugin, const std::string& device_name, const ov::AnyMap& config = {});

    /**
     * @brief Unloads the previously loaded plugin identified by @p device_name from OpenVINO Runtime.
     * The method is needed to remove loaded plugin instance and free its resources. If plugin for a
     * specified device has not been created before, the method throws an exception.
     * @note This method does not remove plugin from the plugins known to OpenVINO Core object.
     * @param device_name Device name identifying plugin to remove from OpenVINO Runtime.
     */
    void unload_plugin(const std::string& device_name);

    /** @brief Registers a device plugin to the OpenVINO Runtime Core instance using an XML configuration file with
     * plugins description.
     *
     *  The XML file has the following structure:
     *
     * ```xml
     * <ie>
     *     <plugins>
     *         <plugin name="" location="">
     *             <extensions>
     *                 <extension location=""/>
     *             </extensions>
     *             <properties>
     *                 <property key="" value=""/>
     *             </properties>
     *         </plugin>
     *     </plugins>
     * </ie>
     * ```
     *
     * - `name` identifies name of a device enabled by a plugin.
     * - `location` specifies absolute path to dynamic library with a plugin.
     *    The path can also be relative to XML file directory. It allows having common config
     *    for different systems with different configurations.
     * - `properties` are set to a plugin via the ov::Core::set_property method.
     * - `extensions` are set to a plugin via the ov::Core::add_extension method.
     * @note For security purposes it suggested to specify absolute path to register plugin.
     *
     * @param xml_config_file A path to .xml file with plugins to register.
     */
    void register_plugins(const std::string& xml_config_file);

    /**
     * @brief Creates a new remote shared context object on the specified accelerator device
     * using specified plugin-specific low-level device API parameters (device handle, pointer, context, etc.).
     * @param device_name Name of a device to create a new shared context on.
     * @param remote_properties Map of device-specific shared context remote properties.
     * @return Reference to a created remote context.
     */
    RemoteContext create_context(const std::string& device_name, const AnyMap& remote_properties);

    /**
     * @brief Creates a new shared context object on specified accelerator device
     * using specified plugin-specific low level device API properties (device handle, pointer, etc.)
     * @tparam Properties Should be the pack of `std::pair<std::string, ov::Any>` types
     * @param device_name Name of a device to create new shared context on.
     * @param remote_properties Pack of device-specific shared context remote properties.
     * @return A shared pointer to a created remote context.
     */
    template <typename... Properties>
    util::EnableIfAllStringAny<RemoteContext, Properties...> create_context(const std::string& device_name,
                                                                            Properties&&... remote_properties) {
        return create_context(device_name, AnyMap{std::forward<Properties>(remote_properties)...});
    }

    /**
     * @brief Gets a pointer to default (plugin-supplied) shared context object for the specified accelerator device.
     * @param device_name Name of a device to get a default shared context from.
     * @return Reference to a default remote context.
     */
    RemoteContext get_default_context(const std::string& device_name);
};

/**
 * @brief Shut down the OpenVINO by deleting all static-duration objects allocated by the library and releasing
 * dependent resources
 *
 * @note This function should be used by advanced user to control unload the resources.
 *
 * You might want to use this function if you are developing a dynamically-loaded library which should clean up all
 * resources after itself when the library is unloaded.
 */
OPENVINO_RUNTIME_API void shutdown();

}  // namespace ov
