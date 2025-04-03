// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the ov_core C API, which is a C wrapper for ov::Core class.
 * This class represents an OpenVINO runtime Core entity.
 * @file ov_core.h
 */

#pragma once

#include "openvino/c/deprecated.h"
#include "openvino/c/ov_common.h"
#include "openvino/c/ov_compiled_model.h"
#include "openvino/c/ov_model.h"
#include "openvino/c/ov_node.h"
#include "openvino/c/ov_property.h"
#include "openvino/c/ov_remote_context.h"
#include "openvino/c/ov_tensor.h"

/**
 * @struct ov_core_t
 * @ingroup ov_core_c_api
 * @brief type define ov_core_t from ov_core
 */
typedef struct ov_core ov_core_t;

/**
 * @struct ov_version
 * @ingroup ov_core_c_api
 * @brief Represents OpenVINO version information
 */
typedef struct ov_version {
    const char* buildNumber;  //!< A string representing OpenVINO version
    const char* description;  //!< A string representing OpenVINO description
} ov_version_t;

/**
 * @struct ov_core_version
 * @ingroup ov_core_c_api
 * @brief  Represents version information that describes device and ov runtime library
 */
typedef struct {
    const char* device_name;  //!< A device name
    ov_version_t version;     //!< Version
} ov_core_version_t;

/**
 * @struct ov_core_version_list
 * @ingroup ov_core_c_api
 * @brief  Represents version information that describes all devices and ov runtime library
 */
typedef struct {
    ov_core_version_t* versions;  //!< An array of device versions
    size_t size;                  //!< A number of versions in the array
} ov_core_version_list_t;

/**
 * @struct ov_available_devices_t
 * @ingroup ov_core_c_api
 * @brief Represent all available devices.
 */
typedef struct {
    char** devices;  //!< devices' name
    size_t size;     //!< devices' number
} ov_available_devices_t;

/**
 * @brief Get version of OpenVINO.
 * @ingroup ov_core_c_api
 * @param ov_version_t a pointer to the version
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_get_openvino_version(ov_version_t* version);

/**
 * @brief Release the memory allocated by ov_version_t.
 * @ingroup ov_core_c_api
 * @param version A pointer to the ov_version_t to free memory.
 */
OPENVINO_C_API(void)
ov_version_free(ov_version_t* version);

/**
 * @brief Constructs OpenVINO Core instance by default.
 * See RegisterPlugins for more details.
 * @ingroup ov_core_c_api
 * @param core A pointer to the newly created ov_core_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_create(ov_core_t** core);

/**
 * @brief Constructs OpenVINO Core instance using XML configuration file with devices description.
 * See RegisterPlugins for more details.
 * @ingroup ov_core_c_api
 * @param xml_config_file A path to .xml file with devices to load from. If XML configuration file is not specified,
 * then default plugin.xml file will be used.
 * @param core A pointer to the newly created ov_core_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_create_with_config(const char* xml_config_file, ov_core_t** core);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Constructs OpenVINO Core instance.
 * See RegisterPlugins for more details.
 * @ingroup ov_core_c_api
 * @param xml_config_file_ws A path to model file with unicode.
 * @param core A pointer to the newly created ov_core_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_create_with_config_unicode(const wchar_t* xml_config_file_ws, ov_core_t** core);
#endif

/**
 * @brief Release the memory allocated by ov_core_t.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t to free memory.
 */
OPENVINO_C_API(void)
ov_core_free(ov_core_t* core);

/**
 * @brief Reads models from IR / ONNX / PDPD / TF / TFLite formats.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param model_path Path to a model.
 * @param bin_path Path to a data file.
 * For IR format (*.bin):
 *  * if `bin_path` is empty, will try to read a bin file with the same name as xml and
 *  * if the bin file with the same name is not found, will load IR without weights.
 * For the following file formats the `bin_path` parameter is not used:
 *  * ONNX format (*.onnx)
 *  * PDPD (*.pdmodel)
 *  * TF (*.pb)
 *  * TFLite (*.tflite)
 * @param model A pointer to the newly created model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_read_model(const ov_core_t* core, const char* model_path, const char* bin_path, ov_model_t** model);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Reads models from IR / ONNX / PDPD / TF / TFLite formats, path is unicode.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param model_path Path to a model.
 * @param bin_path Path to a data file.
 * For IR format (*.bin):
 *  * if `bin_path` is empty, will try to read a bin file with the same name as xml and
 *  * if the bin file with the same name is not found, will load IR without weights.
 * For the following file formats the `bin_path` parameter is not used:
 *  * ONNX format (*.onnx)
 *  * PDPD (*.pdmodel)
 *  * TF (*.pb)
 *  * TFLite (*.tflite)
 * @param model A pointer to the newly created model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_read_model_unicode(const ov_core_t* core,
                           const wchar_t* model_path,
                           const wchar_t* bin_path,
                           ov_model_t** model);
#endif

/**
 * @brief Reads models from IR / ONNX / PDPD / TF / TFLite formats with models string size.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param model_str String with a model in IR / ONNX / PDPD / TF / TFLite format, support model string containing
 * several null chars.
 * @param str_len The length of model string.
 * @param weights Shared pointer to a constant tensor with weights.
 * @param model A pointer to the newly created model.
 * Reading ONNX / PDPD / TF / TFLite models does not support loading weights from the @p weights tensors.
 * @note Created model object shares the weights with the @p weights object.
 * Thus, do not create @p weights on temporary data that can be freed later, since the model
 * constant data will point to an invalid memory.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_read_model_from_memory_buffer(const ov_core_t* core,
                                      const char* model_str,
                                      const size_t str_len,
                                      const ov_tensor_t* weights,
                                      ov_model_t** model);

/**
 * @brief Creates a compiled model from a source model object.
 * Users can create as many compiled models as they need and use
 * them simultaneously (up to the limitation of the hardware resources).
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param model Model object acquired from Core::read_model.
 * @param device_name Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @param ... property paramater: Optional pack of pairs: <char* property_key, char* property_value> relevant only
 * for this load operation operation. Supported property key please see ov_property.h.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_compile_model(const ov_core_t* core,
                      const ov_model_t* model,
                      const char* device_name,
                      const size_t property_args_size,
                      ov_compiled_model_t** compiled_model,
                      ...);

/**
 * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
 * This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow,
 * especially for cases when caching is enabled and a cached model is available.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param model_path Path to a model.
 * @param device_name Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @param ... Optional pack of pairs: <char* property_key, char* property_value> relevant only
 * for this load operation operation. Supported property key please see ov_property.h.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_compile_model_from_file(const ov_core_t* core,
                                const char* model_path,
                                const char* device_name,
                                const size_t property_args_size,
                                ov_compiled_model_t** compiled_model,
                                ...);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Reads a model and creates a compiled model from the IR/ONNX/PDPD file.
 * This can be more efficient than using the ov_core_read_model_from_XXX + ov_core_compile_model flow,
 * especially for cases when caching is enabled and a cached model is available.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param model_path Path to a model.
 * @param device_name Name of a device to load a model to.
 * @param property_args_size How many properties args will be passed, each property contains 2 args: key and value.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @param ... Optional pack of pairs: <char* property_key, char* property_value> relevant only
 * for this load operation operation. Supported property key please see ov_property.h.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_compile_model_from_file_unicode(const ov_core_t* core,
                                        const wchar_t* model_path,
                                        const char* device_name,
                                        const size_t property_args_size,
                                        ov_compiled_model_t** compiled_model,
                                        ...);

#endif

/**
 * @brief Sets properties for a device, acceptable keys can be found in ov_property_key_xxx.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param device_name Name of a device.
 * @param ... variadic paramaters The format is <char* property_key, char* property_value>.
 * Supported property key please see ov_property.h.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_set_property(const ov_core_t* core, const char* device_name, ...);

/**
 * @brief Gets properties related to device behaviour.
 * The method extracts information that can be set via the set_property method.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param device_name  Name of a device to get a property value.
 * @param property_key  Property key.
 * @param property_value A pointer to property value with string format.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_get_property(const ov_core_t* core, const char* device_name, const char* property_key, char** property_value);

/**
 * @brief Returns devices available for inference.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param devices A pointer to the ov_available_devices_t instance.
 * Core objects go over all registered plugins and ask about available devices.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_get_available_devices(const ov_core_t* core, ov_available_devices_t* devices);

/**
 * @brief Releases memory occpuied by ov_available_devices_t
 * @ingroup ov_core_c_api
 * @param devices A pointer to the ov_available_devices_t instance.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(void)
ov_available_devices_free(ov_available_devices_t* devices);

/**
 * @brief Imports a compiled model from the previously exported one.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param content A pointer to content of the exported model.
 * @param content_size Number of bytes in the exported network.
 * @param device_name Name of a device to import a compiled model for.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_import_model(const ov_core_t* core,
                     const char* content,
                     const size_t content_size,
                     const char* device_name,
                     ov_compiled_model_t** compiled_model);

/**
 * @brief Returns device plugins version information.
 * Device name can be complex and identify multiple devices at once like `HETERO:CPU,GPU`;
 * in this case, std::map contains multiple entries, each per device.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param device_name Device name to identify a plugin.
 * @param versions A pointer to versions corresponding to device_name.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_get_versions_by_device_name(const ov_core_t* core, const char* device_name, ov_core_version_list_t* versions);

/**
 * @brief Releases memory occupied by ov_core_version_list_t.
 * @ingroup ov_core_c_api
 * @param versions A pointer to the ov_core_version_list_t to free memory.
 */
OPENVINO_C_API(void)
ov_core_versions_free(ov_core_version_list_t* versions);

/**
 * @brief Creates a new remote shared context object on the specified accelerator device
 * using specified plugin-specific low-level device API parameters (device handle, pointer, context, etc.).
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param device_name Device name to identify a plugin.
 * @param context_args_size How many property args will be for this remote context creation.
 * @param context A pointer to the newly created remote context.
 * @param ... variadic parmameters Actual context property parameter for remote context
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_create_context(const ov_core_t* core,
                       const char* device_name,
                       const size_t context_args_size,
                       ov_remote_context_t** context,
                       ...);

/**
 * @brief Creates a compiled model from a source model within a specified remote context.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param model Model object acquired from ov_core_read_model.
 * @param context A pointer to the newly created remote context.
 * @param property_args_size How many args will be for this compiled model.
 * @param compiled_model A pointer to the newly created compiled_model.
 * @param ... variadic parmameters Actual property parameter for remote context
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_compile_model_with_context(const ov_core_t* core,
                                   const ov_model_t* model,
                                   const ov_remote_context_t* context,
                                   const size_t property_args_size,
                                   ov_compiled_model_t** compiled_model,
                                   ...);

/**
 * @brief Gets a pointer to default (plugin-supplied) shared context object for the specified accelerator device.
 * @ingroup ov_core_c_api
 * @param core A pointer to the ov_core_t instance.
 * @param device_name Name of a device to get a default shared context from.
 * @param context A pointer to the referenced remote context.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_core_get_default_context(const ov_core_t* core, const char* device_name, ov_remote_context_t** context);

/**
 * @brief Shut down the OpenVINO by deleting all static-duration objects allocated by the library and releasing
 * dependent resources
 * @ingroup ov_c_api
 * @note This function should be used by advanced user to control unload the resources.
 *
 * You might want to use this function if you are developing a dynamically-loaded library which should clean up all
 * resources after itself when the library is unloaded.
 */
OPENVINO_C_API(void) ov_shutdown();
