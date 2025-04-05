// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for prepostprocess C API, which is a C wrapper for ov::preprocess class.
 * Main class for adding pre- and post- processing steps to existing ov::Model
 * @file ov_prepostprocess.h
 */

#pragma once

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_layout.h"
#include "openvino/c/ov_model.h"
#include "openvino/c/ov_tensor.h"

/**
 * @struct ov_preprocess_prepostprocessor_t
 * @ingroup ov_prepostprocess_c_api
 * @brief type define ov_preprocess_prepostprocessor_t from ov_preprocess_prepostprocessor
 */
typedef struct ov_preprocess_prepostprocessor ov_preprocess_prepostprocessor_t;

/**
 * @struct ov_preprocess_input_info_t
 * @ingroup ov_prepostprocess_c_api
 * @brief type define ov_preprocess_input_info_t from ov_preprocess_input_info
 */
typedef struct ov_preprocess_input_info ov_preprocess_input_info_t;

/**
 * @struct ov_preprocess_input_tensor_info_t
 * @ingroup ov_prepostprocess_c_api
 * @brief type define ov_preprocess_input_tensor_info_t from ov_preprocess_input_tensor_info
 */
typedef struct ov_preprocess_input_tensor_info ov_preprocess_input_tensor_info_t;

/**
 * @struct ov_preprocess_output_info_t
 * @ingroup ov_prepostprocess_c_api
 * @brief type define ov_preprocess_output_info_t from ov_preprocess_output_info
 */
typedef struct ov_preprocess_output_info ov_preprocess_output_info_t;

/**
 * @struct ov_preprocess_output_tensor_info_t
 * @ingroup ov_prepostprocess_c_api
 * @brief type define ov_preprocess_output_tensor_info_t from ov_preprocess_output_tensor_info
 */
typedef struct ov_preprocess_output_tensor_info ov_preprocess_output_tensor_info_t;

/**
 * @struct ov_preprocess_input_model_info_t
 * @ingroup ov_prepostprocess_c_api
 * @brief type define ov_preprocess_input_model_info_t from ov_preprocess_input_model_info
 */
typedef struct ov_preprocess_input_model_info ov_preprocess_input_model_info_t;

/**
 * @struct ov_preprocess_preprocess_steps_t
 * @ingroup ov_prepostprocess_c_api
 * @brief type define ov_preprocess_preprocess_steps_t from ov_preprocess_preprocess_steps
 */
typedef struct ov_preprocess_preprocess_steps ov_preprocess_preprocess_steps_t;

/**
 * @enum ov_color_format_e
 * @ingroup ov_prepostprocess_c_api
 * @brief This enum contains enumerations for color format.
 */
typedef enum {
    UNDEFINE = 0U,      //!< Undefine color format
    NV12_SINGLE_PLANE,  //!< Image in NV12 format as single tensor
    NV12_TWO_PLANES,    //!< Image in NV12 format represented as separate tensors for Y and UV planes.
    I420_SINGLE_PLANE,  //!< Image in I420 (YUV) format as single tensor
    I420_THREE_PLANES,  //!< Image in I420 format represented as separate tensors for Y, U and V planes.
    RGB,                //!< Image in RGB interleaved format (3 channels)
    BGR,                //!< Image in BGR interleaved format (3 channels)
    GRAY,               //!< Image in GRAY format (1 channel)
    RGBX,               //!< Image in RGBX interleaved format (4 channels)
    BGRX                //!< Image in BGRX interleaved format (4 channels)
} ov_color_format_e;

/**
 * @enum ov_preprocess_resize_algorithm_e
 * @ingroup ov_prepostprocess_c_api
 * @brief This enum contains codes for all preprocess resize algorithm.
 */
typedef enum {
    RESIZE_LINEAR,  //!< linear algorithm
    RESIZE_CUBIC,   //!< cubic algorithm
    RESIZE_NEAREST  //!< nearest algorithm
} ov_preprocess_resize_algorithm_e;

/**
 * @enum ov_padding_mode_e
 * @ingroup ov_prepostprocess_c_api
 * @brief This enum contains enumeration for  padding mode.
 */
typedef enum {
    CONSTANT = 0,  //!< Pads with given constant value.
    EDGE,          //!< Pads with tensor edge values.
    REFLECT,       //!< Pads with reflection of tensor data along axis. Values on the edges are not duplicated.
    SYMMETRIC      //!<  Pads similar like `REFLECT` but values on the edges are duplicated.
} ov_padding_mode_e;

/**
 * @brief Create a ov_preprocess_prepostprocessor_t instance.
 * @ingroup ov_prepostprocess_c_api
 * @param model A pointer to the ov_model_t.
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_create(const ov_model_t* model, ov_preprocess_prepostprocessor_t** preprocess);

/**
 * @brief Release the memory allocated by ov_preprocess_prepostprocessor_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_prepostprocessor_free(ov_preprocess_prepostprocessor_t* preprocess);

/**
 * @brief Get the input info of ov_preprocess_prepostprocessor_t instance.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_input_info(const ov_preprocess_prepostprocessor_t* preprocess,
                                              ov_preprocess_input_info_t** preprocess_input_info);

/**
 * @brief Get the input info of ov_preprocess_prepostprocessor_t instance by tensor name.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_name The name of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_input_info_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                      const char* tensor_name,
                                                      ov_preprocess_input_info_t** preprocess_input_info);

/**
 * @brief Get the input info of ov_preprocess_prepostprocessor_t instance by tensor order.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_index The order of input.
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_input_info_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                       const size_t tensor_index,
                                                       ov_preprocess_input_info_t** preprocess_input_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_info_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_input_info_free(ov_preprocess_input_info_t* preprocess_input_info);

/**
 * @brief Get a ov_preprocess_input_tensor_info_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_tensor_info A pointer to ov_preprocess_input_tensor_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_info_get_tensor_info(const ov_preprocess_input_info_t* preprocess_input_info,
                                         ov_preprocess_input_tensor_info_t** preprocess_input_tensor_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_tensor_info_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_input_tensor_info_free(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info);

/**
 * @brief Get a ov_preprocess_preprocess_steps_t.
 * @ingroup ov_prepostprocess_c_api
 * @param ov_preprocess_input_info_t A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_info_get_preprocess_steps(const ov_preprocess_input_info_t* preprocess_input_info,
                                              ov_preprocess_preprocess_steps_t** preprocess_input_steps);

/**
 * @brief Release the memory allocated by ov_preprocess_preprocess_steps_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocess_steps_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_preprocess_steps_free(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps);

/**
 * @brief Add resize operation to model's dimensions.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @param resize_algorithm A ov_preprocess_resizeAlgorithm instance
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_resize(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                      const ov_preprocess_resize_algorithm_e resize_algorithm);

/**
 * @brief Add scale preprocess operation. Divide each element of input by specified value.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @param value Scaling value
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_scale(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps, float value);

/**
 * @brief Add scale preprocess operation. Divide each channel element of input by different specified value.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @param values Scaling values array for each channels
 * @param value_size Scaling value size
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_scale_multi_channels(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                    const float* values,
                                                    const int32_t value_size);

/**
 * @brief Add mean preprocess operation. Subtract specified value from each element of input.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @param value Value to subtract from each element.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_mean(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps, float value);

/**
 * @brief Add mean preprocess operation. Subtract each channel element of input by different specified value.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @param values Value array to subtract from each element.
 * @param value_size Mean value size
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_mean_multi_channels(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                   const float* values,
                                                   const int32_t value_size);

/**
 * @brief Crop input tensor between begin and end coordinates.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @param begin Pointer to begin indexes for input tensor cropping.
 * Negative values represent counting elements from the end of input tensor
 * @param begin_size The size of begin array
 * @param end Pointer to end indexes for input tensor cropping.
 * End indexes are exclusive, which means values including end edge are not included in the output slice.
 * Negative values represent counting elements from the end of input tensor
 * @param end_size The size of end array
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_crop(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                    int32_t* begin,
                                    int32_t begin_size,
                                    int32_t* end,
                                    int32_t end_size);

/**
 * @brief Add 'convert layout' operation to specified layout.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @param layout A point to ov_layout_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_convert_layout(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                              ov_layout_t* layout);

/**
 * @brief Reverse channels operation.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_process_steps A pointer to ov_preprocess_preprocess_steps_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_reverse_channels(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps);

/**
 * @brief Set ov_preprocess_input_tensor_info_t precesion.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param element_type A point to element_type
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_element_type(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                 const ov_element_type_e element_type);

/**
 * @brief Set ov_preprocess_input_tensor_info_t color format.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param colorFormat The enumerate of colorFormat
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_color_format(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                 const ov_color_format_e colorFormat);

/**
 * @brief Set ov_preprocess_input_tensor_info_t color format with subname.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param colorFormat The enumerate of colorFormat
 * @param sub_names_size The size of sub_names
 * @param ... variadic params sub_names Optional list of sub-names assigned for each plane (e.g. "Y", "UV").
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_color_format_with_subname(
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
    const ov_color_format_e colorFormat,
    const size_t sub_names_size,
    ...);

/**
 * @brief Set ov_preprocess_input_tensor_info_t spatial_static_shape.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param input_height The height of input
 * @param input_width The width of input
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_spatial_static_shape(
    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
    const size_t input_height,
    const size_t input_width);

/**
 * @brief Set ov_preprocess_input_tensor_info_t memory type.
 * @ingroup prepostprocess
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param mem_type Memory type. Refer to ov_remote_context.h to get memory type string info.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_memory_type(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                                const char* mem_type);

/**
 * @brief Convert ov_preprocess_preprocess_steps_t element type.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocess_steps_t.
 * @param element_type preprocess input element type.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_convert_element_type(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                                    const ov_element_type_e element_type);

/**
 * @brief Convert ov_preprocess_preprocess_steps_t color.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_steps A pointer to the ov_preprocess_preprocess_steps_t.
 * @param colorFormat The enumerate of colorFormat.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_convert_color(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                             const ov_color_format_e colorFormat);

/**
 * @brief Helper function to reuse element type and shape from user's created tensor.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param tensor A point to ov_tensor_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_from(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                         const ov_tensor_t* tensor);

/**
 * @brief Set ov_preprocess_input_tensor_info_t layout.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_tensor_info A pointer to the ov_preprocess_input_tensor_info_t.
 * @param layout A point to ov_layout_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_tensor_info_set_layout(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info,
                                           ov_layout_t* layout);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_output_info(const ov_preprocess_prepostprocessor_t* preprocess,
                                               ov_preprocess_output_info_t** preprocess_output_info);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_index The tensor index
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_output_info_by_index(const ov_preprocess_prepostprocessor_t* preprocess,
                                                        const size_t tensor_index,
                                                        ov_preprocess_output_info_t** preprocess_output_info);

/**
 * @brief Get the output info of ov_preprocess_output_info_t instance.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param tensor_name The name of input.
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_get_output_info_by_name(const ov_preprocess_prepostprocessor_t* preprocess,
                                                       const char* tensor_name,
                                                       ov_preprocess_output_info_t** preprocess_output_info);

/**
 * @brief Release the memory allocated by ov_preprocess_output_info_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_output_info_free(ov_preprocess_output_info_t* preprocess_output_info);

/**
 * @brief Get a ov_preprocess_input_tensor_info_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_output_info A pointer to the ov_preprocess_output_info_t.
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_output_info_get_tensor_info(const ov_preprocess_output_info_t* preprocess_output_info,
                                          ov_preprocess_output_tensor_info_t** preprocess_output_tensor_info);

/**
 * @brief Release the memory allocated by ov_preprocess_output_tensor_info_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_output_tensor_info_free(ov_preprocess_output_tensor_info_t* preprocess_output_tensor_info);

/**
 * @brief Set ov_preprocess_input_tensor_info_t precesion.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_output_tensor_info A pointer to the ov_preprocess_output_tensor_info_t.
 * @param element_type A point to element_type
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_output_set_element_type(ov_preprocess_output_tensor_info_t* preprocess_output_tensor_info,
                                      const ov_element_type_e element_type);

/**
 * @brief Get current input model information.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_info A pointer to the ov_preprocess_input_info_t.
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_info_get_model_info(const ov_preprocess_input_info_t* preprocess_input_info,
                                        ov_preprocess_input_model_info_t** preprocess_input_model_info);

/**
 * @brief Release the memory allocated by ov_preprocess_input_model_info_t.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t to free memory.
 */
OPENVINO_C_API(void)
ov_preprocess_input_model_info_free(ov_preprocess_input_model_info_t* preprocess_input_model_info);

/**
 * @brief Set layout for model's input tensor.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess_input_model_info A pointer to the ov_preprocess_input_model_info_t
 * @param layout A point to ov_layout_t
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_input_model_info_set_layout(ov_preprocess_input_model_info_t* preprocess_input_model_info,
                                          ov_layout_t* layout);

/**
 * @brief Adds pre/post-processing operations to function passed in constructor.
 * @ingroup ov_prepostprocess_c_api
 * @param preprocess A pointer to the ov_preprocess_prepostprocessor_t.
 * @param model A pointer to the ov_model_t.
 * @return Status code of the operation: OK(0) for success.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_prepostprocessor_build(const ov_preprocess_prepostprocessor_t* preprocess, ov_model_t** model);

/**
 * @brief Add pad preprocess operation. Extends an input tensor on edges with constants.
 *
 * @param preprocess_input_process_steps  A pointer to the ov_preprocess_preprocess_steps_t.
 * @param pads_begin                      Number of padding elements to add at the beginning of each axis.
 * @param pads_begin_size                 Pads begin size (number of axes).
 * @param pads_end                        Number of padding elements to add at the end of each axis.
 * @param pads_end_size                   Pads end size (number of axes).
 * @param value                           Value to be populated in the padded area (mode=CONSTANT)
 * @param mode                            Padding mode.
 */
OPENVINO_C_API(ov_status_e)
ov_preprocess_preprocess_steps_pad(const ov_preprocess_preprocess_steps_t* preprocess_input_process_steps,
                                   const int* const pads_begin,
                                   size_t pads_begin_size,
                                   const int* const pads_end,
                                   size_t pads_end_size,
                                   float value,
                                   ov_padding_mode_e mode);
