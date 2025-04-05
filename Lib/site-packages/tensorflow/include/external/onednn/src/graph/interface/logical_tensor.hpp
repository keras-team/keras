/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GRAPH_INTERFACE_LOGICAL_TENSOR_HPP
#define GRAPH_INTERFACE_LOGICAL_TENSOR_HPP

#include <algorithm>
#include <assert.h>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "common/type_helpers.hpp"

#include "graph/interface/c_types_map.hpp"

#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {

inline logical_tensor_t zero_logical_tensor() {
    auto zero = logical_tensor_t();
    return zero;
}

inline logical_tensor_t empty_logical_tensor_with_default_id() {
    auto empty = logical_tensor_t();
    empty.id = std::numeric_limits<size_t>::max();
    empty.ndims = -1;
    empty.layout_type = layout_type::any;
    return empty;
}

struct logical_tensor_wrapper_t {
    const logical_tensor_t *lt;

    // constructor
    logical_tensor_wrapper_t(const logical_tensor_t *other) : lt(other) {}
    logical_tensor_wrapper_t(const logical_tensor_t &other)
        : logical_tensor_wrapper_t(&other) {}

    // getter
    size_t id() const { return lt->id; }
    int32_t ndims() const { return lt->ndims; }
    data_type_t data_type() const { return lt->data_type; }
    layout_type_t layout_type() const { return lt->layout_type; }
    size_t layout_id() const { return lt->layout.layout_id; }
    property_type_t property_type() const { return lt->property; }

    const dims_t &dims() const { return lt->dims; }
    const dims_t &strides() const { return lt->layout.strides; };

    // convenient method to return a std::vector
    std::vector<dim_t> vdims() const {
        return {lt->dims, lt->dims + lt->ndims};
    }

    // convenient method to return a std::vector
    std::vector<dim_t> vstrides() const {
        return {lt->layout.strides, lt->layout.strides + lt->ndims};
    }

    // checker
    bool is_any() const { return lt->layout_type == layout_type::any; }
    bool is_strided() const { return lt->layout_type == layout_type::strided; }
    bool is_opaque() const { return lt->layout_type == layout_type::opaque; }
    bool is_constant() const { return lt->property == property_type::constant; }
    bool is_layout_type_undef() const {
        return lt->layout_type == layout_type::undef;
    }
    bool is_data_type_undef() const {
        return lt->data_type == data_type::undef;
    }

    bool is_empty() const { return ndims() < 0; }

    bool is_scalar() const { return ndims() == 0; }

    bool has_zero_dim() const {
        for (int d = 0; d < ndims(); ++d) {
            if (dims()[d] == 0) return true;
        }

        return false;
    }

    bool is_shape_unknown() const {
        // TODO(lvtao): need to specify: DNNL_GRAPH_UNKNOWN_NDIMS?
        if (ndims() < 0) return true;

        for (int d = 0; d < ndims(); ++d) {
            // TODO(xx): need to specify: DNNL_GRAPH_UNKNOWN_DIM?
            if (dims()[d] < 0) { return true; }
        }

        return false;
    }

    // check if layout type is strided before calling this function.
    bool is_stride_unknown() const {
        if (ndims() < 0) return true;

        for (int d = 0; d < ndims(); ++d) {
            if (strides()[d] == DNNL_GRAPH_UNKNOWN_DIM) return true;
        }

        return false;
    }

    // every bit should be same
    bool is_identical(const logical_tensor_wrapper_t &rhs) const {
        return is_identical(*(this->lt), *(rhs.lt));
    }

    // layout info may implicit same in backend's perspective
    // other info should be the same like id, data type
    bool operator==(const logical_tensor_wrapper_t &rhs) const {
        return is_similar(*(this->lt), *(rhs.lt), /* check_id = */ true,
                /* check_dtype = */ true);
    }

    bool operator!=(const logical_tensor_wrapper_t &rhs) const {
        return !operator==(rhs);
    }

    bool operator==(const logical_tensor_t &rhs) const {
        return operator==(logical_tensor_wrapper_t(rhs));
    }

    bool operator!=(const logical_tensor_t &rhs) const {
        return !operator==(rhs);
    }

    // equal, but may have different id
    bool is_similar(const logical_tensor_wrapper_t &rhs) const {
        return is_similar(*(this->lt), *(rhs.lt), /* check_id = */ false,
                /* check_dtype = */ true);
    }

    // return the size of data type
    size_t data_type_size() const { return types::data_type_size(data_type()); }

    // get memory size in byte
    size_t size() const;

    // get element number
    dim_t nelems() const {
        if (is_empty()) return 0;
        if (is_scalar()) return 1;
        // TODO(lvtao): need to specify: DNNL_RUNTIME_DIM_VAL?
        if (is_shape_unknown()) return -1;
        return utils::array_product(dims(), static_cast<size_t>(ndims()));
    }

    std::vector<dim_t> get_weight_spatial_dims(
            const std::string &format) const {
        std::vector<dim_t> spatial_dims = vdims();
        if (format == "OIX" || format == "IOX") {
            spatial_dims.erase(spatial_dims.begin(), spatial_dims.begin() + 2);
        } else if (format == "XIO" || format == "XOI") {
            spatial_dims.erase(spatial_dims.end() - 2, spatial_dims.end());
        } else {
            // For code completeness - return an empty vector in this case
            spatial_dims.clear();
        }

        return spatial_dims;
    }

    std::vector<dim_t> get_src_spatial_dims(const std::string &format) const {
        std::vector<dim_t> spatial_dims = vdims();
        if (format == "NCX") {
            spatial_dims.erase(spatial_dims.begin(), spatial_dims.begin() + 2);
        } else if (format == "NXC") {
            spatial_dims.erase(spatial_dims.begin(), spatial_dims.begin() + 1);
            spatial_dims.erase(spatial_dims.end() - 1, spatial_dims.end());
        } else {
            spatial_dims.clear();
        }

        return spatial_dims;
    }

    dim_t get_weight_i(const std::string &format) const {
        if (format == "OIX") {
            return dims()[1];
        } else if (format == "XIO") {
            return dims()[ndims() - 2];
        } else if (format == "IOX") {
            return dims()[0];
        } else if (format == "XOI") {
            return dims()[ndims() - 1];
        } else {
            // For code completeness
            return DNNL_GRAPH_UNKNOWN_DIM;
        }
    }

    dim_t get_weight_o(const std::string &format) const {
        if (format == "OIX") {
            return dims()[0];
        } else if (format == "XIO") {
            return dims()[ndims() - 1];
        } else if (format == "IOX") {
            return dims()[1];
        } else if (format == "XOI") {
            return dims()[ndims() - 2];
        } else {
            // For code completeness
            return DNNL_GRAPH_UNKNOWN_DIM;
        }
    }

    dim_t get_src_n() const {
        // `n` is always the first element for both `NCX` and `NXC`
        return dims()[0];
    }

    dim_t get_src_c(const std::string &format) const {
        if (format == "NCX") {
            return dims()[1];
        } else if (format == "NXC") {
            return dims()[ndims() - 1];
        } else {
            // For code completeness
            return DNNL_GRAPH_UNKNOWN_DIM;
        }
    }

    logical_tensor_t reorder_data_dims_strides() const {
        assert(lt->ndims != -1 && "data dims haven't be uninitialized.");
        // update input tensor's dims NXC
        // keep HW order
        logical_tensor_t cdata = *lt;
        int32_t i = 1, j = cdata.ndims - 1;
        while (i < j) {
            std::swap(cdata.dims[i], cdata.dims[j]);
            if (cdata.layout_type == layout_type::strided) {
                std::swap(cdata.layout.strides[i], cdata.layout.strides[j]);
            }
            ++i;
        }
        return cdata;
    }

    logical_tensor_t reorder_weight_dims_strides() const { // XIO->OIX
        assert(lt->ndims != -1 && "data dims haven't be uninitialized.");
        logical_tensor_t cweight = *lt;
        int32_t i = 0, j = cweight.ndims - 1;
        while (i < j) {
            std::swap(cweight.dims[i], cweight.dims[j]);
            if (cweight.layout_type == layout_type::strided) {
                std::swap(cweight.layout.strides[i], cweight.layout.strides[j]);
            }
            ++i;
            --j;
        }
        // // keep HW order
        i = 2, j = cweight.ndims - 1;
        while (i < j) {
            std::swap(cweight.dims[i], cweight.dims[j]);
            if (cweight.layout_type == layout_type::strided) {
                std::swap(cweight.layout.strides[i], cweight.layout.strides[j]);
            }
            ++i;
            --j;
        }
        return cweight;
    }

    bool has_same_shape_as(const logical_tensor_wrapper_t &rhs) const {
        if (ndims() != rhs.ndims()) return false;
        return std::equal(dims(), dims() + ndims(), rhs.dims());
    }

    // layout info is same while data type maybe not the same
    bool has_same_layout_as(const logical_tensor_wrapper_t &rhs) const {
        return is_similar(*(this->lt), *(rhs.lt), true, true);
    }

    size_t hash() const noexcept;

private:
    bool is_identical(
            const logical_tensor_t &lhs, const logical_tensor_t &rhs) const;

    bool is_similar(const logical_tensor_t &lhs, const logical_tensor_t &rhs,
            bool check_id = true, bool check_dtype = true) const;
};

} // namespace graph
} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::graph::logical_tensor_t> {
    using argument_type = dnnl::impl::graph::logical_tensor_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &lt) const {
        using namespace dnnl::impl::graph;
        return logical_tensor_wrapper_t(lt).hash();
    }
};

template <>
struct equal_to<dnnl::impl::graph::logical_tensor_t> {
    using result_type = bool;
    using first_argument_type = dnnl::impl::graph::logical_tensor_t;
    using second_argument_type = dnnl::impl::graph::logical_tensor_t;
    result_type operator()(const first_argument_type &lhs,
            const second_argument_type &rhs) const {
        using namespace dnnl::impl::graph;
        const logical_tensor_wrapper_t lhs_wrapper {lhs};
        const logical_tensor_wrapper_t rhs_wrapper {rhs};
        return lhs_wrapper == rhs_wrapper;
    }
};
} // namespace std

#endif
