/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef COMMON_IMPL_LIST_ITEM_HPP
#define COMMON_IMPL_LIST_ITEM_HPP

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace dnnl {
namespace impl {

// This key takes prop_kind and correspondent data_type for src, wei and dst.
struct pk_dt_impl_key_t {
    prop_kind_t kind;
    data_type_t src_dt, wei_dt, dst_dt;

    bool operator<(const pk_dt_impl_key_t &rhs) const {
        return value() < rhs.value();
    }

private:
    size_t value() const {
        const size_t dtm = data_type::data_type_max;
        const size_t m1 = static_cast<size_t>(kind) * dtm;
        const size_t m2 = (m1 + static_cast<size_t>(src_dt)) * dtm;
        const size_t m3 = (m2 + static_cast<size_t>(wei_dt)) * dtm;
        return m3 + static_cast<size_t>(dst_dt);
    }
};

// This is a simpler version of key to use only prop_kind.
struct pk_impl_key_t {
    prop_kind_t kind;

    bool operator<(const pk_impl_key_t &rhs) const {
        return value() < rhs.value();
    }

private:
    size_t value() const { return (size_t)kind; }
};

struct impl_list_item_t {
    constexpr impl_list_item_t() = default;
    constexpr impl_list_item_t(const impl_list_item_t &other) = default;
    constexpr impl_list_item_t(impl_list_item_t &&other) = default;
    impl_list_item_t &operator=(const impl_list_item_t &other) = default;
    impl_list_item_t &operator=(impl_list_item_t &&other) = default;

    constexpr impl_list_item_t(std::nullptr_t) {}

    template <typename pd_t>
    struct type_deduction_helper_t {
        using type = pd_t;
        constexpr type_deduction_helper_t() {
            static_assert(std::is_base_of<primitive_desc_t, pd_t>::value,
                    "type_deduction_helper_t is expected to be used for "
                    "primitive descriptor classes only.");
        }
    };

    template <typename pd_t>
    struct concat_type_deduction_helper_t
        : public type_deduction_helper_t<pd_t> {
        constexpr concat_type_deduction_helper_t() = default;
    };

    template <typename pd_t>
    struct sum_type_deduction_helper_t : public type_deduction_helper_t<pd_t> {
    };

    template <typename pd_t>
    struct reorder_type_deduction_helper_t
        : public type_deduction_helper_t<pd_t> {};

    template <typename pd_t>
    constexpr impl_list_item_t(type_deduction_helper_t<pd_t>)
        : create_pd_func_(&primitive_desc_t::create<
                          typename type_deduction_helper_t<pd_t>::type>) {}

    template <typename pd_t>
    constexpr impl_list_item_t(concat_type_deduction_helper_t<pd_t>)
        : create_concat_pd_func_(
                concat_type_deduction_helper_t<pd_t>::type::create) {}

    template <typename pd_t>
    constexpr impl_list_item_t(sum_type_deduction_helper_t<pd_t>)
        : create_sum_pd_func_(sum_type_deduction_helper_t<pd_t>::type::create) {
    }

    template <typename pd_t>
    constexpr impl_list_item_t(reorder_type_deduction_helper_t<pd_t>)
        : create_reorder_pd_func_(
                reorder_type_deduction_helper_t<pd_t>::type::create) {}

    explicit operator bool() const {
        return !utils::everyone_is(nullptr, create_pd_func_,
                create_concat_pd_func_, create_sum_pd_func_,
                create_reorder_pd_func_);
    }

    // Currently, this only supports iterator friendly primitives. Can be
    // extended to sum, concat and reorder if needed.
    template <typename pd_t>
    static int find(const impl_list_item_t *list) {
        int idx = 0;
        for (const impl_list_item_t *cur = list; *cur; cur++) {
            if (cur->create_pd_func_ == &primitive_desc_t::create<pd_t>)
                return idx;
            idx++;
        }
        return -1;
    }

private:
    status_t operator()(primitive_desc_t **pd, const op_desc_t *adesc,
            const primitive_attr_t *attr, engine_t *engine,
            const primitive_desc_t *hint_fwd, int pd_iterator_offset,
            int skip_idx) const {
        assert(create_pd_func_);
        if (!create_pd_func_) return status::runtime_error;
        auto status = create_pd_func_(pd, adesc, attr, engine, hint_fwd);
        if (status == status::success) {
            (*pd)->init_pd_iterator_offset(pd_iterator_offset);
            (*pd)->init_skip_idx(skip_idx);
        }
        return status;
    }

    status_t operator()(concat_pd_t **concat_pd, engine_t *engine,
            const primitive_attr_t *attr, const memory_desc_t *dst_md, int n,
            int concat_dim, const memory_desc_t *const *src_mds) const {
        assert(create_concat_pd_func_);
        if (!create_concat_pd_func_) return status::runtime_error;
        return create_concat_pd_func_(
                concat_pd, engine, attr, dst_md, n, concat_dim, src_mds);
    }

    status_t operator()(sum_pd_t **sum_pd, engine_t *engine,
            const primitive_attr_t *attr, const memory_desc_t *dst_md, int n,
            const float *scales, const memory_desc_t *const *src_mds) const {
        assert(create_sum_pd_func_);
        if (!create_sum_pd_func_) return status::runtime_error;
        return create_sum_pd_func_(
                sum_pd, engine, attr, dst_md, n, scales, src_mds);
    }

    status_t operator()(reorder_pd_t **reorder_pd, engine_t *engine,
            const primitive_attr_t *attr, engine_t *src_engine,
            const memory_desc_t *src_md, engine_t *dst_engine,
            const memory_desc_t *dst_md) const {
        if (!create_reorder_pd_func_) return status::runtime_error;
        return create_reorder_pd_func_(reorder_pd, engine, attr, src_engine,
                src_md, dst_engine, dst_md);
    }

    using create_pd_func_t = status_t (*)(primitive_desc_t **,
            const op_desc_t *, const primitive_attr_t *, engine_t *,
            const primitive_desc_t *);

    using create_concat_pd_func_t = status_t (*)(concat_pd_t **, engine_t *,
            const primitive_attr_t *, const memory_desc_t *, int, int,
            const memory_desc_t *const *);

    using create_sum_pd_func_t = status_t (*)(sum_pd_t **, engine_t *,
            const primitive_attr_t *, const memory_desc_t *, int, const float *,
            const memory_desc_t *const *);

    using create_reorder_pd_func_t = status_t (*)(reorder_pd_t **, engine_t *,
            const primitive_attr_t *, engine_t *, const memory_desc_t *,
            engine_t *, const memory_desc_t *);

    create_pd_func_t create_pd_func_ = nullptr;
    create_concat_pd_func_t create_concat_pd_func_ = nullptr;
    create_sum_pd_func_t create_sum_pd_func_ = nullptr;
    create_reorder_pd_func_t create_reorder_pd_func_ = nullptr;

    // List of functions/classes that have permissions to create primitive
    // descriptors.
    friend struct primitive_desc_iterator_t;
    friend status_t concat_primitive_desc_create(
            std::shared_ptr<primitive_desc_t> &, engine_t *,
            const memory_desc_t *, int, int, const memory_desc_t *const *,
            const primitive_attr_t *);
    friend status_t sum_primitive_desc_create(primitive_desc_iface_t **,
            const memory_desc_t *, int, const float *,
            const memory_desc_t *const *, const primitive_attr_t *, engine_t *);
    friend status_t reorder_primitive_desc_create(
            std::shared_ptr<primitive_desc_t> &, engine_t *,
            const memory_desc_t *, engine_t *, const memory_desc_t *,
            engine_t *, const primitive_attr_t *);
};

} // namespace impl
} // namespace dnnl

#endif
