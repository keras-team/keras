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

#ifndef COMMON_ENGINE_ID_HPP
#define COMMON_ENGINE_ID_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

struct engine_id_impl_t;

struct engine_id_t {
    engine_id_t(engine_id_impl_t *impl) : impl_(impl) {}

    engine_id_t() = default;
    engine_id_t(engine_id_t &&other) = default;
    engine_id_t(const engine_id_t &other) = default;
    engine_id_t &operator=(const engine_id_t &other) = default;
    virtual ~engine_id_t() = default;

    bool operator==(const engine_id_t &other) const;
    size_t hash() const;
    engine_kind_t kind() const;
    runtime_kind_t runtime_kind() const;

    operator bool() const { return bool(impl_); }

private:
    std::shared_ptr<engine_id_impl_t> impl_;
};

struct engine_id_impl_t {
    engine_id_impl_t() = delete;
    engine_id_impl_t(
            engine_kind_t kind, runtime_kind_t runtime_kind, size_t index)
        : kind_(kind), runtime_kind_(runtime_kind), index_(index) {}

    virtual ~engine_id_impl_t() = default;

    bool compare(const engine_id_impl_t *id_impl) {
        bool ret = kind_ == id_impl->kind_
                && runtime_kind_ == id_impl->runtime_kind_
                && index_ == id_impl->index_;
        if (!ret) return ret;
        return compare_resource(id_impl);
    }

    size_t hash() const {
        size_t seed = 0;
        seed = hash_combine(seed, static_cast<size_t>(kind_));
        seed = hash_combine(seed, static_cast<size_t>(runtime_kind_));
        seed = hash_combine(seed, index_);
        return hash_combine(seed, hash_resource());
    }

    engine_kind_t kind() const { return kind_; }
    runtime_kind_t runtime_kind() const { return runtime_kind_; }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(engine_id_impl_t);

    engine_kind_t kind_;
    runtime_kind_t runtime_kind_;
    size_t index_;

    virtual bool compare_resource(const engine_id_impl_t *id_impl) const = 0;
    virtual size_t hash_resource() const = 0;
};

inline bool engine_id_t::operator==(const engine_id_t &other) const {
    // All regular CPU engines are considered equal.
    if (utils::everyone_is(nullptr, impl_, other.impl_)) return true;

    if (utils::one_of(nullptr, impl_, other.impl_)) return false;
    return impl_->compare(other.impl_.get());
}

inline size_t engine_id_t::hash() const {
    // It doesn't make much sense to calculate hash for regular CPU engines
    // because the hash will always be the same.
    if (!impl_) return 0;
    return impl_->hash();
}

inline engine_kind_t engine_id_t::kind() const {
    if (!impl_) return engine_kind::any_engine;
    return impl_->kind();
}

inline runtime_kind_t engine_id_t::runtime_kind() const {
    if (!impl_) return runtime_kind::none;
    return impl_->runtime_kind();
}

} // namespace impl
} // namespace dnnl

namespace std {
template <>
struct hash<dnnl::impl::engine_id_t> {
    std::size_t operator()(const dnnl::impl::engine_id_t &id) const {
        return id.hash();
    }
};
} // namespace std

#endif
