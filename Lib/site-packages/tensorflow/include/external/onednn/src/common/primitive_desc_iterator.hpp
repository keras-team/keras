/*******************************************************************************
* Copyright 2018-2023 Intel Corporation
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

#ifndef COMMON_PRIMITIVE_ITERATOR_HPP
#define COMMON_PRIMITIVE_ITERATOR_HPP

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "impl_list_item.hpp"
#include "primitive_attr.hpp"
#include "primitive_cache.hpp"
#include "primitive_hashing.hpp"
#include "type_helpers.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;

struct primitive_desc_iterator_t : public c_compatible {
    primitive_desc_iterator_t(engine_t *engine, const op_desc_t *op_desc,
            const primitive_attr_t *attr, const primitive_desc_t *hint_fwd_pd,
            int skip_idx = -1)
        : idx_(-1)
        , engine_(engine)
        , op_desc_(nullptr)
        , attr_(attr ? *attr : primitive_attr_t())
        , hint_fwd_pd_(hint_fwd_pd)
        , impl_list_(nullptr)
        , last_idx_(0)
        , skip_idx_(skip_idx)
        , offset_(-1) {

        op_desc_ = (op_desc_t *)std::malloc(sizeof(op_desc_t));
        copy_c_op_desc(op_desc_, op_desc);

        impl_list_ = engine_->get_implementation_list(op_desc_);

        while (impl_list_[last_idx_])
            ++last_idx_;
        is_initialized_ = is_initialized_ && attr_.is_initialized();
    }

    ~primitive_desc_iterator_t() { std::free(op_desc_); }

    engine_t *engine() const { return engine_; }

    bool operator==(const primitive_desc_iterator_t &rhs) const {
        return idx_ == rhs.idx_ && engine_ == rhs.engine_;
    }
    bool operator!=(const primitive_desc_iterator_t &rhs) const {
        return !operator==(rhs);
    }

    primitive_desc_iterator_t end() const {
        return primitive_desc_iterator_t(engine_, last_idx_);
    }

    primitive_desc_iterator_t &operator++() {
        // Quick return to preserve state of the iterator that reached the end.
        // The state is equal to the state of the iterator that end() returns.
        if (idx_ == last_idx_) return *this;

        offset_++;
        pd_.reset();

        std::vector<dnnl::impl::memory_desc_t> hint_mds;
        if (hint_fwd_pd_) hint_mds = hint_fwd_pd_->hint_mds(true /* is_hint */);
        primitive_hashing::key_t key(
                engine_, op_desc_, &attr_, offset_, hint_mds, skip_idx_);

        pd_ = primitive_cache().get_pd(key);
        if (pd_) { return *this; }

        while (++idx_ != last_idx_) {
            if (idx_ == skip_idx_) continue;
            primitive_desc_t *candidate_pd = nullptr;
            auto s = impl_list_[idx_](&candidate_pd, op_desc_, &attr_, engine_,
                    hint_fwd_pd_, offset_, skip_idx_);
            if (s == status::success) {
                pd_.reset(candidate_pd);
                break;
            }
        }
        return *this;
    }

    const std::shared_ptr<primitive_desc_t> &operator*() const { return pd_; }

    const primitive_attr_t &attr() const { return attr_; }

    bool is_initialized() const { return is_initialized_; }

protected:
    int idx_;
    engine_t *engine_;
    std::shared_ptr<primitive_desc_t> pd_;
    op_desc_t *op_desc_;
    const primitive_attr_t attr_;
    const primitive_desc_t *hint_fwd_pd_;
    const impl_list_item_t *impl_list_;
    int last_idx_;
    int skip_idx_;
    int offset_;

private:
    primitive_desc_iterator_t(engine_t *engine, int last_idx)
        : idx_(last_idx)
        , engine_(engine)
        , op_desc_(nullptr)
        , hint_fwd_pd_(nullptr)
        , impl_list_(nullptr)
        , last_idx_(last_idx)
        , skip_idx_(-1)
        , offset_(-1) {}

    primitive_desc_iterator_t(primitive_desc_iterator_t &&other)
        : idx_(other.idx_)
        , engine_(other.engine_)
        , pd_(std::move(other.pd_))
        , op_desc_(other.op_desc_)
        , attr_(other.attr_)
        , hint_fwd_pd_(other.hint_fwd_pd_)
        , impl_list_(other.impl_list_)
        , skip_idx_(other.skip_idx_)
        , offset_(other.offset_) {}

    DNNL_DISALLOW_COPY_AND_ASSIGN(primitive_desc_iterator_t);
};

} // namespace impl
} // namespace dnnl

#endif
