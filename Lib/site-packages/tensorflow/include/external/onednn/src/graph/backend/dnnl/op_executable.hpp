/*******************************************************************************
 * Copyright 2021-2024 Intel Corporation
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
#ifndef BACKEND_DNNL_OP_EXECUTABLE_HPP
#define BACKEND_DNNL_OP_EXECUTABLE_HPP

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>
#include <unordered_map>

#include "oneapi/dnnl/dnnl.hpp"
#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <CL/sycl.hpp>
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "graph/utils/ocl_check.hpp"
#include "graph/utils/ocl_usm_utils.hpp"

#include "gpu/intel/ocl/ocl_usm_utils.hpp"

#include "oneapi/dnnl/dnnl_ocl.hpp"
#endif

#include <graph/utils/utils.hpp>

#include "graph/interface/backend.hpp"

#include "graph/backend/dnnl/common.hpp"
#include "graph/backend/dnnl/fusion_info.hpp"
#include "graph/backend/dnnl/internal_attrs.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

struct indices_t {
    // the type_t is used to indicate the indices is for input or output
    enum class type_t {
        input = 0,
        output = 1,
    };

    type_t type_;
    size_t value_;
};

// DNNL arg to in/outputs indices mapping. For example, <DNNL_ARG_SRC, {input,
// 0}> means the 0-th input of an op should be used as primitive's src argument.
// We should be able to know this map according the information on an op.
using arg_indices_t = std::unordered_map<int, indices_t>;

using arg_indices_getter_func
        = std::function<arg_indices_t(const op_t *, fusion_info_mgr_t &)>;

// A dummy arg indices getter which is only used for those internal ops that are
// only for fusion purpose, like dnnl_add_zps and dnnl_sub_zps. The dummy getter
// should never be called.
inline arg_indices_t dummy_arg_indices_getter(
        const op_t *op, fusion_info_mgr_t &mgr) {
    UNUSED(op);
    UNUSED(mgr);
    assertm(false, "dummy getter should never be called");
    return arg_indices_t {};
}

// Used to declare the arg indices getter inside an op executable class. The
// getter can be used to generate the <dnnl_arg, in/output index> map. According
// to that, we can form the execution args by using the in/outputs list in op.
#define DECLARE_ARG_INDICES_GETTER \
    static arg_indices_t get_arg_indices( \
            const op_t *op, fusion_info_mgr_t &mgr);

struct op_executable_t {
    virtual ~op_executable_t() = default;
    virtual void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const = 0;
#ifdef DNNL_WITH_SYCL
    virtual ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const = 0;
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    virtual cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const = 0;
#endif
};

using executable_creator_func = std::function<std::shared_ptr<op_executable_t>(
        std::shared_ptr<op_t> &, const dnnl::engine &, fusion_info_mgr_t &,
        pd_cache_t &)>;

// A dummy executable creator which is only used for those internal ops that are
// only for fusion purpose, like dnnl_add_zps and dnnl_sub_zps. The dummy
// creator should never be called.
inline std::shared_ptr<op_executable_t> dummy_executable_creator(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    UNUSED(op);
    UNUSED(p_engine);
    UNUSED(mgr);
    UNUSED(pd_cache);
    assertm(false, "dummy executable creator should never be called");
    return {};
}

// A general template executable fcreator function, which can be specialized by
// using different op executable class types
template <typename T>
inline std::shared_ptr<op_executable_t> executable_creator(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
    return std::make_shared<T>(op, p_engine, mgr, pd_cache);
}

// Used to declare the desc_t class and the static create_desc method inside an
// op executable class
#define DECLARE_DESC_CLASS_AND_CREATOR(primitive_desc) \
    using type = primitive_desc; /* NOLINT */ \
    class desc_t : public type { \
        bool from_cache_; \
\
    public: \
        desc_t(const type &pd, bool from_cache) \
            : type(pd), from_cache_(from_cache) {} \
        bool is_from_cache() const { return from_cache_; } \
    }; \
    static desc_t create_desc(std::shared_ptr<op_t> &op, \
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr, \
            pd_cache_t &pd_cache);

// This class is a dummy executable which doesn't do any actual computation.
// This dummy executable can be used to:
// - support data formatting ops like permute/reshape/transpose
// - support zero-volume tensor (empty tensor) like (1024, 64)x(64, 0)
//
// In the execute_sycl fuction, we will run a dummy sycl kernel to gather all
// the input events
struct dummy_impl_t : public op_executable_t {
    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(stream);
        UNUSED(args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        UNUSED(stream);

        // Fast path: if no event, return an immediate event.
        if (deps.empty()) return {};

        // Fast path: if only one event, return it.
        if (deps.size() == 1) return deps[0];

        // Otherwise, we run a trivial kernel to gather all deps. The
        // dummy task is needed to not get an error related to empty
        // kernel.
        auto q = dnnl::sycl_interop::get_queue(stream);
        auto e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(deps);
            cgh.single_task<class dnnl_graph_dummy_kernel>([]() {});
        });
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        UNUSED(stream);

        // Fast path: if no event, return an immediate event.
        if (deps.empty()) return {};

        // Fast path: if only one event, return it.
        if (deps.size() == 1) return deps[0];

        // Otherwise, gather all dependencies.
        auto q = dnnl::ocl_interop::get_command_queue(stream);
        cl_event e;
        auto err = clEnqueueMarkerWithWaitList(
                q, static_cast<cl_uint>(deps.size()), deps.data(), &e);
        assert(err == CL_SUCCESS);
        MAYBE_UNUSED(err);
        return e;
    }
#endif
};

struct memory_reparser_t : public dummy_impl_t {
    DECLARE_ARG_INDICES_GETTER;

    memory_reparser_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        UNUSED(op);
        UNUSED(p_engine);
        UNUSED(mgr);
        UNUSED(pd_cache);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        assertm(args.find(DNNL_ARG_FROM)->second.get_data_handle()
                        == args.find(DNNL_ARG_TO)->second.get_data_handle(),
                "memory reparser must be inplaced");
        dummy_impl_t::execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        assertm(args.find(DNNL_ARG_FROM)->second.get_data_handle()
                        == args.find(DNNL_ARG_TO)->second.get_data_handle(),
                "memory reparser must be inplaced");
        return dummy_impl_t::execute_sycl(stream, args, deps);
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        assertm(args.find(DNNL_ARG_FROM)->second.get_data_handle()
                        == args.find(DNNL_ARG_TO)->second.get_data_handle(),
                "memory reparser must be inplaced");
        return dummy_impl_t::execute_ocl(stream, args, deps);
    }
#endif
};

template <op_attr_t attr_name, typename attr_dt, typename target_dt>
struct const_memory_filler_t : public op_executable_t {
    static arg_indices_t get_arg_indices(
            const op_t *op, fusion_info_mgr_t &mgr) {
        UNUSED(mgr);
        arg_indices_t arg_indices;
        // We only set dst argument, to which constant data will be copied
        arg_indices.insert(
                {DNNL_ARG_TO, indices_t {indices_t::type_t::output, 0}});
        return arg_indices;
    }

    const_memory_filler_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        UNUSED(p_engine);
        UNUSED(mgr);
        UNUSED(pd_cache);
        attr_data_
                = get_attr_data(op->get_attr<std::vector<attr_dt>>(attr_name),
                        std::is_same<attr_dt, target_dt>());
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        void *data_handle = static_cast<void *>(
                const_cast<target_dt *>(attr_data_.data()));
        auto it = args.find(DNNL_ARG_TO);
        if (it == args.end()) {
            // TODO(xxx): we should propagate the error by returning a status.
            assert(!"cannot find memory for DNNL_ARG_TO");
            return;
        }
        const memory &dst_mem = it->second;

        auto is_cpu = dst_mem.get_engine().get_kind() == engine::kind::cpu;
        // handle cross-engine case
        auto src_eng = (is_cpu) ? dst_mem.get_engine()
                                : engine(dflt_eng_kind, dflt_eng_idx);

        const memory src_mem
                = make_dnnl_memory(dst_mem.get_desc(), src_eng, data_handle);
        dnnl::reorder(src_mem, dst_mem)
                .execute(stream, const_cast<memory &>(src_mem),
                        const_cast<memory &>(dst_mem));
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        void *data_handle = static_cast<void *>(
                const_cast<target_dt *>(attr_data_.data()));
        const memory &dst_mem = args.find(DNNL_ARG_TO)->second;
        auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        auto e = sycl_queue.memcpy(dst_mem.get_data_handle(), data_handle,
                dst_mem.get_desc().get_size());
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        void *data_handle = static_cast<void *>(
                const_cast<target_dt *>(attr_data_.data()));
        const memory &dst_mem = args.find(DNNL_ARG_TO)->second;
        assert(deps.size() <= 1);
        // Passing the empty event to memcpy below causes failure.
        const bool empty = deps.size() == 0 || deps[0] == 0;
        const cl_uint num = empty ? 0 : static_cast<cl_uint>(deps.size());
        cl_event e;
        UNUSED_STATUS(gpu::intel::ocl::usm::memcpy(stream.get(),
                dst_mem.get_data_handle(), data_handle,
                dst_mem.get_desc().get_size(), num,
                empty ? nullptr : deps.data(), &e));
        return e;
    }
#endif

private:
    std::vector<target_dt> get_attr_data(
            const std::vector<attr_dt> &orig_data, std::true_type) {
        return orig_data;
    }
    std::vector<target_dt> get_attr_data(
            const std::vector<attr_dt> &orig_data, std::false_type) {
        return std::vector<target_dt>(orig_data.begin(), orig_data.end());
    }

    const engine::kind dflt_eng_kind = engine::kind::cpu;
    const size_t dflt_eng_idx = 0;
    std::vector<target_dt> attr_data_;
};

using const_scales_filler
        = const_memory_filler_t<op_attr::scales, float, float>;
using const_zps_filler = const_memory_filler_t<op_attr::zps, int64_t, int32_t>;

extern "C" dnnl_status_t dnnl_memory_desc_create_with_string_tag(
        dnnl_memory_desc_t *, int, const dnnl_dims_t, dnnl_data_type_t,
        const char *);

struct conv_fwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::convolution_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    conv_fwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::convolution_forward(desc);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                // psrc_mem and dst_mem may have different data type bug same
                // buffer size(u8 and s8) for such case, need to reorder
                // psrc_mem to dst_mem with original data type
                if (psrc_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::s8
                        && dst_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::u8) {
                    dnnl::memory::desc to_desc = dst_mem.get_desc();
                    auto format_tag = get_format_tag_str(to_desc);
                    const auto &dims = to_desc.get_dims();
                    const auto &dtype = psrc_mem.get_desc().get_data_type();
                    dnnl_memory_desc_t new_to_desc_c;
                    dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                            static_cast<int>(dims.size()), dims.data(),
                            static_cast<dnnl_data_type_t>(dtype),
                            format_tag.data());
                    dnnl::memory::desc new_to_desc;
                    new_to_desc.reset(new_to_desc_c);
                    const memory to_mem
                            = dnnl::memory(new_to_desc, psrc_mem.get_engine());
                    to_mem.set_data_handle(dst_mem.get_data_handle());
                    dnnl::reorder(psrc_mem, to_mem)
                            .execute(stream, const_cast<memory &>(psrc_mem),
                                    const_cast<memory &>(to_mem));
                } else {
                    dnnl::reorder(psrc_mem, dst_mem)
                            .execute(stream, const_cast<memory &>(psrc_mem),
                                    const_cast<memory &>(dst_mem));
                }
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                // psrc_mem and dst_mem may have different data type bug same
                // buffer size(u8 and s8) for such case, need to reorder
                // psrc_mem to dst_mem with original data type
                if (psrc_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::s8
                        && dst_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::u8) {
                    dnnl::memory::desc to_desc = dst_mem.get_desc();
                    auto format_tag = get_format_tag_str(to_desc);
                    const auto &dims = to_desc.get_dims();
                    const auto &dtype = psrc_mem.get_desc().get_data_type();
                    dnnl_memory_desc_t new_to_desc_c;
                    dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                            static_cast<int>(dims.size()), dims.data(),
                            static_cast<dnnl_data_type_t>(dtype),
                            format_tag.data());
                    dnnl::memory::desc new_to_desc;
                    new_to_desc.reset(new_to_desc_c);
                    const memory to_mem
                            = dnnl::memory(new_to_desc, psrc_mem.get_engine());
                    to_mem.set_data_handle(dst_mem.get_data_handle());
                    auto prim = dnnl::reorder(psrc_mem, to_mem);
                    auto e = dnnl::sycl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(to_mem)}},
                            sycl_deps);
                    sycl_deps = {e};
                    if (stream.get_engine().get_kind() == engine::kind::cpu)
                        e.wait();
                } else {
                    auto prim = dnnl::reorder(psrc_mem, dst_mem);
                    auto e = dnnl::sycl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(dst_mem)}},
                            sycl_deps);
                    sycl_deps = {e};
                }
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto ocl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                // psrc_mem and dst_mem may have different data type bug same
                // buffer size(u8 and s8) for such case, need to reorder
                // psrc_mem to dst_mem with original data type
                if (psrc_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::s8
                        && dst_mem.get_desc().get_data_type()
                                == dnnl::memory::data_type::u8) {
                    dnnl::memory::desc to_desc = dst_mem.get_desc();
                    auto format_tag = get_format_tag_str(to_desc);
                    const auto &dims = to_desc.get_dims();
                    const auto &dtype = psrc_mem.get_desc().get_data_type();
                    dnnl_memory_desc_t new_to_desc_c;
                    dnnl_memory_desc_create_with_string_tag(&new_to_desc_c,
                            static_cast<int>(dims.size()), dims.data(),
                            static_cast<dnnl_data_type_t>(dtype),
                            format_tag.data());
                    dnnl::memory::desc new_to_desc;
                    new_to_desc.reset(new_to_desc_c);

                    const memory to_mem
                            = dnnl::ocl_interop::get_memory_kind(dst_mem)
                                    == dnnl::ocl_interop::memory_kind::usm
                            ? dnnl::ocl_interop::make_memory(new_to_desc,
                                    psrc_mem.get_engine(),
                                    dnnl::ocl_interop::memory_kind::usm,
                                    dst_mem.get_data_handle())
                            : dnnl::ocl_interop::make_memory(new_to_desc,
                                    psrc_mem.get_engine(),
                                    reinterpret_cast<cl_mem>(
                                            dst_mem.get_data_handle()));

                    auto prim = dnnl::reorder(psrc_mem, to_mem);
                    auto e = dnnl::ocl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(to_mem)}},
                            ocl_deps);
                    ocl_deps = {e};
                } else {
                    auto prim = dnnl::reorder(psrc_mem, dst_mem);
                    auto e = dnnl::ocl_interop::execute(prim, stream,
                            {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                    {DNNL_ARG_TO,
                                            const_cast<memory &>(dst_mem)}},
                            ocl_deps);
                    ocl_deps = {e};
                }
            }
        }
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::convolution_forward prim_;
    bool with_sum_ {false};
};

struct deconv_fwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::deconvolution_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    deconv_fwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::deconvolution_forward(desc);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto ocl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::ocl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps);
                ocl_deps = {e};
            }
        }
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::deconvolution_forward prim_;
    bool with_sum_ {false};
};

struct deconv_bwd_data_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::deconvolution_backward_data::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    deconv_bwd_data_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::deconvolution_backward_data(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::deconvolution_backward_data prim_;
};

struct deconv_bwd_weights_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::deconvolution_backward_weights::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    deconv_bwd_weights_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::deconvolution_backward_weights(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::deconvolution_backward_weights prim_;
};

struct matmul_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::matmul::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    matmul_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        using ltw = logical_tensor_wrapper_t;
        // if with zero dimension, the matmul op will take no effect, we
        // construct a dummy kernel
        if (ltw(op->get_input_value(0)->get_logical_tensor()).has_zero_dim()
                || ltw(op->get_input_value(1)->get_logical_tensor())
                           .has_zero_dim()) {
            is_dummy_ = true;
            return;
        }

        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::matmul(desc);

        // The scratchpad size of pd created by using any format tag may be
        // different from the scratchpad size of pd created by using queried
        // optimal format tag
        dnnl::memory::desc stored = make_dnnl_memory_desc(
                op->get_output_value(1)->get_logical_tensor());
        dnnl::memory::desc real = desc.scratchpad_desc();
        if (stored != real) {
            auto scratchpad_val = op->get_output_value(1);
            scratchpad_val->set_layout_type(layout_type::any);
            fill_layout_info(scratchpad_val, real);
        }

        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (is_dummy_) {
            dummy_impl_.execute(stream, args);
            return;
        }

        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return;
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, psrc_mem, dst_mem);
            }
        }
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        if (is_dummy_) { return dummy_impl_.execute_sycl(stream, args, deps); }

        auto sycl_deps = deps;
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return {};
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        if (is_dummy_) { return dummy_impl_.execute_ocl(stream, args, deps); }

        auto ocl_deps = deps;
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return {};
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::ocl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps);
                ocl_deps = {e};
            }
        }
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::matmul prim_;
    bool with_sum_ {false};
    bool is_dummy_ {false};
    dummy_impl_t dummy_impl_;
};

struct eltwise_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::eltwise_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    eltwise_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::eltwise_forward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::eltwise_forward prim_;
};

struct eltwise_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::eltwise_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    eltwise_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::eltwise_backward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::eltwise_backward prim_;
};

struct binary_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::binary::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    binary_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        using ltw = logical_tensor_wrapper_t;
        // if with zero dimension, the binary op will take no effect, we
        // construct a dummy kernel
        if (ltw(op->get_input_value(0)->get_logical_tensor()).has_zero_dim()
                || ltw(op->get_input_value(1)->get_logical_tensor())
                           .has_zero_dim()) {
            is_dummy_ = true;
            return;
        }

        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::binary(desc);

        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (is_dummy_) {
            dummy_impl_.execute(stream, args);
            return;
        }

        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return;
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, psrc_mem, dst_mem);
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        if (is_dummy_) { return dummy_impl_.execute_sycl(stream, args, deps); }

        auto sycl_deps = deps;
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return {};
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        if (is_dummy_) { return dummy_impl_.execute_ocl(stream, args, deps); }

        auto ocl_deps = deps;
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return {};
            }

            memory &dst_mem = const_cast<memory &>(it_dst->second);
            memory &psrc_mem = const_cast<memory &>(it_src->second);

            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::ocl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps);
                ocl_deps = {e};
            }
        }
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::binary prim_;
    bool with_sum_ {false};
    bool is_dummy_ {false};
    dummy_impl_t dummy_impl_;
};

struct concat_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::concat::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    concat_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::concat(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::concat prim_;
};

struct shuffle_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::shuffle_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    shuffle_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::shuffle_forward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::shuffle_forward prim_;
};

struct pool_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::pooling_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    pool_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::pooling_forward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::pooling_forward prim_;
};

struct pool_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::pooling_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    pool_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::pooling_backward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::pooling_backward prim_;
};

struct prelu_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::prelu_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    prelu_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::prelu_forward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::prelu_forward prim_;
};

struct prelu_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::prelu_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    prelu_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::prelu_backward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::prelu_backward prim_;
};

struct reorder_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::reorder::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    reorder_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::reorder(desc);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return;
            }

            const memory &psrc_mem = it_src->second;
            const memory &dst_mem = it_dst->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return {};
            }

            const memory &psrc_mem = it_src->second;
            const memory &dst_mem = it_dst->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto ocl_deps = deps;
        if (with_sum_) {
            auto it_dst = args.find(DNNL_ARG_DST);
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            if (it_dst == args.end() || it_src == args.end()) {
                assert(!("cannot find the required memory"));
                return {};
            }

            const memory &psrc_mem = it_src->second;
            const memory &dst_mem = it_dst->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::ocl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps);
                ocl_deps = {e};
            }
        }
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::reorder prim_;
    bool with_sum_ {false};
};

struct bn_folding_t : public op_executable_t {
    DECLARE_ARG_INDICES_GETTER

    // bn_folding_t is a aggregated executable by using multiple primitives, so
    // we need a customized desc class to describe it.
    class desc_t {
        friend struct bn_folding_t;

        float epsilon_ = 1e-5f;
        std::string data_format_;
        std::string filter_format_;

        memory::desc epsilon_desc_;
        memory::desc new_scale_desc_;
        memory::desc new_variance_desc_;
        memory::desc scratchpad_desc_;

        dnnl::binary::primitive_desc add_pd_;
        dnnl::binary::primitive_desc mul_pd_;
        dnnl::binary::primitive_desc sub_pd_;

        bool with_bias_ {false};

    public:
        const memory::desc &scratchpad_desc() const { return scratchpad_desc_; }
    };

    static desc_t create_desc(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache);

    bn_folding_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        desc_ = create_desc(op, p_engine, mgr, pd_cache);
        add_prim_ = dnnl::binary(desc_.add_pd_);
        mul_prim_ = dnnl::binary(desc_.mul_pd_);
        sub_prim_ = dnnl::binary(desc_.sub_pd_);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second
                                     : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance
        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = make_dnnl_memory(variance.get_desc(),
                    scratchpad.get_engine(), (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epsilon
        memory epsilon_mem = make_dnnl_memory(desc_.epsilon_desc_,
                scratchpad.get_engine(), (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        if (variance.get_engine().get_kind() == engine::kind::cpu) {
            float *ptr = (float *)epsilon_mem.get_data_handle();
            *ptr = desc_.epsilon_;
        } else {
            engine cpu_eng(engine::kind::cpu, 0);
            memory cpu_mem = make_dnnl_memory(
                    desc_.epsilon_desc_, cpu_eng, (void *)&desc_.epsilon_);
            dnnl::reorder(cpu_mem, epsilon_mem)
                    .execute(stream, cpu_mem, epsilon_mem);
        }

        add_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}});

        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale(desc_.new_scale_desc_, scale.get_engine(),
                scale.get_data_handle());
        memory new_sqrt_variance(desc_.new_variance_desc_,
                sqrt_variance.get_engine(), sqrt_variance.get_data_handle());
        mul_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
            if (mean.get_engine().get_kind() == engine::kind::cpu) {
                std::memcpy(valid_bias.get_data_handle(), zero.data(),
                        valid_bias.get_desc().get_size());
            } else {
                engine cpu_eng(engine::kind::cpu, 0);
                memory cpu_mem = make_dnnl_memory(
                        variance.get_desc(), cpu_eng, zero.data());
                dnnl::reorder(cpu_mem, valid_bias)
                        .execute(stream, cpu_mem, valid_bias);
            }
        }

        sub_prim_.execute(stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}});
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second
                                     : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance
        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = make_dnnl_memory(variance.get_desc(),
                scratchpad.get_engine(), (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = make_dnnl_memory(variance.get_desc(),
                    scratchpad.get_engine(), (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epsilon
        memory epsilon_mem = make_dnnl_memory(desc_.epsilon_desc_,
                scratchpad.get_engine(), (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        auto sycl_queue = dnnl::sycl_interop::get_queue(stream);
        sycl_queue
                .memcpy(epsilon_mem.get_data_handle(), &desc_.epsilon_,
                        epsilon_mem.get_desc().get_size())
                .wait();

        auto sycl_deps = dnnl::sycl_interop::execute(add_prim_, stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}},
                deps);

        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale(desc_.new_scale_desc_, scale.get_engine(),
                scale.get_data_handle());
        memory new_sqrt_variance(desc_.new_variance_desc_,
                sqrt_variance.get_engine(), sqrt_variance.get_data_handle());

        auto sycl_deps2 = dnnl::sycl_interop::execute(mul_prim_, stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}},
                {sycl_deps});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
            sycl_queue
                    .memcpy(valid_bias.get_data_handle(), zero.data(),
                            valid_bias.get_desc().get_size())
                    .wait();
            auto sycl_deps3 = dnnl::sycl_interop::execute(sub_prim_, stream,
                    {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                            {DNNL_ARG_DST, updated_bias},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                    scale},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                    sqrt_variance},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                    shift}},
                    {sycl_deps2});
            if (stream.get_engine().get_kind() == engine::kind::cpu)
                sycl_deps3.wait();
            return sycl_deps3;
        }

        auto sycl_deps3 = dnnl::sycl_interop::execute(sub_prim_, stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}},
                {sycl_deps2});
        if (stream.get_engine().get_kind() == engine::kind::cpu)
            sycl_deps3.wait();
        return sycl_deps3;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        UNUSED(args);

        auto weights = args.find(DNNL_ARG_WEIGHTS)->second;
        auto bias = desc_.with_bias_ ? args.find(DNNL_ARG_BIAS)->second
                                     : memory();
        auto scale = args.find(DNNL_ARG_WEIGHTS_1)->second;
        auto shift = args.find(DNNL_ARG_WEIGHTS_2)->second;
        auto mean = args.find(DNNL_ARG_MEAN)->second;
        auto variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto scratchpad = args.find(DNNL_ARG_SCRATCHPAD)->second;

        auto updated_weights = args.find(DNNL_ARG_DST_0)->second;
        auto updated_bias = args.find(DNNL_ARG_DST_1)->second;

        // 0. split scratchpad buffer to specific intermediate memory
        // sqrt_variance

        char *buf_start = (char *)scratchpad.get_data_handle();
        memory sqrt_variance = dnnl::ocl_interop::make_memory(
                variance.get_desc(), scratchpad.get_engine(),
                dnnl::ocl_interop::memory_kind::usm, (void *)buf_start);
        buf_start += variance.get_desc().get_size();
        // zero_bias
        memory valid_bias = bias;
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            valid_bias = dnnl::ocl_interop::make_memory(variance.get_desc(),
                    scratchpad.get_engine(),
                    dnnl::ocl_interop::memory_kind::usm, (void *)buf_start);
            buf_start += valid_bias.get_desc().get_size();
        }
        // epsilon
        memory epsilon_mem = dnnl::ocl_interop::make_memory(desc_.epsilon_desc_,
                scratchpad.get_engine(), dnnl::ocl_interop::memory_kind::usm,
                (void *)buf_start);

        // 1. sqrt_variance = sqrt(variance + epsilon)
        cl_event e;
        gpu::intel::ocl::usm::memcpy(stream.get(),
                epsilon_mem.get_data_handle(), &desc_.epsilon_,
                epsilon_mem.get_desc().get_size(), 0, nullptr, &e);
        clWaitForEvents(1, &e);

        auto ocl_deps = dnnl::ocl_interop::execute(add_prim_, stream,
                {{DNNL_ARG_SRC_0, variance}, {DNNL_ARG_SRC_1, epsilon_mem},
                        {DNNL_ARG_DST, sqrt_variance}},
                deps);

        // 2. updated_weight = weights * scale / sqrt_variance
        memory new_scale = dnnl::ocl_interop::make_memory(desc_.new_scale_desc_,
                scale.get_engine(), dnnl::ocl_interop::memory_kind::usm,
                scale.get_data_handle());
        memory new_sqrt_variance = dnnl::ocl_interop::make_memory(
                desc_.new_variance_desc_, sqrt_variance.get_engine(),
                dnnl::ocl_interop::memory_kind::usm,
                sqrt_variance.get_data_handle());

        auto ocl_deps2 = dnnl::ocl_interop::execute(mul_prim_, stream,
                {{DNNL_ARG_SRC_0, weights}, {DNNL_ARG_SRC_1, new_scale},
                        {DNNL_ARG_DST, updated_weights},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                new_sqrt_variance}},
                {ocl_deps});

        // 3. updated_bias = (bias - mean) * scale / sqrt_variance + shift
        if (bias.get(true) == nullptr || bias.get_data_handle() == nullptr) {
            // initialize the bias with zero value
            std::vector<float> zero(
                    graph::utils::prod(variance.get_desc().get_dims()), 0.0f);
            gpu::intel::ocl::usm::memcpy(stream.get(),
                    valid_bias.get_data_handle(), zero.data(),
                    valid_bias.get_desc().get_size(), 0, nullptr, &e);
            clWaitForEvents(1, &e);

            auto ocl_deps3 = dnnl::ocl_interop::execute(sub_prim_, stream,
                    {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                            {DNNL_ARG_DST, updated_bias},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                    scale},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                    sqrt_variance},
                            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                    shift}},
                    {ocl_deps2});
            return ocl_deps3;
        }

        auto ocl_deps3 = dnnl::ocl_interop::execute(sub_prim_, stream,
                {{DNNL_ARG_SRC_0, valid_bias}, {DNNL_ARG_SRC_1, mean},
                        {DNNL_ARG_DST, updated_bias},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                                scale},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1,
                                sqrt_variance},
                        {DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1,
                                shift}},
                {ocl_deps2});
        return ocl_deps3;
    }
#endif

private:
    desc_t desc_;
    dnnl::binary add_prim_;
    dnnl::binary mul_prim_;
    dnnl::binary sub_prim_;
};

struct conv_bwd_data_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::convolution_backward_data::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    conv_bwd_data_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::convolution_backward_data(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::convolution_backward_data prim_;
};

struct conv_bwd_weights_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::convolution_backward_weights::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    conv_bwd_weights_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::convolution_backward_weights(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::convolution_backward_weights prim_;
};

struct batchnorm_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::batch_normalization_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    batchnorm_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        is_training_ = op->get_attr<bool>(op_attr::is_training);
        float momentum = 0.5;
        if (op->has_attr(op_attr::momentum))
            momentum = op->get_attr<float>(op_attr::momentum);
        scales_ = {momentum, 1 - momentum};
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::batch_normalization_forward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (!is_training_) {
            prim_.execute(stream, args);
            return;
        }

        std::unordered_map<int, memory> exe_args = args;
        exe_args.erase(DNNL_ARG_SRC_1);
        exe_args.erase(DNNL_ARG_SRC_2);
        exe_args.erase(DNNL_ARG_DST_1);
        exe_args.erase(DNNL_ARG_DST_2);

        prim_.execute(stream, exe_args);

        // calculate running_mean and running_variance
        auto it_mean = args.find(DNNL_ARG_MEAN);
        auto it_var = args.find(DNNL_ARG_VARIANCE);
        auto it_src1 = args.find(DNNL_ARG_SRC_1);
        auto it_src2 = args.find(DNNL_ARG_SRC_2);
        auto it_dst1 = args.find(DNNL_ARG_DST_1);
        auto it_dst2 = args.find(DNNL_ARG_DST_2);

        if (graph::utils::one_of(args.end(), it_mean, it_var, it_src1, it_src2,
                    it_dst1, it_dst2)) {
            assert(!"cannot find one of the required memories");
            return;
        }

        auto batch_mean = it_mean->second;
        auto batch_variance = it_var->second;
        auto old_running_mean = it_src1->second;
        auto old_running_variance = it_src2->second;
        auto new_running_mean = it_dst1->second;
        auto new_running_variance = it_dst2->second;

        dnnl::engine p_engine = stream.get_engine();
        // new_running_mean = momentum * old_running_mean +
        //                                      (1 - momentum) * batch_mean
        dnnl::sum({p_engine, scales_,
                          {old_running_mean.get_desc(), batch_mean.get_desc()}})
                .execute(stream,
                        {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                                {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                                {DNNL_ARG_DST, new_running_mean}});
        // new_running_variance = momentum * old_running_variance +
        //                                  (1 - momentum) * batch_variance
        dnnl::sum({p_engine, scales_,
                          {old_running_variance.get_desc(),
                                  batch_variance.get_desc()}})
                .execute(stream,
                        {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                                {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                                {DNNL_ARG_DST, new_running_variance}});
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        if (!is_training_) {
            auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
            if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
            return e;
        }

        std::unordered_map<int, memory> exe_args = args;
        exe_args.erase(DNNL_ARG_SRC_1);
        exe_args.erase(DNNL_ARG_SRC_2);
        exe_args.erase(DNNL_ARG_DST_1);
        exe_args.erase(DNNL_ARG_DST_2);

        auto e0 = dnnl::sycl_interop::execute(prim_, stream, exe_args, deps);

        // calculate running_mean and running_variance
        auto batch_mean = args.find(DNNL_ARG_MEAN)->second;
        auto batch_variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto old_running_mean = args.find(DNNL_ARG_SRC_1)->second;
        auto old_running_variance = args.find(DNNL_ARG_SRC_2)->second;
        auto new_running_mean = args.find(DNNL_ARG_DST_1)->second;
        auto new_running_variance = args.find(DNNL_ARG_DST_2)->second;

        dnnl::engine p_engine = stream.get_engine();
        // new_running_mean = momentum * old_running_mean +
        //                                      (1 - momentum) * batch_mean
        auto sum_prim_0 = dnnl::sum({p_engine, scales_,
                {old_running_mean.get_desc(), batch_mean.get_desc()}});
        auto e1 = dnnl::sycl_interop::execute(sum_prim_0, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                        {DNNL_ARG_DST, new_running_mean}},
                {e0});
        // new_running_variance = momentum * old_running_variance +
        //                                  (1 - momentum) * batch_variance
        auto sum_prim_1 = dnnl::sum({p_engine, scales_,
                {old_running_variance.get_desc(), batch_variance.get_desc()}});
        auto e2 = dnnl::sycl_interop::execute(sum_prim_1, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                        {DNNL_ARG_DST, new_running_variance}},
                {e1});
        if (stream.get_engine().get_kind() == engine::kind::cpu) e2.wait();
        return e2;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        if (!is_training_) {
            auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
            return e;
        }

        std::unordered_map<int, memory> exe_args = args;
        exe_args.erase(DNNL_ARG_SRC_1);
        exe_args.erase(DNNL_ARG_SRC_2);
        exe_args.erase(DNNL_ARG_DST_1);
        exe_args.erase(DNNL_ARG_DST_2);

        auto e0 = dnnl::ocl_interop::execute(prim_, stream, exe_args, deps);

        // calculate running_mean and running_variance
        auto batch_mean = args.find(DNNL_ARG_MEAN)->second;
        auto batch_variance = args.find(DNNL_ARG_VARIANCE)->second;
        auto old_running_mean = args.find(DNNL_ARG_SRC_1)->second;
        auto old_running_variance = args.find(DNNL_ARG_SRC_2)->second;
        auto new_running_mean = args.find(DNNL_ARG_DST_1)->second;
        auto new_running_variance = args.find(DNNL_ARG_DST_2)->second;

        dnnl::engine p_engine = stream.get_engine();
        // new_running_mean = momentum * old_running_mean +
        //                                      (1 - momentum) * batch_mean
        auto sum_prim_0 = dnnl::sum({p_engine, scales_,
                {old_running_mean.get_desc(), batch_mean.get_desc()}});
        auto e1 = dnnl::ocl_interop::execute(sum_prim_0, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_mean},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_mean},
                        {DNNL_ARG_DST, new_running_mean}},
                {e0});
        // new_running_variance = momentum * old_running_variance +
        //                                  (1 - momentum) * batch_variance
        auto sum_prim_1 = dnnl::sum({p_engine, scales_,
                {old_running_variance.get_desc(), batch_variance.get_desc()}});
        auto e2 = dnnl::ocl_interop::execute(sum_prim_1, stream,
                {{DNNL_ARG_MULTIPLE_SRC, old_running_variance},
                        {DNNL_ARG_MULTIPLE_SRC + 1, batch_variance},
                        {DNNL_ARG_DST, new_running_variance}},
                {e1});
        return e2;
    }
#endif

private:
    dnnl::batch_normalization_forward prim_;
    bool is_training_ {false};
    std::vector<float> scales_;
};

struct batchnorm_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::batch_normalization_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    batchnorm_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::batch_normalization_backward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::batch_normalization_backward prim_;
};

struct resampling_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::resampling_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    resampling_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::resampling_forward(desc);
        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            auto it_src = args.find(DNNL_GRAPH_ARG_POST_SRC);
            auto it_dst = args.find(DNNL_ARG_DST);
            if (it_src == args.end() || it_dst == args.end()) {
                assert(!"cannot find src or dst memory");
                return;
            }

            const memory &psrc_mem = it_src->second;
            const memory &dst_mem = it_dst->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto ocl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::ocl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps);
                ocl_deps = {e};
            }
        }
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::resampling_forward prim_;
    bool with_sum_ {false};
};

struct resampling_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::resampling_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    resampling_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::resampling_backward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::resampling_backward prim_;
};

struct layernorm_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::layer_normalization_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    layernorm_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::layer_normalization_forward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::layer_normalization_forward prim_;
};

struct layernorm_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(
            dnnl::layer_normalization_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    layernorm_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::layer_normalization_backward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::layer_normalization_backward prim_;
};

struct sum_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::sum::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    sum_executable_t(std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
            fusion_info_mgr_t &mgr, pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::sum(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::sum prim_;
};

struct softmax_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::softmax_forward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    softmax_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::softmax_forward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::softmax_forward prim_;
};

struct softmax_bwd_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::softmax_backward::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    softmax_bwd_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::softmax_backward(desc);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto e = dnnl::sycl_interop::execute(prim_, stream, args, deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto e = dnnl::ocl_interop::execute(prim_, stream, args, deps);
        return e;
    }
#endif

private:
    dnnl::softmax_backward prim_;
};

struct reduction_executable_t : public op_executable_t {
    DECLARE_DESC_CLASS_AND_CREATOR(dnnl::reduction::primitive_desc);
    DECLARE_ARG_INDICES_GETTER;

    reduction_executable_t(std::shared_ptr<op_t> &op,
            const dnnl::engine &p_engine, fusion_info_mgr_t &mgr,
            pd_cache_t &pd_cache) {
        auto desc = create_desc(op, p_engine, mgr, pd_cache);
        prim_ = dnnl::reduction(desc);

        if (op->has_attr(op_attr::with_sum))
            with_sum_ = op->get_attr<bool>(op_attr::with_sum);
    }

    void execute(const stream &stream,
            const std::unordered_map<int, memory> &args) const override {
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                dnnl::reorder(psrc_mem, dst_mem)
                        .execute(stream, const_cast<memory &>(psrc_mem),
                                const_cast<memory &>(dst_mem));
            }
        }

        prim_.execute(stream, args);
    }

#ifdef DNNL_WITH_SYCL
    ::sycl::event execute_sycl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<::sycl::event> &deps = {}) const override {
        auto sycl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::sycl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        sycl_deps);
                sycl_deps = {e};
            }
        }

        auto e = dnnl::sycl_interop::execute(prim_, stream, args, sycl_deps);
        if (stream.get_engine().get_kind() == engine::kind::cpu) e.wait();
        return e;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    cl_event execute_ocl(const stream &stream,
            const std::unordered_map<int, memory> &args,
            const std::vector<cl_event> &deps = {}) const override {
        auto ocl_deps = deps;
        if (with_sum_) {
            const memory &psrc_mem = args.find(DNNL_GRAPH_ARG_POST_SRC)->second;
            const memory &dst_mem = args.find(DNNL_ARG_DST)->second;
            if (psrc_mem.get_data_handle() != dst_mem.get_data_handle()) {
                auto prim = dnnl::reorder(psrc_mem, dst_mem);
                auto e = dnnl::ocl_interop::execute(prim, stream,
                        {{DNNL_ARG_FROM, const_cast<memory &>(psrc_mem)},
                                {DNNL_ARG_TO, const_cast<memory &>(dst_mem)}},
                        ocl_deps);
                ocl_deps = {e};
            }
        }

        auto e = dnnl::ocl_interop::execute(prim_, stream, args, ocl_deps);
        return e;
    }
#endif

private:
    dnnl::reduction prim_;
    bool with_sum_ {false};
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
