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

#ifndef GRAPH_BACKEND_FAKE_TRANSFORMATION_PASS_HPP
#define GRAPH_BACKEND_FAKE_TRANSFORMATION_PASS_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph/utils/pm/nested_matcher.hpp"
#include "graph/utils/pm/pass_base.hpp"
#include "graph/utils/pm/pbuilder.hpp"

#include "graph/backend/fake/pattern_utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace fake_impl {
namespace pass {

class transformation_pass_t : public graph::pass::pass_base {
public:
    explicit transformation_pass_t(std::string pbackend, std::string pname)
        : graph::pass::pass_base(std::move(pbackend), std::move(pname)) {}

    static graph::pass::pass_base_ptr create(
            std::string pbackend, std::string pname) {
        return std::make_shared<transformation_pass_t>(
                std::move(pbackend), std::move(pname));
    }

    // the criteria of pass execution
    impl::status_t run(graph_t &agraph) override {
        pattern_utils_t pu;
        graph::pass::Pattern pgraph
                = get_attr<graph::pass::Pattern>("Pattern")[0];

        // for each pattern. match it
        std::vector<op_t *> matched_op_list;
        if (get_verbose(verbose_t::create_dispatch, component_t::graph)) {
            printf("onednn_verbose,graph,create:dispatch,pattern_"
                   "matcher,%s,fake_backend\n",
                    get_pass_name().c_str());
        }
        pu.match(agraph, pgraph, matched_op_list);
        if (!matched_op_list.empty()) {
            // temporary solution here for showing which pattern matched
            if (getenv_int_user("GRAPH_DUMP", 0) > 0
                    || utils::check_verbose_string_user(
                            "GRAPH_DUMP", "pattern")) {
                printf("onednn_graph_verbose,info,pattern,hit,%s\n",
                        get_pass_name().c_str());
                fflush(stdout);
            }

            // Only fuse not rewrite. Will remove the fuse once dnnl
            // backend support subgraph mode
            pu.fuse(agraph, matched_op_list);
        }
        return impl::status::success;
    }
};

#define DECLARE_PASS_EX(bname, pname, counter) \
    static auto _registered_pass_##pname##_##bname##_##counter##_

#define DECLARE_PASS(bname, pname, counter) \
    DECLARE_PASS_EX(bname, pname, counter)

#define FAKE_BACKEND_REGISTER_TRANSFORMATION_PASS( \
        backend_name, pass_class_name) \
    DECLARE_PASS(backend_name, pass_class_name, __COUNTER__) \
            = registry.register_pass(#backend_name, #pass_class_name, \
                    &transformation_pass_t::create)

#define FAKE_BACKEND_REGISTER_PASSES_DEF_BEGIN(passes_class_) \
    inline void register_##passes_class_( \
            graph::pass::pass_registry_t &registry) {
#define FAKE_BACKEND_REGISTER_PASSES_DEF_END }

#define FAKE_BACKEND_REGISTER_PASSES_CALL(passes_class_, pass_registry_) \
    pass::register_##passes_class_(pass_registry_);

} // namespace pass
} // namespace fake_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
