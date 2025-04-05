/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_REF_CONVOLUTION_UTILS_HPP
#define CPU_REF_CONVOLUTION_UTILS_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace ref_conv_utils {
inline dim_t get_data_off(const memory_desc_wrapper &mdw, int ndims, dim_t mb,
        dim_t c, dim_t id, dim_t ih, dim_t iw) {
    switch (ndims) {
        case 5: return mdw.off(mb, c, id, ih, iw);
        case 4: return mdw.off(mb, c, ih, iw);
        case 3: return mdw.off(mb, c, iw);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }
}

inline dim_t get_weights_off(const memory_desc_wrapper &mdw, bool with_groups,
        int ndims, dim_t g, dim_t oc, dim_t ic, dim_t kd, dim_t kh, dim_t kw) {
    switch (ndims) {
        case 5:
            return with_groups ? mdw.off(g, oc, ic, kd, kh, kw)
                               : mdw.off(oc, ic, kd, kh, kw);
        case 4:
            return with_groups ? mdw.off(g, oc, ic, kh, kw)
                               : mdw.off(oc, ic, kh, kw);
        case 3:
            return with_groups ? mdw.off(g, oc, ic, kw) : mdw.off(oc, ic, kw);
        default: assert(!"unsupported ndims"); return dim_t(0);
    }
}
} // namespace ref_conv_utils

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
