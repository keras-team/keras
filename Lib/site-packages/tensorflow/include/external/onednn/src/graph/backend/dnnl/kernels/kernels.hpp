/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_KERNELS_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_KERNELS_HPP

#include "graph/backend/dnnl/kernels/batchnorm.hpp"
#include "graph/backend/dnnl/kernels/binary.hpp"
#include "graph/backend/dnnl/kernels/concat.hpp"
#include "graph/backend/dnnl/kernels/conv.hpp"
#include "graph/backend/dnnl/kernels/convtranspose.hpp"
#include "graph/backend/dnnl/kernels/dummy.hpp"
#include "graph/backend/dnnl/kernels/eltwise.hpp"
#include "graph/backend/dnnl/kernels/large_partition.hpp"
#include "graph/backend/dnnl/kernels/layernorm.hpp"
#include "graph/backend/dnnl/kernels/logsoftmax.hpp"
#include "graph/backend/dnnl/kernels/matmul.hpp"
#include "graph/backend/dnnl/kernels/mqa.hpp"
#include "graph/backend/dnnl/kernels/mqa_base.hpp"
#include "graph/backend/dnnl/kernels/pool.hpp"
#include "graph/backend/dnnl/kernels/prelu.hpp"
#include "graph/backend/dnnl/kernels/quantize.hpp"
#include "graph/backend/dnnl/kernels/reduction.hpp"
#include "graph/backend/dnnl/kernels/reorder.hpp"
#include "graph/backend/dnnl/kernels/resampling.hpp"
#include "graph/backend/dnnl/kernels/sdp.hpp"
#include "graph/backend/dnnl/kernels/sdp_base.hpp"
#include "graph/backend/dnnl/kernels/select.hpp"
#include "graph/backend/dnnl/kernels/shuffle.hpp"
#include "graph/backend/dnnl/kernels/softmax.hpp"
#include "graph/backend/dnnl/kernels/sum.hpp"

#endif
