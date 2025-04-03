/* Copyright 2024 The Shardy Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SHARDY_DIALECT_SDY_IR_PARSERS_H_
#define SHARDY_DIALECT_SDY_IR_PARSERS_H_

#include <cstdint>

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "shardy/dialect/sdy/ir/dialect.h"

namespace mlir {
namespace sdy {

// Parses the factor sizes of an OpShardingRule. The keys in the list are the
// indices in the factor mapping, with i=0, j=1, k=2,... z=17. For any index
// greater than 17 it should be parsed as z_X where `X>0` and the index
// being X-17 (for example `z_1`, `z_123`, etc.). For example a list of
// `{6, 2, 4}` is printed as `{i=6, j=2, k=4}`.
ParseResult parseFactorSizes(AsmParser& parser,
                             SmallVector<int64_t>& factorSizes);

ParseResult parseIsCustomRule(AsmParser& parser, bool& isCustomRule);

// Parses a single block region without the block id. This is an example of what
// it parses:
//
// (%blockArg1, ..., %blockArgM) {
//   // ops in the block
// }
//
// This is needed for using `custom<SingleBlockRegionNoBlockId>` in
// `assemblyFormat`.
ParseResult parseSingleBlockRegionNoBlockId(OpAsmParser& parser,
                                            Region& region);

// Parses the TensorShardingPerValueAttr without the outside <>.
//
// The default assemblyFormat of TensorShardingPerValueAttr would have us
// print it as:
//
// <[<@mesh, ...>, ..., <@mesh, ...>]>
//
// In some ops we want to avoid the extra <> so we have a custom parser/printer
// for it. So we get the following instead:
//
// [<@mesh, ...>, ..., <@mesh, ...>]
//
// This is needed for using `custom<StrippedTensorShardingPerValueAttr>` in
// `assemblyFormat`.
ParseResult parseStrippedTensorShardingPerValueAttr(
    AsmParser& parser, TensorShardingPerValueAttr& shardingPerValue);

}  // namespace sdy
}  // namespace mlir

#endif  // SHARDY_DIALECT_SDY_IR_PARSERS_H_
