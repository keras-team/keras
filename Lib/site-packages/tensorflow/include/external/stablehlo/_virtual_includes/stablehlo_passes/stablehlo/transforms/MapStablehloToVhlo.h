/* Copyright 2022 The StableHLO Authors.
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

#ifndef STABLEHLO_TRANSFORMS_MAPSTABLEHLOTOVHLO_H
#define STABLEHLO_TRANSFORMS_MAPSTABLEHLOTOVHLO_H

#include <type_traits>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/VhloOps.h"

namespace mlir {
namespace stablehlo {

template <typename VhloOpTy>
struct VhloToStablehloOpImpl {
  using Type = std::false_type;
};
template <typename VhloOpTy>
using VhloToStablehloOp = typename VhloToStablehloOpImpl<VhloOpTy>::Type;

template <typename StablehloOpTy>
struct StablehloToVhloOpImpl {
  using Type = std::false_type;
};
template <typename StablehloOpTy>
using StablehloToVhloOp = typename StablehloToVhloOpImpl<StablehloOpTy>::Type;

#define MAP_STABLEHLO_TO_VHLO(OpName, OpVer)          \
  template <>                                         \
  struct StablehloToVhloOpImpl<stablehlo::OpName> {   \
    using Type = vhlo::OpName##OpVer;                 \
  };                                                  \
  template <>                                         \
  struct VhloToStablehloOpImpl<vhlo::OpName##OpVer> { \
    using Type = stablehlo::OpName;                   \
  };

MAP_STABLEHLO_TO_VHLO(AbsOp, V1)
MAP_STABLEHLO_TO_VHLO(AddOp, V1)
MAP_STABLEHLO_TO_VHLO(AfterAllOp, V1)
MAP_STABLEHLO_TO_VHLO(AllGatherOp, V2)
MAP_STABLEHLO_TO_VHLO(AllReduceOp, V2)
MAP_STABLEHLO_TO_VHLO(AllToAllOp, V2)
MAP_STABLEHLO_TO_VHLO(AndOp, V1)
MAP_STABLEHLO_TO_VHLO(Atan2Op, V1)
MAP_STABLEHLO_TO_VHLO(BatchNormGradOp, V1)
MAP_STABLEHLO_TO_VHLO(BatchNormInferenceOp, V1)
MAP_STABLEHLO_TO_VHLO(BatchNormTrainingOp, V1)
MAP_STABLEHLO_TO_VHLO(BitcastConvertOp, V1)
MAP_STABLEHLO_TO_VHLO(BroadcastInDimOp, V1)
MAP_STABLEHLO_TO_VHLO(BroadcastOp, V1)
MAP_STABLEHLO_TO_VHLO(CaseOp, V1)
MAP_STABLEHLO_TO_VHLO(CbrtOp, V1)
MAP_STABLEHLO_TO_VHLO(CeilOp, V1)
MAP_STABLEHLO_TO_VHLO(CholeskyOp, V1)
MAP_STABLEHLO_TO_VHLO(ClampOp, V1)
MAP_STABLEHLO_TO_VHLO(ClzOp, V1)
MAP_STABLEHLO_TO_VHLO(CollectiveBroadcastOp, V1)
MAP_STABLEHLO_TO_VHLO(CollectivePermuteOp, V1)
MAP_STABLEHLO_TO_VHLO(CompareOp, V1)
MAP_STABLEHLO_TO_VHLO(ComplexOp, V1)
MAP_STABLEHLO_TO_VHLO(CompositeOp, V1)
MAP_STABLEHLO_TO_VHLO(ConcatenateOp, V1)
MAP_STABLEHLO_TO_VHLO(ConstantOp, V1)
MAP_STABLEHLO_TO_VHLO(ConvertOp, V1)
MAP_STABLEHLO_TO_VHLO(ConvolutionOp, V1)
MAP_STABLEHLO_TO_VHLO(CosineOp, V1)
MAP_STABLEHLO_TO_VHLO(CreateTokenOp, V1)
MAP_STABLEHLO_TO_VHLO(CrossReplicaSumOp, V1)
MAP_STABLEHLO_TO_VHLO(CustomCallOp, V1)
MAP_STABLEHLO_TO_VHLO(DivOp, V1)
MAP_STABLEHLO_TO_VHLO(DotGeneralOp, V2)
MAP_STABLEHLO_TO_VHLO(DotOp, V1)
MAP_STABLEHLO_TO_VHLO(DynamicBroadcastInDimOp, V1)
MAP_STABLEHLO_TO_VHLO(DynamicConvOp, V2)
MAP_STABLEHLO_TO_VHLO(DynamicGatherOp, V2)
MAP_STABLEHLO_TO_VHLO(DynamicIotaOp, V1)
MAP_STABLEHLO_TO_VHLO(DynamicPadOp, V1)
MAP_STABLEHLO_TO_VHLO(DynamicReshapeOp, V1)
MAP_STABLEHLO_TO_VHLO(DynamicSliceOp, V1)
MAP_STABLEHLO_TO_VHLO(DynamicUpdateSliceOp, V1)
MAP_STABLEHLO_TO_VHLO(EinsumOp, V1)
MAP_STABLEHLO_TO_VHLO(Expm1Op, V1)
MAP_STABLEHLO_TO_VHLO(ExpOp, V1)
MAP_STABLEHLO_TO_VHLO(FftOp, V1)
MAP_STABLEHLO_TO_VHLO(FloorOp, V1)
MAP_STABLEHLO_TO_VHLO(GatherOp, V2)
MAP_STABLEHLO_TO_VHLO(GetDimensionSizeOp, V1)
MAP_STABLEHLO_TO_VHLO(GetTupleElementOp, V1)
MAP_STABLEHLO_TO_VHLO(IfOp, V1)
MAP_STABLEHLO_TO_VHLO(ImagOp, V1)
MAP_STABLEHLO_TO_VHLO(InfeedOp, V1)
MAP_STABLEHLO_TO_VHLO(IotaOp, V1)
MAP_STABLEHLO_TO_VHLO(IsFiniteOp, V1)
MAP_STABLEHLO_TO_VHLO(Log1pOp, V1)
MAP_STABLEHLO_TO_VHLO(LogisticOp, V1)
MAP_STABLEHLO_TO_VHLO(LogOp, V1)
MAP_STABLEHLO_TO_VHLO(MapOp, V1)
MAP_STABLEHLO_TO_VHLO(MaxOp, V1)
MAP_STABLEHLO_TO_VHLO(MinOp, V1)
MAP_STABLEHLO_TO_VHLO(MulOp, V1)
MAP_STABLEHLO_TO_VHLO(NegOp, V1)
MAP_STABLEHLO_TO_VHLO(NotOp, V1)
MAP_STABLEHLO_TO_VHLO(OptimizationBarrierOp, V1)
MAP_STABLEHLO_TO_VHLO(OrOp, V1)
MAP_STABLEHLO_TO_VHLO(OutfeedOp, V1)
MAP_STABLEHLO_TO_VHLO(PadOp, V1)
MAP_STABLEHLO_TO_VHLO(PartitionIdOp, V1)
MAP_STABLEHLO_TO_VHLO(PopulationCountOp, V1)
MAP_STABLEHLO_TO_VHLO(PowOp, V1)
MAP_STABLEHLO_TO_VHLO(RealDynamicSliceOp, V1)
MAP_STABLEHLO_TO_VHLO(RealOp, V1)
MAP_STABLEHLO_TO_VHLO(RecvOp, V1)
MAP_STABLEHLO_TO_VHLO(ReduceOp, V1)
MAP_STABLEHLO_TO_VHLO(ReducePrecisionOp, V1)
MAP_STABLEHLO_TO_VHLO(ReduceScatterOp, V1)
MAP_STABLEHLO_TO_VHLO(ReduceWindowOp, V1)
MAP_STABLEHLO_TO_VHLO(RemOp, V1)
MAP_STABLEHLO_TO_VHLO(ReplicaIdOp, V1)
MAP_STABLEHLO_TO_VHLO(ReshapeOp, V1)
MAP_STABLEHLO_TO_VHLO(ReturnOp, V1)
MAP_STABLEHLO_TO_VHLO(ReverseOp, V1)
MAP_STABLEHLO_TO_VHLO(RngBitGeneratorOp, V1)
MAP_STABLEHLO_TO_VHLO(RngOp, V1)
MAP_STABLEHLO_TO_VHLO(RoundOp, V1)
MAP_STABLEHLO_TO_VHLO(RoundNearestEvenOp, V1)
MAP_STABLEHLO_TO_VHLO(RsqrtOp, V1)
MAP_STABLEHLO_TO_VHLO(ScatterOp, V2)
MAP_STABLEHLO_TO_VHLO(SelectAndScatterOp, V1)
MAP_STABLEHLO_TO_VHLO(SelectOp, V1)
MAP_STABLEHLO_TO_VHLO(SendOp, V1)
MAP_STABLEHLO_TO_VHLO(SetDimensionSizeOp, V1)
MAP_STABLEHLO_TO_VHLO(ShiftLeftOp, V1)
MAP_STABLEHLO_TO_VHLO(ShiftRightArithmeticOp, V1)
MAP_STABLEHLO_TO_VHLO(ShiftRightLogicalOp, V1)
MAP_STABLEHLO_TO_VHLO(SignOp, V1)
MAP_STABLEHLO_TO_VHLO(SineOp, V1)
MAP_STABLEHLO_TO_VHLO(SliceOp, V1)
MAP_STABLEHLO_TO_VHLO(SortOp, V1)
MAP_STABLEHLO_TO_VHLO(SqrtOp, V1)
MAP_STABLEHLO_TO_VHLO(SubtractOp, V1)
MAP_STABLEHLO_TO_VHLO(TanOp, V1)
MAP_STABLEHLO_TO_VHLO(TanhOp, V1)
MAP_STABLEHLO_TO_VHLO(TorchIndexSelectOp, V1)
MAP_STABLEHLO_TO_VHLO(TransposeOp, V1)
MAP_STABLEHLO_TO_VHLO(TriangularSolveOp, V1)
MAP_STABLEHLO_TO_VHLO(TupleOp, V1)
MAP_STABLEHLO_TO_VHLO(UnaryEinsumOp, V1)
MAP_STABLEHLO_TO_VHLO(UniformDequantizeOp, V1)
MAP_STABLEHLO_TO_VHLO(UniformQuantizeOp, V1)
MAP_STABLEHLO_TO_VHLO(WhileOp, V1)
MAP_STABLEHLO_TO_VHLO(XorOp, V1)

#undef MAP_STABLEHLO_TO_VHLO
#undef MAP_STABLEHLO_TO_VHLO_V0

// Nonstandard mappings
#define MAP_UPSTREAM_TO_VHLO(UpstreamOpName, VhloOpName, OpVer) \
  template <>                                                   \
  struct StablehloToVhloOpImpl<UpstreamOpName> {                \
    using Type = VhloOpName##OpVer;                             \
  };                                                            \
  template <>                                                   \
  struct VhloToStablehloOpImpl<VhloOpName##OpVer> {             \
    using Type = UpstreamOpName;                                \
  };

MAP_UPSTREAM_TO_VHLO(func::FuncOp, vhlo::FuncOp, V1)
MAP_UPSTREAM_TO_VHLO(func::CallOp, vhlo::CallOp, V1)

// Slight ambiguity between stablehlo::ReturnOp and func::ReturnOp
// Only map in one direction for func.return --> vhlo.return
template <>
struct StablehloToVhloOpImpl<func::ReturnOp> {
  using Type = vhlo::ReturnOpV1;
};

#undef MAP_UPSTREAM_TO_VHLO

}  // namespace stablehlo
}  // namespace mlir

#endif  // STABLEHLO_TRANSFORMS_MAPSTABLEHLOTOVHLO_H
