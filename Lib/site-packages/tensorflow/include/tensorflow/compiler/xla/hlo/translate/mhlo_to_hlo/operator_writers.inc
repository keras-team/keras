/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* MLIR XLA Builders                                                          *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

mlir::LogicalResult ExportXlaOp(mlir::mhlo::AbsOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Abs(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::AddOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Add(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::AfterAllOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  std::vector<xla::XlaOp> xla_arg_0;
  for (auto operand : op.getODSOperands(0)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_0.push_back(result);
  }
  auto xla_result = xla::AfterAll(ctx.builder, Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::AndOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::And(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::Atan2Op op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Atan2(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::BatchNormInferenceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_2;
  if (failed(GetXlaOp(*op.getODSOperands(2).begin(), value_map, &xla_arg_2, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_3;
  if (failed(GetXlaOp(*op.getODSOperands(3).begin(), value_map, &xla_arg_3, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_4;
  if (failed(GetXlaOp(*op.getODSOperands(4).begin(), value_map, &xla_arg_4, op)))
    return mlir::failure();
  auto xla_arg_5 = ConvertAPFloat(op.getEpsilon());
  auto xla_arg_6 = Convertuint64_t(op.getFeatureIndex());
  auto xla_result = xla::BatchNormInference(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4), Unwrap(xla_arg_5), Unwrap(xla_arg_6));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::BroadcastOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_broadcast_sizes(op.getBroadcastSizes());
  auto xla_result = xla::Broadcast(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CbrtOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Cbrt(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CeilOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Ceil(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CholeskyOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertbool(op.getLower());
  auto xla_result = xla::Cholesky(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ClampOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_2;
  if (failed(GetXlaOp(*op.getODSOperands(2).begin(), value_map, &xla_arg_2, op)))
    return mlir::failure();
  auto xla_result = xla::Clamp(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ClzOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Clz(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CollectivePermuteOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_source_target_pairs(op.getSourceTargetPairs());
  auto xla_arg_2 = Convert_channel_handle(op.getChannelHandle());
  auto xla_result = xla::CollectivePermute(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ComplexOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Complex(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ConcatenateOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  std::vector<xla::XlaOp> xla_arg_0;
  for (auto operand : op.getODSOperands(0)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_0.push_back(result);
  }
  auto xla_arg_1 = Convertuint64_t(op.getDimension());
  auto xla_result = xla::ConcatInDim(ctx.builder, Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CreateTokenOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  auto xla_result = xla::CreateToken(ctx.builder);
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CrossReplicaSumOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_replica_groups(op.getReplicaGroups());
  auto xla_result = xla::CrossReplicaSum(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::DivOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Div(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::DynamicSliceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  std::vector<xla::XlaOp> xla_arg_1;
  for (auto operand : op.getODSOperands(1)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_1.push_back(result);
  }
  auto xla_arg_2 = Convert_slice_sizes(op.getSliceSizes());
  auto xla_result = xla::DynamicSlice(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::DynamicUpdateSliceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  std::vector<xla::XlaOp> xla_arg_2;
  for (auto operand : op.getODSOperands(2)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_2.push_back(result);
  }
  auto xla_result = xla::DynamicUpdateSlice(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::EinsumOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = ConvertStringRef(op.getEinsumConfig());
  auto xla_result = xla::Einsum(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ErfOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Erf(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ExpOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Exp(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::Expm1Op op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Expm1(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::FftOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_fft_type(op.getFftType());
  auto xla_arg_2 = Convert_fft_length(op.getFftLength());
  auto xla_result = xla::Fft(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::FloorOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Floor(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::GatherOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convert_dimension_numbers(op.getDimensionNumbers());
  auto xla_arg_3 = Convert_slice_sizes(op.getSliceSizes());
  auto xla_arg_4 = Convertbool(op.getIndicesAreSorted());
  auto xla_result = xla::Gather(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::GetDimensionSizeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertuint64_t(op.getDimension());
  auto xla_result = xla::GetDimensionSize(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::GetTupleElementOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertuint32_t(op.getIndex());
  auto xla_result = xla::GetTupleElement(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ImagOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Imag(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::IsFiniteOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::IsFinite(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::Log1pOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Log1p(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::LogOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Log(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::LogisticOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Logistic(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::MaxOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Max(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::MinOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Min(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::MulOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Mul(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::NegOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Neg(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::NotOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Not(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::OrOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Or(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::PopulationCountOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::PopulationCount(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::PowOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Pow(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RealOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Real(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ReducePrecisionOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convertuint32_t(op.getExponentBits());
  auto xla_arg_2 = Convertuint32_t(op.getMantissaBits());
  auto xla_result = xla::ReducePrecision(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RemOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Rem(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ReplicaIdOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  auto xla_result = xla::ReplicaId(ctx.builder);
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ReverseOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_dimensions(op.getDimensions());
  auto xla_result = xla::Rev(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RoundNearestEvenOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::RoundNearestEven(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RoundOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Round(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::RsqrtOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Rsqrt(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SelectOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_2;
  if (failed(GetXlaOp(*op.getODSOperands(2).begin(), value_map, &xla_arg_2, op)))
    return mlir::failure();
  auto xla_result = xla::Select(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ShiftLeftOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::ShiftLeft(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ShiftRightArithmeticOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::ShiftRightArithmetic(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::ShiftRightLogicalOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::ShiftRightLogical(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SignOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Sign(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SliceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_start_indices(op.getStartIndices());
  auto xla_arg_2 = Convert_limit_indices(op.getLimitIndices());
  auto xla_arg_3 = Convert_strides(op.getStrides());
  auto xla_result = xla::Slice(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SqrtOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Sqrt(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TanhOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Tanh(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TorchIndexSelectOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convertuint64_t(op.getDim());
  auto xla_arg_3 = Convertuint64_t(op.getBatchDims());
  auto xla_result = xla::TorchIndexSelect(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TransposeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_arg_1 = Convert_permutation(op.getPermutation());
  auto xla_result = xla::Transpose(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TriangularSolveOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_arg_2 = Convertbool(op.getLeftSide());
  auto xla_arg_3 = Convertbool(op.getLower());
  auto xla_arg_4 = Convertbool(op.getUnitDiagonal());
  auto xla_arg_5 = Convert_transpose_a(op.getTransposeA());
  auto xla_result = xla::TriangularSolve(Unwrap(xla_arg_0), Unwrap(xla_arg_1), Unwrap(xla_arg_2), Unwrap(xla_arg_3), Unwrap(xla_arg_4), Unwrap(xla_arg_5));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::TupleOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  std::vector<xla::XlaOp> xla_arg_0;
  for (auto operand : op.getODSOperands(0)) {
    xla::XlaOp result;
    if (failed(GetXlaOp(operand, value_map, &result, op)))
      return mlir::failure();
    xla_arg_0.push_back(result);
  }
  auto xla_result = xla::Tuple(ctx.builder, Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOp(mlir::mhlo::XorOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  xla::XlaOp xla_arg_1;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &xla_arg_1, op)))
    return mlir::failure();
  auto xla_result = xla::Xor(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
  value_map[result] = xla_result;
  return mlir::success();
}
mlir::LogicalResult ExportXlaOperator(
mlir::Operation* op, OpLoweringContext lowering_context) {

  xla::XlaScopedShardingAssignment sharding(lowering_context.builder, CreateOpShardingFromAttribute(op));

  xla::XlaScopedFrontendAttributesAssignment frontend_attributes(lowering_context.builder, CreateXlaFrontendAttributesFromOp(op));

  xla::XlaScopedOpMetadataAssignment op_metadata(lowering_context.builder, mlir::mhlo::CreateOpMetadataFromLocation(op, lowering_context.frame_index_builder));

  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AbsOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AddDependencyOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AddOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AfterAllOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AllGatherOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AllReduceOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AllToAllOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AndOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AsyncDoneOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AsyncStartOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::AsyncUpdateOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::Atan2Op>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BatchNormGradOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BatchNormInferenceOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BatchNormTrainingOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BitcastConvertOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BitcastOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BroadcastInDimOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::BroadcastOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CaseOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CbrtOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CeilOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CholeskyOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ClampOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ClzOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CollectiveBroadcastOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CollectivePermuteOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CompareOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ComplexOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CompositeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ConcatenateOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ConstantOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ConvertOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ConvolutionOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CopyOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CosineOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CreateTokenOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CrossReplicaSumOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::CustomCallOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DivOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DomainOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DotGeneralOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DotOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicBroadcastInDimOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicConvOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicGatherOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicIotaOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicPadOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicReshapeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicSliceOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::DynamicUpdateSliceOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::EinsumOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ErfOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ExpOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::Expm1Op>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::FftOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::FloorOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::FusionOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::GatherOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::GetDimensionSizeOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::GetTupleElementOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::IfOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ImagOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::InfeedOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::IotaOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::IsFiniteOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::Log1pOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::LogOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::LogisticOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MapOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MaxOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MinOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MinimumBroadcastShapesOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::MulOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::NegOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::NotOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::OptimizationBarrierOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::OrOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::OutfeedOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::PadOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::PartitionIdOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::PopulationCountOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::PowOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RealDynamicSliceOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RealOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RecvOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReduceOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReducePrecisionOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReduceScatterOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReduceWindowOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RemOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReplicaIdOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReshapeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReturnOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ReverseOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RngBitGeneratorOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RngOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RoundNearestEvenOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RoundOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::RsqrtOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ScatterOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SelectAndScatterOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SelectOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SendOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SetDimensionSizeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ShiftLeftOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ShiftRightArithmeticOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::ShiftRightLogicalOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SignOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SineOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SliceOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SortOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SparseDotOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SqrtOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::StochasticConvertOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::SubtractOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TanOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TanhOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TopKOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TorchIndexSelectOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TraceOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TransposeOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TriangularSolveOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::TupleOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::UniformDequantizeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::UniformQuantizeOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::WhileOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::XlaRngGetAndUpdateStateOp>(op)) {
    return mlir::mhlo::ExportXlaOp(xla_op, lowering_context);
  }
  if (auto xla_op = llvm::dyn_cast<mlir::mhlo::XorOp>(op)) {
    return ExportXlaOp(xla_op, lowering_context);
  }
  return mlir::failure();
}
