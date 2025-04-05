namespace mlir {
namespace omp {
struct AlignedClauseOps {
  ::llvm::SmallVector<::mlir::Value> alignedVars;
  ::llvm::SmallVector<::mlir::Attribute> alignments;
};
struct AllocateClauseOps {
  ::llvm::SmallVector<::mlir::Value> allocateVars;
  ::llvm::SmallVector<::mlir::Value> allocatorVars;
};
struct CancelDirectiveNameClauseOps {
  ::mlir::omp::ClauseCancellationConstructTypeAttr cancelDirective;
};
struct CopyprivateClauseOps {
  ::llvm::SmallVector<::mlir::Value> copyprivateVars;
  ::llvm::SmallVector<::mlir::Attribute> copyprivateSyms;
};
struct CriticalNameClauseOps {
  ::mlir::StringAttr symName;
};
struct DependClauseOps {
  ::llvm::SmallVector<::mlir::Attribute> dependKinds;
  ::llvm::SmallVector<::mlir::Value> dependVars;
};
struct DeviceClauseOps {
  ::mlir::Value device;
};
struct DistScheduleClauseOps {
  ::mlir::UnitAttr distScheduleStatic;
  ::mlir::Value distScheduleChunkSize;
};
struct DoacrossClauseOps {
  ::mlir::omp::ClauseDependAttr doacrossDependType;
  ::mlir::IntegerAttr doacrossNumLoops;
  ::llvm::SmallVector<::mlir::Value> doacrossDependVars;
};
struct FilterClauseOps {
  ::mlir::Value filteredThreadId;
};
struct FinalClauseOps {
  ::mlir::Value final;
};
struct GrainsizeClauseOps {
  ::mlir::Value grainsize;
};
struct HasDeviceAddrClauseOps {
  ::llvm::SmallVector<::mlir::Value> hasDeviceAddrVars;
};
struct HintClauseOps {
  ::mlir::IntegerAttr hint;
};
struct IfClauseOps {
  ::mlir::Value ifExpr;
};
struct InReductionClauseOps {
  ::llvm::SmallVector<::mlir::Value> inReductionVars;
  ::llvm::SmallVector<bool> inReductionByref;
  ::llvm::SmallVector<::mlir::Attribute> inReductionSyms;
};
struct IsDevicePtrClauseOps {
  ::llvm::SmallVector<::mlir::Value> isDevicePtrVars;
};
struct LinearClauseOps {
  ::llvm::SmallVector<::mlir::Value> linearVars;
  ::llvm::SmallVector<::mlir::Value> linearStepVars;
};
struct LoopRelatedClauseOps {
  ::llvm::SmallVector<::mlir::Value> loopLowerBounds;
  ::llvm::SmallVector<::mlir::Value> loopUpperBounds;
  ::llvm::SmallVector<::mlir::Value> loopSteps;
  ::mlir::UnitAttr loopInclusive;
};
struct MapClauseOps {
  ::llvm::SmallVector<::mlir::Value> mapVars;
};
struct MemoryOrderClauseOps {
  ::mlir::omp::ClauseMemoryOrderKindAttr memoryOrder;
};
struct MergeableClauseOps {
  ::mlir::UnitAttr mergeable;
};
struct NogroupClauseOps {
  ::mlir::UnitAttr nogroup;
};
struct NontemporalClauseOps {
  ::llvm::SmallVector<::mlir::Value> nontemporalVars;
};
struct NowaitClauseOps {
  ::mlir::UnitAttr nowait;
};
struct NumTasksClauseOps {
  ::mlir::Value numTasks;
};
struct NumTeamsClauseOps {
  ::mlir::Value numTeamsLower;
  ::mlir::Value numTeamsUpper;
};
struct NumThreadsClauseOps {
  ::mlir::Value numThreads;
};
struct OrderClauseOps {
  ::mlir::omp::ClauseOrderKindAttr order;
  ::mlir::omp::OrderModifierAttr orderMod;
};
struct OrderedClauseOps {
  ::mlir::IntegerAttr ordered;
};
struct ParallelizationLevelClauseOps {
  ::mlir::UnitAttr parLevelSimd;
};
struct PriorityClauseOps {
  ::mlir::Value priority;
};
struct PrivateClauseOps {
  ::llvm::SmallVector<::mlir::Value> privateVars;
  ::llvm::SmallVector<::mlir::Attribute> privateSyms;
};
struct ProcBindClauseOps {
  ::mlir::omp::ClauseProcBindKindAttr procBindKind;
};
struct ReductionClauseOps {
  ::llvm::SmallVector<::mlir::Value> reductionVars;
  ::llvm::SmallVector<bool> reductionByref;
  ::llvm::SmallVector<::mlir::Attribute> reductionSyms;
};
struct SafelenClauseOps {
  ::mlir::IntegerAttr safelen;
};
struct ScheduleClauseOps {
  ::mlir::omp::ClauseScheduleKindAttr scheduleKind;
  ::mlir::Value scheduleChunk;
  ::mlir::omp::ScheduleModifierAttr scheduleMod;
  ::mlir::UnitAttr scheduleSimd;
};
struct SimdlenClauseOps {
  ::mlir::IntegerAttr simdlen;
};
struct TaskReductionClauseOps {
  ::llvm::SmallVector<::mlir::Value> taskReductionVars;
  ::llvm::SmallVector<bool> taskReductionByref;
  ::llvm::SmallVector<::mlir::Attribute> taskReductionSyms;
};
struct ThreadLimitClauseOps {
  ::mlir::Value threadLimit;
};
struct UntiedClauseOps {
  ::mlir::UnitAttr untied;
};
struct UseDeviceAddrClauseOps {
  ::llvm::SmallVector<::mlir::Value> useDeviceAddrVars;
};
struct UseDevicePtrClauseOps {
  ::llvm::SmallVector<::mlir::Value> useDevicePtrVars;
};

namespace detail {
template <typename... Mixins>
struct Clauses : public Mixins... {};
} // namespace detail

using AtomicCaptureOperands = detail::Clauses<HintClauseOps, MemoryOrderClauseOps>;

using AtomicReadOperands = detail::Clauses<HintClauseOps, MemoryOrderClauseOps>;

using AtomicUpdateOperands = detail::Clauses<HintClauseOps, MemoryOrderClauseOps>;

using AtomicWriteOperands = detail::Clauses<HintClauseOps, MemoryOrderClauseOps>;

using BarrierOperands = detail::Clauses<>;

using CancelOperands = detail::Clauses<CancelDirectiveNameClauseOps, IfClauseOps>;

using CancellationPointOperands = detail::Clauses<CancelDirectiveNameClauseOps>;

using CriticalDeclareOperands = detail::Clauses<CriticalNameClauseOps, HintClauseOps>;

using CriticalOperands = detail::Clauses<>;

using DeclareReductionOperands = detail::Clauses<>;

using DistributeOperands = detail::Clauses<AllocateClauseOps, DistScheduleClauseOps, OrderClauseOps, PrivateClauseOps>;

using FlushOperands = detail::Clauses<>;

using LoopNestOperands = detail::Clauses<LoopRelatedClauseOps>;

using MapBoundsOperands = detail::Clauses<>;

using MapInfoOperands = detail::Clauses<>;

using MaskedOperands = detail::Clauses<FilterClauseOps>;

using MasterOperands = detail::Clauses<>;

using OrderedOperands = detail::Clauses<DoacrossClauseOps>;

using OrderedRegionOperands = detail::Clauses<ParallelizationLevelClauseOps>;

using ParallelOperands = detail::Clauses<AllocateClauseOps, IfClauseOps, NumThreadsClauseOps, PrivateClauseOps, ProcBindClauseOps, ReductionClauseOps>;

using PrivateClauseOperands = detail::Clauses<>;

using SectionOperands = detail::Clauses<>;

using SectionsOperands = detail::Clauses<AllocateClauseOps, NowaitClauseOps, PrivateClauseOps, ReductionClauseOps>;

using SimdOperands = detail::Clauses<AlignedClauseOps, IfClauseOps, LinearClauseOps, NontemporalClauseOps, OrderClauseOps, PrivateClauseOps, ReductionClauseOps, SafelenClauseOps, SimdlenClauseOps>;

using SingleOperands = detail::Clauses<AllocateClauseOps, CopyprivateClauseOps, NowaitClauseOps, PrivateClauseOps>;

using TargetDataOperands = detail::Clauses<DeviceClauseOps, IfClauseOps, MapClauseOps, UseDeviceAddrClauseOps, UseDevicePtrClauseOps>;

using TargetEnterDataOperands = detail::Clauses<DependClauseOps, DeviceClauseOps, IfClauseOps, MapClauseOps, NowaitClauseOps>;

using TargetExitDataOperands = detail::Clauses<DependClauseOps, DeviceClauseOps, IfClauseOps, MapClauseOps, NowaitClauseOps>;

using TargetOperands = detail::Clauses<AllocateClauseOps, DependClauseOps, DeviceClauseOps, HasDeviceAddrClauseOps, IfClauseOps, InReductionClauseOps, IsDevicePtrClauseOps, MapClauseOps, NowaitClauseOps, PrivateClauseOps, ThreadLimitClauseOps>;

using TargetUpdateOperands = detail::Clauses<DependClauseOps, DeviceClauseOps, IfClauseOps, MapClauseOps, NowaitClauseOps>;

using TaskOperands = detail::Clauses<AllocateClauseOps, DependClauseOps, FinalClauseOps, IfClauseOps, InReductionClauseOps, MergeableClauseOps, PriorityClauseOps, PrivateClauseOps, UntiedClauseOps>;

using TaskgroupOperands = detail::Clauses<AllocateClauseOps, TaskReductionClauseOps>;

using TaskloopOperands = detail::Clauses<AllocateClauseOps, FinalClauseOps, GrainsizeClauseOps, IfClauseOps, InReductionClauseOps, MergeableClauseOps, NogroupClauseOps, NumTasksClauseOps, PriorityClauseOps, PrivateClauseOps, ReductionClauseOps, UntiedClauseOps>;

using TaskwaitOperands = detail::Clauses<DependClauseOps, NowaitClauseOps>;

using TaskyieldOperands = detail::Clauses<>;

using TeamsOperands = detail::Clauses<AllocateClauseOps, IfClauseOps, NumTeamsClauseOps, PrivateClauseOps, ReductionClauseOps, ThreadLimitClauseOps>;

using TerminatorOperands = detail::Clauses<>;

using ThreadprivateOperands = detail::Clauses<>;

using WsloopOperands = detail::Clauses<AllocateClauseOps, LinearClauseOps, NowaitClauseOps, OrderClauseOps, OrderedClauseOps, PrivateClauseOps, ReductionClauseOps, ScheduleClauseOps>;

using YieldOperands = detail::Clauses<>;
} // namespace omp
} // namespace mlir
