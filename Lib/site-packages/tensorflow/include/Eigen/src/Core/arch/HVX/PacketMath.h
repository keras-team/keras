
#ifndef EIGEN_HVX_PACKET_MATH_H
#define EIGEN_HVX_PACKET_MATH_H

// Only support 128B HVX now.
// Floating-point operations are supported only since V68.
#if defined __HVX__ && (__HVX_LENGTH__ == 128) && __HVX_ARCH__ >= 68

// All the floating-point operations do not support IEEE standard.
// From HVX document:
//   There is no concept of infinity or NaN. QFloat saturates to maximum
//   exponent with maximum positive or minimum negative significand.

#ifndef EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS
#define EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS 32
#endif

namespace Eigen {
namespace internal {

// HVX utilities.

template <int D>
EIGEN_STRONG_INLINE HVX_Vector HVX_vmem(const void* m) {
  HVX_Vector v;
#if EIGEN_COMP_CLANG
  // Use inlined assembly for aligned vmem load on unaligned memory.
  // Use type cast to HVX_Vector* may mess up with compiler data alignment.
  __asm__("%0 = vmem(%1+#%2)" : "=v"(v) : "r"(m), "i"(D) : "memory");
#else
  void* aligned_mem =
      reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(m) & ~(__HVX_LENGTH__ - 1)) + D * __HVX_LENGTH__);
  memcpy(&v, aligned_mem, __HVX_LENGTH__);
#endif
  return v;
}

template <typename T>
EIGEN_STRONG_INLINE HVX_Vector HVX_load(const T* mem) {
  HVX_Vector v;
  memcpy(&v, reinterpret_cast<const HVX_Vector*>(mem), __HVX_LENGTH__);
  return v;
}

template <typename T>
EIGEN_STRONG_INLINE HVX_Vector HVX_loadu(const T* mem) {
  HVX_Vector v;
  memcpy(&v, mem, __HVX_LENGTH__);
  return v;
}

template <size_t Size, size_t Alignment, typename T>
EIGEN_STRONG_INLINE HVX_Vector HVX_load_partial(const T* mem) {
#if defined(EIGEN_HVX_FAST_PARTIAL_VECTOR_LOAD)
  // Fast partial vector load through aligned vmem load.
  // The load may past end of array but is aligned to prevent memory fault.
  HVX_Vector v0 = HVX_vmem<0>(mem);
  HVX_Vector v1 = v0;
  uintptr_t mem_addr = reinterpret_cast<uintptr_t>(mem);
  EIGEN_IF_CONSTEXPR(Size * sizeof(T) <= Alignment) {
    // Data size less than alignment will never cross multiple aligned vectors.
    v1 = v0;
  }
  else {
    uintptr_t left_off = mem_addr & (__HVX_LENGTH__ - 1);
    if (left_off + Size * sizeof(T) > __HVX_LENGTH__) {
      v1 = HVX_vmem<1>(mem);
    } else {
      v1 = v0;
    }
  }
  return Q6_V_valign_VVR(v1, v0, mem_addr);
#else
  HVX_Vector v;
  memcpy(&v, mem, Size * sizeof(T));
  return v;
#endif
}

template <typename T>
EIGEN_STRONG_INLINE void HVX_store(T* mem, HVX_Vector v) {
  memcpy(reinterpret_cast<HVX_Vector*>(mem), &v, __HVX_LENGTH__);
}

template <typename T>
EIGEN_STRONG_INLINE void HVX_storeu(T* mem, HVX_Vector v) {
  memcpy(mem, &v, __HVX_LENGTH__);
}

template <size_t Size, size_t Alignment, typename T>
EIGEN_STRONG_INLINE void HVX_store_partial(T* mem, HVX_Vector v) {
  uintptr_t mem_addr = reinterpret_cast<uintptr_t>(mem);
  HVX_Vector value = Q6_V_vlalign_VVR(v, v, mem_addr);
  uintptr_t left_off = mem_addr & (__HVX_LENGTH__ - 1);
  uintptr_t right_off = left_off + Size * sizeof(T);

  HVX_VectorPred ql_not = Q6_Q_vsetq_R(mem_addr);
  HVX_VectorPred qr = Q6_Q_vsetq2_R(right_off);

  EIGEN_IF_CONSTEXPR(Size * sizeof(T) > Alignment) {
    if (right_off > __HVX_LENGTH__) {
      Q6_vmem_QRIV(qr, mem + __HVX_LENGTH__ / sizeof(T), value);
      qr = Q6_Q_vcmp_eq_VbVb(value, value);
    }
  }

  ql_not = Q6_Q_or_QQn(ql_not, qr);
  Q6_vmem_QnRIV(ql_not, mem, value);
}

// Packet definitions.
enum class HVXPacketSize {
  Full,
  Half,
  Quarter,
};

// Hexagon compiler uses same HVX_Vector to represent all HVX vector types.
// Wrap different vector type (float32, int32, etc) to different class with
// explicit constructor and casting back-and-force to HVX_Vector.
template <HVXPacketSize T>
class HVXPacket {
 public:
  HVXPacket() = default;
  static HVXPacket Create(HVX_Vector v) { return HVXPacket(v); }
  HVX_Vector Get() const { return m_val; }

 private:
  explicit HVXPacket(HVX_Vector v) : m_val(v) {}
  HVX_Vector m_val = Q6_V_vzero();
};

typedef HVXPacket<HVXPacketSize::Full> Packet32f;
typedef HVXPacket<HVXPacketSize::Half> Packet16f;
typedef HVXPacket<HVXPacketSize::Quarter> Packet8f;

// Packet traits.
template <>
struct packet_traits<float> : default_packet_traits {
  typedef Packet32f type;
  typedef Packet16f half;
  enum {
    Vectorizable = 1,
    AlignedOnScalar = 1,
    size = 32,

    HasCmp = 1,
    HasAdd = 1,
    HasSub = 1,
    HasShift = 0,
    HasMul = 1,
    HasNegate = 1,
    HasAbs = 1,
    HasArg = 0,
    HasAbs2 = 0,
    HasAbsDiff = 0,
    HasMin = 1,
    HasMax = 1,
    HasConj = 0,
    HasSetLinear = 0,
    HasBlend = 0,

    HasDiv = 0,

    HasSin = 0,
    HasCos = 0,
    HasACos = 0,
    HasASin = 0,
    HasATan = 0,
    HasATanh = 0,
    HasLog = 0,
    HasExp = 0,
    HasSqrt = 0,
    HasRsqrt = 0,
    HasTanh = 0,
    HasErf = 0,
    HasBessel = 0,
    HasNdtri = 0
  };
};

template <>
struct unpacket_traits<Packet32f> {
  typedef float type;
  typedef Packet16f half;
  enum {
    size = 32,
    alignment = Aligned128,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<Packet16f> {
  typedef float type;
  typedef Packet8f half;
  enum {
    size = 16,
    // Many code assume alignment on packet size instead of following trait
    // So we do not use Aligned128 to optimize aligned load/store,
    alignment = Aligned64,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

template <>
struct unpacket_traits<Packet8f> {
  typedef float type;
  typedef Packet8f half;
  enum {
    size = 8,
    // Many code assume alignment on packet size instead of following trait
    // So we do not use Aligned128 to optimize aligned load/store,
    alignment = Aligned32,
    vectorizable = true,
    masked_load_available = false,
    masked_store_available = false
  };
};

// float32 operations.
template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pzero_hvx(const HVXPacket<T>&) {
  return HVXPacket<T>::Create(Q6_V_vzero());
}
template <>
EIGEN_STRONG_INLINE Packet32f pzero<Packet32f>(const Packet32f&) {
  return pzero_hvx(Packet32f());
}
template <>
EIGEN_STRONG_INLINE Packet16f pzero<Packet16f>(const Packet16f&) {
  return pzero_hvx(Packet16f());
}
template <>
EIGEN_STRONG_INLINE Packet8f pzero<Packet8f>(const Packet8f&) {
  return pzero_hvx(Packet8f());
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE typename unpacket_traits<HVXPacket<T>>::half predux_half_dowto4_hvx(const HVXPacket<T>& a) {
  const Index packet_size = unpacket_traits<HVXPacket<T>>::size;
  return unpacket_traits<HVXPacket<T>>::half::Create(
      Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(Q6_V_vror_VR(a.Get(), sizeof(float) * packet_size / 2), a.Get())));
}
template <>
EIGEN_STRONG_INLINE Packet16f predux_half_dowto4(const Packet32f& a) {
  return predux_half_dowto4_hvx(a);
}
template <>
EIGEN_STRONG_INLINE Packet8f predux_half_dowto4(const Packet16f& a) {
  return predux_half_dowto4_hvx(a);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pset1_hvx(const float& from) {
  union {
    float f;
    int32_t i;
  } u;
  u.f = from;
  return HVXPacket<T>::Create(Q6_V_vsplat_R(u.i));
}
template <>
EIGEN_STRONG_INLINE Packet32f pset1<Packet32f>(const float& from) {
  return pset1_hvx<HVXPacketSize::Full>(from);
}
template <>
EIGEN_STRONG_INLINE Packet16f pset1<Packet16f>(const float& from) {
  return pset1_hvx<HVXPacketSize::Half>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8f pset1<Packet8f>(const float& from) {
  return pset1_hvx<HVXPacketSize::Quarter>(from);
}

template <>
EIGEN_STRONG_INLINE Packet32f pload<Packet32f>(const float* from) {
  return Packet32f::Create(HVX_load(from));
}
template <>
EIGEN_STRONG_INLINE Packet16f pload<Packet16f>(const float* from) {
  return Packet16f::Create(
      HVX_load_partial<unpacket_traits<Packet16f>::size, unpacket_traits<Packet16f>::alignment>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8f pload<Packet8f>(const float* from) {
  return Packet8f::Create(
      HVX_load_partial<unpacket_traits<Packet8f>::size, unpacket_traits<Packet8f>::alignment>(from));
}

template <>
EIGEN_STRONG_INLINE Packet32f ploadu<Packet32f>(const float* from) {
  return Packet32f::Create(HVX_loadu(from));
}
template <>
EIGEN_STRONG_INLINE Packet16f ploadu<Packet16f>(const float* from) {
  return Packet16f::Create(HVX_load_partial<unpacket_traits<Packet16f>::size, 0>(from));
}
template <>
EIGEN_STRONG_INLINE Packet8f ploadu<Packet8f>(const float* from) {
  return Packet8f::Create(HVX_load_partial<unpacket_traits<Packet8f>::size, 0>(from));
}

template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet32f& from) {
  HVX_store(to, from.Get());
}
template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet16f& from) {
  HVX_store_partial<unpacket_traits<Packet16f>::size, unpacket_traits<Packet16f>::alignment>(to, from.Get());
}
template <>
EIGEN_STRONG_INLINE void pstore<float>(float* to, const Packet8f& from) {
  HVX_store_partial<unpacket_traits<Packet8f>::size, unpacket_traits<Packet8f>::alignment>(to, from.Get());
}

template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet32f& from) {
  HVX_storeu(to, from.Get());
}
template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet16f& from) {
  HVX_store_partial<unpacket_traits<Packet16f>::size, 0>(to, from.Get());
}
template <>
EIGEN_STRONG_INLINE void pstoreu<float>(float* to, const Packet8f& from) {
  HVX_store_partial<unpacket_traits<Packet8f>::size, 0>(to, from.Get());
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pmul_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  return HVXPacket<T>::Create(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a.Get(), b.Get())));
}
template <>
EIGEN_STRONG_INLINE Packet32f pmul<Packet32f>(const Packet32f& a, const Packet32f& b) {
  return pmul_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pmul<Packet16f>(const Packet16f& a, const Packet16f& b) {
  return pmul_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pmul<Packet8f>(const Packet8f& a, const Packet8f& b) {
  return pmul_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> padd_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  return HVXPacket<T>::Create(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a.Get(), b.Get())));
}
template <>
EIGEN_STRONG_INLINE Packet32f padd<Packet32f>(const Packet32f& a, const Packet32f& b) {
  return padd_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f padd<Packet16f>(const Packet16f& a, const Packet16f& b) {
  return padd_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f padd<Packet8f>(const Packet8f& a, const Packet8f& b) {
  return padd_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> psub_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  return HVXPacket<T>::Create(Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a.Get(), b.Get())));
}
template <>
EIGEN_STRONG_INLINE Packet32f psub<Packet32f>(const Packet32f& a, const Packet32f& b) {
  return psub_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f psub<Packet16f>(const Packet16f& a, const Packet16f& b) {
  return psub_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f psub<Packet8f>(const Packet8f& a, const Packet8f& b) {
  return psub_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pnegate_hvx(const HVXPacket<T>& a) {
  return HVXPacket<T>::Create(a.Get() ^ Q6_V_vsplat_R(0x80000000));
}
template <>
EIGEN_STRONG_INLINE Packet32f pnegate(const Packet32f& a) {
  return pnegate_hvx(a);
}
template <>
EIGEN_STRONG_INLINE Packet16f pnegate(const Packet16f& a) {
  return pnegate_hvx(a);
}
template <>
EIGEN_STRONG_INLINE Packet8f pnegate(const Packet8f& a) {
  return pnegate_hvx(a);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pcmp_le_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  HVX_Vector v_true = Q6_Vb_vsplat_R(0xff);
  HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(a.Get(), b.Get());
  return HVXPacket<T>::Create(Q6_V_vmux_QVV(pred, Q6_V_vzero(), v_true));
}
template <>
EIGEN_STRONG_INLINE Packet32f pcmp_le(const Packet32f& a, const Packet32f& b) {
  return pcmp_le_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pcmp_le(const Packet16f& a, const Packet16f& b) {
  return pcmp_le_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pcmp_le(const Packet8f& a, const Packet8f& b) {
  return pcmp_le_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pcmp_eq_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  HVX_Vector v_true = Q6_Vb_vsplat_R(0xff);
  HVX_VectorPred pred = Q6_Q_vcmp_eq_VwVw(a.Get(), b.Get());
  return HVXPacket<T>::Create(Q6_V_vmux_QVV(pred, v_true, Q6_V_vzero()));
}
template <>
EIGEN_STRONG_INLINE Packet32f pcmp_eq(const Packet32f& a, const Packet32f& b) {
  return pcmp_eq_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pcmp_eq(const Packet16f& a, const Packet16f& b) {
  return pcmp_eq_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pcmp_eq(const Packet8f& a, const Packet8f& b) {
  return pcmp_eq_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pcmp_lt_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  HVX_Vector v_true = Q6_Vb_vsplat_R(0xff);
  HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(b.Get(), a.Get());
  return HVXPacket<T>::Create(Q6_V_vmux_QVV(pred, v_true, Q6_V_vzero()));
}
template <>
EIGEN_STRONG_INLINE Packet32f pcmp_lt(const Packet32f& a, const Packet32f& b) {
  return pcmp_lt_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pcmp_lt(const Packet16f& a, const Packet16f& b) {
  return pcmp_lt_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pcmp_lt(const Packet8f& a, const Packet8f& b) {
  return pcmp_lt_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pcmp_lt_or_nan_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  HVX_Vector v_true = Q6_Vb_vsplat_R(0xff);
  HVX_VectorPred pred = Q6_Q_vcmp_gt_VsfVsf(b.Get(), a.Get());
  return HVXPacket<T>::Create(Q6_V_vmux_QVV(pred, v_true, Q6_V_vzero()));
}
template <>
EIGEN_STRONG_INLINE Packet32f pcmp_lt_or_nan(const Packet32f& a, const Packet32f& b) {
  return pcmp_lt_or_nan_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pcmp_lt_or_nan(const Packet16f& a, const Packet16f& b) {
  return pcmp_lt_or_nan_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pcmp_lt_or_nan(const Packet8f& a, const Packet8f& b) {
  return pcmp_lt_or_nan_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pabs_hvx(const HVXPacket<T>& a) {
  return HVXPacket<T>::Create(a.Get() & Q6_V_vsplat_R(0x7FFFFFFF));
}
template <>
EIGEN_STRONG_INLINE Packet32f pabs(const Packet32f& a) {
  return pabs_hvx(a);
}
template <>
EIGEN_STRONG_INLINE Packet16f pabs(const Packet16f& a) {
  return pabs_hvx(a);
}
template <>
EIGEN_STRONG_INLINE Packet8f pabs(const Packet8f& a) {
  return pabs_hvx(a);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE float pfirst_hvx(const HVXPacket<T>& a) {
  union {
    float array[1];
    HVX_Vector vector;
  } HVX_and_array;
  HVX_and_array.vector = a.Get();
  return HVX_and_array.array[0];
}
template <>
EIGEN_STRONG_INLINE float pfirst(const Packet32f& a) {
  return pfirst_hvx(a);
}
template <>
EIGEN_STRONG_INLINE float pfirst(const Packet16f& a) {
  return pfirst_hvx(a);
}
template <>
EIGEN_STRONG_INLINE float pfirst(const Packet8f& a) {
  return pfirst_hvx(a);
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet32f, 4>& kernel) {
  // Shuffle the 32-bit lanes.
  HVX_VectorPair v_0_1_0 = Q6_W_vshuff_VVR(kernel.packet[1].Get(), kernel.packet[0].Get(), -4);
  HVX_VectorPair v_0_3_2 = Q6_W_vshuff_VVR(kernel.packet[3].Get(), kernel.packet[2].Get(), -4);

  // Shuffle the 64-bit lanes.
  HVX_VectorPair v_1_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_3_2), HEXAGON_HVX_GET_V0(v_0_1_0), -8);
  HVX_VectorPair v_1_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_3_2), HEXAGON_HVX_GET_V1(v_0_1_0), -8);
  kernel.packet[0] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_1_1_0));
  kernel.packet[1] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_1_1_0));
  kernel.packet[2] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_1_3_2));
  kernel.packet[3] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_1_3_2));
}
EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 4>& kernel) {
  // Shuffle the 32-bit lanes.
  HVX_VectorPair v_0_1_0 = Q6_W_vshuff_VVR(kernel.packet[1].Get(), kernel.packet[0].Get(), -4);
  HVX_VectorPair v_0_3_2 = Q6_W_vshuff_VVR(kernel.packet[3].Get(), kernel.packet[2].Get(), -4);

  // Shuffle the 64-bit lanes.
  HVX_VectorPair v_1_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_3_2), HEXAGON_HVX_GET_V0(v_0_1_0), -8);

  kernel.packet[0] = Packet16f::Create(HEXAGON_HVX_GET_V0(v_1_1_0));
  kernel.packet[1] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_1_1_0), HEXAGON_HVX_GET_V0(v_1_1_0), 64));
  kernel.packet[2] = Packet16f::Create(HEXAGON_HVX_GET_V1(v_1_1_0));
  kernel.packet[3] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V1(v_1_1_0), HEXAGON_HVX_GET_V1(v_1_1_0), 64));
}
EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8f, 4>& kernel) {
  // Shuffle the 32-bit lanes.
  HVX_VectorPair v_0_1_0 = Q6_W_vshuff_VVR(kernel.packet[1].Get(), kernel.packet[0].Get(), -4);
  HVX_VectorPair v_0_3_2 = Q6_W_vshuff_VVR(kernel.packet[3].Get(), kernel.packet[2].Get(), -4);

  // Shuffle the 64-bit lanes.
  HVX_VectorPair v_1_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_3_2), HEXAGON_HVX_GET_V0(v_0_1_0), -8);

  kernel.packet[0] = Packet8f::Create(HEXAGON_HVX_GET_V0(v_1_1_0));
  kernel.packet[1] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_1_1_0), HEXAGON_HVX_GET_V0(v_1_1_0), 32));
  kernel.packet[2] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_1_1_0), HEXAGON_HVX_GET_V0(v_1_1_0), 64));
  kernel.packet[3] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_1_1_0), HEXAGON_HVX_GET_V0(v_1_1_0), 96));
}

EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet8f, 8>& kernel) {
  // Shuffle the 32-bit lanes.
  HVX_VectorPair v_0_1_0 = Q6_W_vshuff_VVR(kernel.packet[1].Get(), kernel.packet[0].Get(), -4);
  HVX_VectorPair v_0_3_2 = Q6_W_vshuff_VVR(kernel.packet[3].Get(), kernel.packet[2].Get(), -4);
  HVX_VectorPair v_0_5_4 = Q6_W_vshuff_VVR(kernel.packet[5].Get(), kernel.packet[4].Get(), -4);
  HVX_VectorPair v_0_7_6 = Q6_W_vshuff_VVR(kernel.packet[7].Get(), kernel.packet[6].Get(), -4);

  // Shuffle the 64-bit lanes.
  HVX_VectorPair v_1_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_3_2), HEXAGON_HVX_GET_V0(v_0_1_0), -8);
  HVX_VectorPair v_1_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_7_6), HEXAGON_HVX_GET_V0(v_0_5_4), -8);

  // Shuffle the 128-bit lanes.
  v_0_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_3_2), HEXAGON_HVX_GET_V0(v_1_1_0), -16);

  kernel.packet[0] = Packet8f::Create(HEXAGON_HVX_GET_V0(v_0_1_0));
  kernel.packet[1] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_0_1_0), HEXAGON_HVX_GET_V0(v_0_1_0), 32));
  kernel.packet[2] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_0_1_0), HEXAGON_HVX_GET_V0(v_0_1_0), 64));
  kernel.packet[3] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_0_1_0), HEXAGON_HVX_GET_V0(v_0_1_0), 96));
  kernel.packet[4] = Packet8f::Create(HEXAGON_HVX_GET_V1(v_0_1_0));
  kernel.packet[5] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V1(v_0_1_0), HEXAGON_HVX_GET_V1(v_0_1_0), 32));
  kernel.packet[6] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V1(v_0_1_0), HEXAGON_HVX_GET_V1(v_0_1_0), 64));
  kernel.packet[7] = Packet8f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V1(v_0_1_0), HEXAGON_HVX_GET_V1(v_0_1_0), 96));
}
EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet16f, 16>& kernel) {
  // Shuffle the 32-bit lanes.
  HVX_VectorPair v_0_1_0 = Q6_W_vshuff_VVR(kernel.packet[1].Get(), kernel.packet[0].Get(), -4);
  HVX_VectorPair v_0_3_2 = Q6_W_vshuff_VVR(kernel.packet[3].Get(), kernel.packet[2].Get(), -4);
  HVX_VectorPair v_0_5_4 = Q6_W_vshuff_VVR(kernel.packet[5].Get(), kernel.packet[4].Get(), -4);
  HVX_VectorPair v_0_7_6 = Q6_W_vshuff_VVR(kernel.packet[7].Get(), kernel.packet[6].Get(), -4);
  HVX_VectorPair v_0_9_8 = Q6_W_vshuff_VVR(kernel.packet[9].Get(), kernel.packet[8].Get(), -4);
  HVX_VectorPair v_0_11_10 = Q6_W_vshuff_VVR(kernel.packet[11].Get(), kernel.packet[10].Get(), -4);
  HVX_VectorPair v_0_13_12 = Q6_W_vshuff_VVR(kernel.packet[13].Get(), kernel.packet[12].Get(), -4);
  HVX_VectorPair v_0_15_14 = Q6_W_vshuff_VVR(kernel.packet[15].Get(), kernel.packet[14].Get(), -4);

  // Shuffle the 64-bit lanes.
  HVX_VectorPair v_1_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_3_2), HEXAGON_HVX_GET_V0(v_0_1_0), -8);
  HVX_VectorPair v_1_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_7_6), HEXAGON_HVX_GET_V0(v_0_5_4), -8);
  HVX_VectorPair v_1_5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_11_10), HEXAGON_HVX_GET_V0(v_0_9_8), -8);
  HVX_VectorPair v_1_7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_15_14), HEXAGON_HVX_GET_V0(v_0_13_12), -8);

  // Shuffle the 128-bit lanes.
  v_0_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_3_2), HEXAGON_HVX_GET_V0(v_1_1_0), -16);
  v_0_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_3_2), HEXAGON_HVX_GET_V1(v_1_1_0), -16);
  v_0_9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_7_6), HEXAGON_HVX_GET_V0(v_1_5_4), -16);
  v_0_11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_7_6), HEXAGON_HVX_GET_V1(v_1_5_4), -16);

  // Shuffle the 256-bit lanes.
  v_1_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_9_8), HEXAGON_HVX_GET_V0(v_0_1_0), -32);
  v_1_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_9_8), HEXAGON_HVX_GET_V1(v_0_1_0), -32);
  v_1_5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_11_10), HEXAGON_HVX_GET_V0(v_0_3_2), -32);
  v_1_7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_11_10), HEXAGON_HVX_GET_V1(v_0_3_2), -32);

  kernel.packet[0] = Packet16f::Create(HEXAGON_HVX_GET_V0(v_1_1_0));
  kernel.packet[1] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_1_1_0), HEXAGON_HVX_GET_V0(v_1_1_0), 64));
  kernel.packet[2] = Packet16f::Create(HEXAGON_HVX_GET_V1(v_1_1_0));
  kernel.packet[3] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V1(v_1_1_0), HEXAGON_HVX_GET_V1(v_1_1_0), 64));
  kernel.packet[4] = Packet16f::Create(HEXAGON_HVX_GET_V0(v_1_3_2));
  kernel.packet[5] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_1_3_2), HEXAGON_HVX_GET_V0(v_1_3_2), 64));
  kernel.packet[6] = Packet16f::Create(HEXAGON_HVX_GET_V1(v_1_3_2));
  kernel.packet[7] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V1(v_1_3_2), HEXAGON_HVX_GET_V1(v_1_3_2), 64));
  kernel.packet[8] = Packet16f::Create(HEXAGON_HVX_GET_V0(v_1_5_4));
  kernel.packet[9] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_1_5_4), HEXAGON_HVX_GET_V0(v_1_5_4), 64));
  kernel.packet[10] = Packet16f::Create(HEXAGON_HVX_GET_V1(v_1_5_4));
  kernel.packet[11] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V1(v_1_5_4), HEXAGON_HVX_GET_V1(v_1_5_4), 64));
  kernel.packet[12] = Packet16f::Create(HEXAGON_HVX_GET_V0(v_1_7_6));
  kernel.packet[13] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V0(v_1_7_6), HEXAGON_HVX_GET_V0(v_1_7_6), 64));
  kernel.packet[14] = Packet16f::Create(HEXAGON_HVX_GET_V1(v_1_7_6));
  kernel.packet[15] = Packet16f::Create(Q6_V_valign_VVR(HEXAGON_HVX_GET_V1(v_1_7_6), HEXAGON_HVX_GET_V1(v_1_7_6), 64));
}
EIGEN_STRONG_INLINE void ptranspose(PacketBlock<Packet32f, 32>& kernel) {
  // Shuffle the 32-bit lanes.
  HVX_VectorPair v_0_1_0 = Q6_W_vshuff_VVR(kernel.packet[1].Get(), kernel.packet[0].Get(), -4);
  HVX_VectorPair v_0_3_2 = Q6_W_vshuff_VVR(kernel.packet[3].Get(), kernel.packet[2].Get(), -4);
  HVX_VectorPair v_0_5_4 = Q6_W_vshuff_VVR(kernel.packet[5].Get(), kernel.packet[4].Get(), -4);
  HVX_VectorPair v_0_7_6 = Q6_W_vshuff_VVR(kernel.packet[7].Get(), kernel.packet[6].Get(), -4);
  HVX_VectorPair v_0_9_8 = Q6_W_vshuff_VVR(kernel.packet[9].Get(), kernel.packet[8].Get(), -4);
  HVX_VectorPair v_0_11_10 = Q6_W_vshuff_VVR(kernel.packet[11].Get(), kernel.packet[10].Get(), -4);
  HVX_VectorPair v_0_13_12 = Q6_W_vshuff_VVR(kernel.packet[13].Get(), kernel.packet[12].Get(), -4);
  HVX_VectorPair v_0_15_14 = Q6_W_vshuff_VVR(kernel.packet[15].Get(), kernel.packet[14].Get(), -4);
  HVX_VectorPair v_0_17_16 = Q6_W_vshuff_VVR(kernel.packet[17].Get(), kernel.packet[16].Get(), -4);
  HVX_VectorPair v_0_19_18 = Q6_W_vshuff_VVR(kernel.packet[19].Get(), kernel.packet[18].Get(), -4);
  HVX_VectorPair v_0_21_20 = Q6_W_vshuff_VVR(kernel.packet[21].Get(), kernel.packet[20].Get(), -4);
  HVX_VectorPair v_0_23_22 = Q6_W_vshuff_VVR(kernel.packet[23].Get(), kernel.packet[22].Get(), -4);
  HVX_VectorPair v_0_25_24 = Q6_W_vshuff_VVR(kernel.packet[25].Get(), kernel.packet[24].Get(), -4);
  HVX_VectorPair v_0_27_26 = Q6_W_vshuff_VVR(kernel.packet[27].Get(), kernel.packet[26].Get(), -4);
  HVX_VectorPair v_0_29_28 = Q6_W_vshuff_VVR(kernel.packet[29].Get(), kernel.packet[28].Get(), -4);
  HVX_VectorPair v_0_31_30 = Q6_W_vshuff_VVR(kernel.packet[31].Get(), kernel.packet[30].Get(), -4);

  // Shuffle the 64-bit lanes.
  HVX_VectorPair v_1_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_3_2), HEXAGON_HVX_GET_V0(v_0_1_0), -8);
  HVX_VectorPair v_1_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_3_2), HEXAGON_HVX_GET_V1(v_0_1_0), -8);
  HVX_VectorPair v_1_5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_7_6), HEXAGON_HVX_GET_V0(v_0_5_4), -8);
  HVX_VectorPair v_1_7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_7_6), HEXAGON_HVX_GET_V1(v_0_5_4), -8);
  HVX_VectorPair v_1_9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_11_10), HEXAGON_HVX_GET_V0(v_0_9_8), -8);
  HVX_VectorPair v_1_11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_11_10), HEXAGON_HVX_GET_V1(v_0_9_8), -8);
  HVX_VectorPair v_1_13_12 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_15_14), HEXAGON_HVX_GET_V0(v_0_13_12), -8);
  HVX_VectorPair v_1_15_14 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_15_14), HEXAGON_HVX_GET_V1(v_0_13_12), -8);
  HVX_VectorPair v_1_17_16 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_19_18), HEXAGON_HVX_GET_V0(v_0_17_16), -8);
  HVX_VectorPair v_1_19_18 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_19_18), HEXAGON_HVX_GET_V1(v_0_17_16), -8);
  HVX_VectorPair v_1_21_20 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_23_22), HEXAGON_HVX_GET_V0(v_0_21_20), -8);
  HVX_VectorPair v_1_23_22 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_23_22), HEXAGON_HVX_GET_V1(v_0_21_20), -8);
  HVX_VectorPair v_1_25_24 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_27_26), HEXAGON_HVX_GET_V0(v_0_25_24), -8);
  HVX_VectorPair v_1_27_26 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_27_26), HEXAGON_HVX_GET_V1(v_0_25_24), -8);
  HVX_VectorPair v_1_29_28 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_31_30), HEXAGON_HVX_GET_V0(v_0_29_28), -8);
  HVX_VectorPair v_1_31_30 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_31_30), HEXAGON_HVX_GET_V1(v_0_29_28), -8);

  // Shuffle the 128-bit lanes.
  v_0_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_5_4), HEXAGON_HVX_GET_V0(v_1_1_0), -16);
  v_0_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_5_4), HEXAGON_HVX_GET_V1(v_1_1_0), -16);
  v_0_5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_7_6), HEXAGON_HVX_GET_V0(v_1_3_2), -16);
  v_0_7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_7_6), HEXAGON_HVX_GET_V1(v_1_3_2), -16);
  v_0_9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_13_12), HEXAGON_HVX_GET_V0(v_1_9_8), -16);
  v_0_11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_13_12), HEXAGON_HVX_GET_V1(v_1_9_8), -16);
  v_0_13_12 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_15_14), HEXAGON_HVX_GET_V0(v_1_11_10), -16);
  v_0_15_14 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_15_14), HEXAGON_HVX_GET_V1(v_1_11_10), -16);
  v_0_17_16 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_21_20), HEXAGON_HVX_GET_V0(v_1_17_16), -16);
  v_0_19_18 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_21_20), HEXAGON_HVX_GET_V1(v_1_17_16), -16);
  v_0_21_20 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_23_22), HEXAGON_HVX_GET_V0(v_1_19_18), -16);
  v_0_23_22 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_23_22), HEXAGON_HVX_GET_V1(v_1_19_18), -16);
  v_0_25_24 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_29_28), HEXAGON_HVX_GET_V0(v_1_25_24), -16);
  v_0_27_26 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_29_28), HEXAGON_HVX_GET_V1(v_1_25_24), -16);
  v_0_29_28 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_31_30), HEXAGON_HVX_GET_V0(v_1_27_26), -16);
  v_0_31_30 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_31_30), HEXAGON_HVX_GET_V1(v_1_27_26), -16);

  // Shuffle the 256-bit lanes.
  v_1_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_9_8), HEXAGON_HVX_GET_V0(v_0_1_0), -32);
  v_1_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_9_8), HEXAGON_HVX_GET_V1(v_0_1_0), -32);
  v_1_5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_11_10), HEXAGON_HVX_GET_V0(v_0_3_2), -32);
  v_1_7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_11_10), HEXAGON_HVX_GET_V1(v_0_3_2), -32);
  v_1_9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_13_12), HEXAGON_HVX_GET_V0(v_0_5_4), -32);
  v_1_11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_13_12), HEXAGON_HVX_GET_V1(v_0_5_4), -32);
  v_1_13_12 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_15_14), HEXAGON_HVX_GET_V0(v_0_7_6), -32);
  v_1_15_14 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_15_14), HEXAGON_HVX_GET_V1(v_0_7_6), -32);
  v_1_17_16 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_25_24), HEXAGON_HVX_GET_V0(v_0_17_16), -32);
  v_1_19_18 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_25_24), HEXAGON_HVX_GET_V1(v_0_17_16), -32);
  v_1_21_20 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_27_26), HEXAGON_HVX_GET_V0(v_0_19_18), -32);
  v_1_23_22 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_27_26), HEXAGON_HVX_GET_V1(v_0_19_18), -32);
  v_1_25_24 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_29_28), HEXAGON_HVX_GET_V0(v_0_21_20), -32);
  v_1_27_26 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_29_28), HEXAGON_HVX_GET_V1(v_0_21_20), -32);
  v_1_29_28 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_0_31_30), HEXAGON_HVX_GET_V0(v_0_23_22), -32);
  v_1_31_30 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_0_31_30), HEXAGON_HVX_GET_V1(v_0_23_22), -32);

  // Shuffle the 512-bit lanes.
  v_0_1_0 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_17_16), HEXAGON_HVX_GET_V0(v_1_1_0), -64);
  v_0_3_2 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_17_16), HEXAGON_HVX_GET_V1(v_1_1_0), -64);
  v_0_5_4 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_19_18), HEXAGON_HVX_GET_V0(v_1_3_2), -64);
  v_0_7_6 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_19_18), HEXAGON_HVX_GET_V1(v_1_3_2), -64);
  v_0_9_8 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_21_20), HEXAGON_HVX_GET_V0(v_1_5_4), -64);
  v_0_11_10 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_21_20), HEXAGON_HVX_GET_V1(v_1_5_4), -64);
  v_0_13_12 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_23_22), HEXAGON_HVX_GET_V0(v_1_7_6), -64);
  v_0_15_14 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_23_22), HEXAGON_HVX_GET_V1(v_1_7_6), -64);
  v_0_17_16 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_25_24), HEXAGON_HVX_GET_V0(v_1_9_8), -64);
  v_0_19_18 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_25_24), HEXAGON_HVX_GET_V1(v_1_9_8), -64);
  v_0_21_20 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_27_26), HEXAGON_HVX_GET_V0(v_1_11_10), -64);
  v_0_23_22 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_27_26), HEXAGON_HVX_GET_V1(v_1_11_10), -64);
  v_0_25_24 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_29_28), HEXAGON_HVX_GET_V0(v_1_13_12), -64);
  v_0_27_26 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_29_28), HEXAGON_HVX_GET_V1(v_1_13_12), -64);
  v_0_29_28 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(v_1_31_30), HEXAGON_HVX_GET_V0(v_1_15_14), -64);
  v_0_31_30 = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V1(v_1_31_30), HEXAGON_HVX_GET_V1(v_1_15_14), -64);

  kernel.packet[0] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_1_0));
  kernel.packet[1] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_1_0));
  kernel.packet[2] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_3_2));
  kernel.packet[3] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_3_2));
  kernel.packet[4] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_5_4));
  kernel.packet[5] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_5_4));
  kernel.packet[6] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_7_6));
  kernel.packet[7] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_7_6));
  kernel.packet[8] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_9_8));
  kernel.packet[9] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_9_8));
  kernel.packet[10] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_11_10));
  kernel.packet[11] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_11_10));
  kernel.packet[12] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_13_12));
  kernel.packet[13] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_13_12));
  kernel.packet[14] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_15_14));
  kernel.packet[15] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_15_14));
  kernel.packet[16] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_17_16));
  kernel.packet[17] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_17_16));
  kernel.packet[18] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_19_18));
  kernel.packet[19] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_19_18));
  kernel.packet[20] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_21_20));
  kernel.packet[21] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_21_20));
  kernel.packet[22] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_23_22));
  kernel.packet[23] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_23_22));
  kernel.packet[24] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_25_24));
  kernel.packet[25] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_25_24));
  kernel.packet[26] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_27_26));
  kernel.packet[27] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_27_26));
  kernel.packet[28] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_29_28));
  kernel.packet[29] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_29_28));
  kernel.packet[30] = Packet32f::Create(HEXAGON_HVX_GET_V0(v_0_31_30));
  kernel.packet[31] = Packet32f::Create(HEXAGON_HVX_GET_V1(v_0_31_30));
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE float predux_hvx(const HVXPacket<T>& a) {
  const Index packet_size = unpacket_traits<HVXPacket<T>>::size;
  HVX_Vector vsum = Q6_Vqf32_vadd_VsfVsf(a.Get(), Q6_V_vror_VR(a.Get(), sizeof(float)));
  for (int i = 2; i < packet_size; i <<= 1) {
    vsum = Q6_Vqf32_vadd_Vqf32Vqf32(vsum, Q6_V_vror_VR(vsum, i * sizeof(float)));
  }
  return pfirst(HVXPacket<T>::Create(Q6_Vsf_equals_Vqf32(vsum)));
}
template <>
EIGEN_STRONG_INLINE float predux<Packet32f>(const Packet32f& a) {
  return predux_hvx(a);
}
template <>
EIGEN_STRONG_INLINE float predux<Packet16f>(const Packet16f& a) {
  return predux_hvx(a);
}
template <>
EIGEN_STRONG_INLINE float predux<Packet8f>(const Packet8f& a) {
  return predux_hvx(a);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> ploaddup_hvx(const float* from) {
  constexpr Index size = unpacket_traits<HVXPacket<T>>::size / 2;
  HVX_Vector load = HVX_load_partial<size, 0>(from);
  HVX_VectorPair dup = Q6_W_vshuff_VVR(load, load, -4);
  return HVXPacket<T>::Create(HEXAGON_HVX_GET_V0(dup));
}
template <>
EIGEN_STRONG_INLINE Packet32f ploaddup(const float* from) {
  return ploaddup_hvx<HVXPacketSize::Full>(from);
}
template <>
EIGEN_STRONG_INLINE Packet16f ploaddup(const float* from) {
  return ploaddup_hvx<HVXPacketSize::Half>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8f ploaddup(const float* from) {
  return ploaddup_hvx<HVXPacketSize::Quarter>(from);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> ploadquad_hvx(const float* from) {
  constexpr Index size = unpacket_traits<HVXPacket<T>>::size / 4;
  HVX_Vector load = HVX_load_partial<size, 0>(from);
  HVX_VectorPair dup = Q6_W_vshuff_VVR(load, load, -4);
  HVX_VectorPair quad = Q6_W_vshuff_VVR(HEXAGON_HVX_GET_V0(dup), HEXAGON_HVX_GET_V0(dup), -8);
  return HVXPacket<T>::Create(HEXAGON_HVX_GET_V0(quad));
}
template <>
EIGEN_STRONG_INLINE Packet32f ploadquad(const float* from) {
  return ploadquad_hvx<HVXPacketSize::Full>(from);
}
template <>
EIGEN_STRONG_INLINE Packet16f ploadquad(const float* from) {
  return ploadquad_hvx<HVXPacketSize::Half>(from);
}
template <>
EIGEN_STRONG_INLINE Packet8f ploadquad(const float* from) {
  return ploadquad_hvx<HVXPacketSize::Quarter>(from);
}

template <>
EIGEN_STRONG_INLINE Packet32f preverse(const Packet32f& a) {
  HVX_Vector delta = Q6_Vb_vsplat_R(0x7c);
  return Packet32f::Create(Q6_V_vdelta_VV(a.Get(), delta));
}

template <>
EIGEN_STRONG_INLINE Packet16f preverse(const Packet16f& a) {
  HVX_Vector delta = Q6_Vb_vsplat_R(0x3c);
  return Packet16f::Create(Q6_V_vdelta_VV(a.Get(), delta));
}

template <>
EIGEN_STRONG_INLINE Packet8f preverse(const Packet8f& a) {
  HVX_Vector delta = Q6_Vb_vsplat_R(0x1c);
  return Packet8f::Create(Q6_V_vdelta_VV(a.Get(), delta));
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pmin_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  return HVXPacket<T>::Create(Q6_Vsf_vmin_VsfVsf(a.Get(), b.Get()));
}
template <>
EIGEN_STRONG_INLINE Packet32f pmin(const Packet32f& a, const Packet32f& b) {
  return pmin_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pmin(const Packet16f& a, const Packet16f& b) {
  return pmin_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pmin(const Packet8f& a, const Packet8f& b) {
  return pmin_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pmax_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  return HVXPacket<T>::Create(Q6_Vsf_vmax_VsfVsf(a.Get(), b.Get()));
}
template <>
EIGEN_STRONG_INLINE Packet32f pmax(const Packet32f& a, const Packet32f& b) {
  return pmax_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pmax(const Packet16f& a, const Packet16f& b) {
  return pmax_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pmax(const Packet8f& a, const Packet8f& b) {
  return pmax_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pand_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  return HVXPacket<T>::Create(a.Get() & b.Get());
}
template <>
EIGEN_STRONG_INLINE Packet32f pand(const Packet32f& a, const Packet32f& b) {
  return pand_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pand(const Packet16f& a, const Packet16f& b) {
  return pand_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pand(const Packet8f& a, const Packet8f& b) {
  return pand_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> por_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  return HVXPacket<T>::Create(a.Get() | b.Get());
}
template <>
EIGEN_STRONG_INLINE Packet32f por(const Packet32f& a, const Packet32f& b) {
  return por_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f por(const Packet16f& a, const Packet16f& b) {
  return por_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f por(const Packet8f& a, const Packet8f& b) {
  return por_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pxor_hvx(const HVXPacket<T>& a, const HVXPacket<T>& b) {
  return HVXPacket<T>::Create(a.Get() ^ b.Get());
}
template <>
EIGEN_STRONG_INLINE Packet32f pxor(const Packet32f& a, const Packet32f& b) {
  return pxor_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pxor(const Packet16f& a, const Packet16f& b) {
  return pxor_hvx(a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pxor(const Packet8f& a, const Packet8f& b) {
  return pxor_hvx(a, b);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pnot_hvx(const HVXPacket<T>& a) {
  return HVXPacket<T>::Create(~a.Get());
}
template <>
EIGEN_STRONG_INLINE Packet32f pnot(const Packet32f& a) {
  return pnot_hvx(a);
}
template <>
EIGEN_STRONG_INLINE Packet16f pnot(const Packet16f& a) {
  return pnot_hvx(a);
}
template <>
EIGEN_STRONG_INLINE Packet8f pnot(const Packet8f& a) {
  return pnot_hvx(a);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pselect_hvx(const HVXPacket<T>& mask, const HVXPacket<T>& a, const HVXPacket<T>& b) {
  HVX_VectorPred pred = Q6_Q_vcmp_eq_VwVw(mask.Get(), Q6_V_vzero());
  return HVXPacket<T>::Create(Q6_V_vmux_QVV(pred, b.Get(), a.Get()));
}
template <>
EIGEN_STRONG_INLINE Packet32f pselect(const Packet32f& mask, const Packet32f& a, const Packet32f& b) {
  return pselect_hvx(mask, a, b);
}
template <>
EIGEN_STRONG_INLINE Packet16f pselect(const Packet16f& mask, const Packet16f& a, const Packet16f& b) {
  return pselect_hvx(mask, a, b);
}
template <>
EIGEN_STRONG_INLINE Packet8f pselect(const Packet8f& mask, const Packet8f& a, const Packet8f& b) {
  return pselect_hvx(mask, a, b);
}

template <HVXPacketSize T, typename Op>
EIGEN_STRONG_INLINE float predux_generic(const HVXPacket<T>& a, Op op) {
  const Index packet_size = unpacket_traits<HVXPacket<T>>::size;
  HVXPacket<T> vredux = a;
  for (int i = 1; i < packet_size; i <<= 1) {
    vredux = op(vredux, HVXPacket<T>::Create(Q6_V_vror_VR(vredux.Get(), i * sizeof(float))));
  }
  return pfirst(vredux);
}

template <>
EIGEN_STRONG_INLINE float predux_max(const Packet32f& a) {
  return predux_generic(a, pmax<Packet32f>);
}
template <>
EIGEN_STRONG_INLINE float predux_max(const Packet16f& a) {
  return predux_generic(a, pmax<Packet16f>);
}
template <>
EIGEN_STRONG_INLINE float predux_max(const Packet8f& a) {
  return predux_generic(a, pmax<Packet8f>);
}

template <>
EIGEN_STRONG_INLINE float predux_min(const Packet32f& a) {
  return predux_generic(a, pmin<Packet32f>);
}
template <>
EIGEN_STRONG_INLINE float predux_min(const Packet16f& a) {
  return predux_generic(a, pmin<Packet16f>);
}
template <>
EIGEN_STRONG_INLINE float predux_min(const Packet8f& a) {
  return predux_generic(a, pmin<Packet8f>);
}

template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet32f& a) {
  return predux_generic(a, por<Packet32f>) != 0.0f;
}
template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet16f& a) {
  return predux_generic(a, por<Packet16f>) != 0.0f;
}
template <>
EIGEN_STRONG_INLINE bool predux_any(const Packet8f& a) {
  return predux_generic(a, por<Packet8f>) != 0.0f;
}

static const float index_vsf[32]
    __attribute__((aligned(__HVX_LENGTH__))) = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> plset_hvx(const float& a) {
  return padd(pload<HVXPacket<T>>(index_vsf), pset1<HVXPacket<T>>(a));
}
template <>
EIGEN_STRONG_INLINE Packet32f plset(const float& a) {
  return plset_hvx<HVXPacketSize::Full>(a);
}
template <>
EIGEN_STRONG_INLINE Packet16f plset(const float& a) {
  return plset_hvx<HVXPacketSize::Half>(a);
}
template <>
EIGEN_STRONG_INLINE Packet8f plset(const float& a) {
  return plset_hvx<HVXPacketSize::Quarter>(a);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE void pscatter_hvx(float* to, const HVXPacket<T>& from, Index stride) {
  const Index packet_size = unpacket_traits<HVXPacket<T>>::size;
  float elements[packet_size] __attribute__((aligned(__HVX_LENGTH__)));
  pstore<float>(elements, from);
  for (Index i = 0; i < packet_size; ++i) {
    to[i * stride] = elements[i];
  }
}
template <>
EIGEN_STRONG_INLINE void pscatter<float, Packet32f>(float* to, const Packet32f& from, Index stride) {
  pscatter_hvx(to, from, stride);
}
template <>
EIGEN_STRONG_INLINE void pscatter<float, Packet16f>(float* to, const Packet16f& from, Index stride) {
  pscatter_hvx(to, from, stride);
}
template <>
EIGEN_STRONG_INLINE void pscatter<float, Packet8f>(float* to, const Packet8f& from, Index stride) {
  pscatter_hvx(to, from, stride);
}

template <HVXPacketSize T>
EIGEN_STRONG_INLINE HVXPacket<T> pgather_hvx(const float* from, Index stride) {
  const Index packet_size = unpacket_traits<HVXPacket<T>>::size;
  float elements[packet_size] __attribute__((aligned(__HVX_LENGTH__)));
  for (Index i = 0; i < packet_size; i++) {
    elements[i] = from[i * stride];
  }
  return pload<HVXPacket<T>>(elements);
}
template <>
EIGEN_STRONG_INLINE Packet32f pgather<float, Packet32f>(const float* from, Index stride) {
  return pgather_hvx<HVXPacketSize::Full>(from, stride);
}
template <>
EIGEN_STRONG_INLINE Packet16f pgather<float, Packet16f>(const float* from, Index stride) {
  return pgather_hvx<HVXPacketSize::Half>(from, stride);
}
template <>
EIGEN_STRONG_INLINE Packet8f pgather<float, Packet8f>(const float* from, Index stride) {
  return pgather_hvx<HVXPacketSize::Quarter>(from, stride);
}

}  // end namespace internal
}  // end namespace Eigen

#endif  // __HVX__ && (__HVX_LENGTH__ == 128) && __HVX_ARCH__ >= 68

#endif  // EIGEN_HVX_PACKET_MATH_H
