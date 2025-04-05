#pragma once
#ifndef CPUINFO_H
#define CPUINFO_H

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include <stdint.h>

/* Identify architecture and define corresponding macro */

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(_M_IX86)
#define CPUINFO_ARCH_X86 1
#endif

#if defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
#define CPUINFO_ARCH_X86_64 1
#endif

#if defined(__arm__) || defined(_M_ARM)
#define CPUINFO_ARCH_ARM 1
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define CPUINFO_ARCH_ARM64 1
#endif

#if defined(__PPC64__) || defined(__powerpc64__) || defined(_ARCH_PPC64)
#define CPUINFO_ARCH_PPC64 1
#endif

#if defined(__asmjs__)
#define CPUINFO_ARCH_ASMJS 1
#endif

#if defined(__wasm__)
#if defined(__wasm_simd128__)
#define CPUINFO_ARCH_WASMSIMD 1
#else
#define CPUINFO_ARCH_WASM 1
#endif
#endif

#if defined(__riscv)
#if (__riscv_xlen == 32)
#define CPUINFO_ARCH_RISCV32 1
#elif (__riscv_xlen == 64)
#define CPUINFO_ARCH_RISCV64 1
#endif
#endif

/* Define other architecture-specific macros as 0 */

#ifndef CPUINFO_ARCH_X86
#define CPUINFO_ARCH_X86 0
#endif

#ifndef CPUINFO_ARCH_X86_64
#define CPUINFO_ARCH_X86_64 0
#endif

#ifndef CPUINFO_ARCH_ARM
#define CPUINFO_ARCH_ARM 0
#endif

#ifndef CPUINFO_ARCH_ARM64
#define CPUINFO_ARCH_ARM64 0
#endif

#ifndef CPUINFO_ARCH_PPC64
#define CPUINFO_ARCH_PPC64 0
#endif

#ifndef CPUINFO_ARCH_ASMJS
#define CPUINFO_ARCH_ASMJS 0
#endif

#ifndef CPUINFO_ARCH_WASM
#define CPUINFO_ARCH_WASM 0
#endif

#ifndef CPUINFO_ARCH_WASMSIMD
#define CPUINFO_ARCH_WASMSIMD 0
#endif

#ifndef CPUINFO_ARCH_RISCV32
#define CPUINFO_ARCH_RISCV32 0
#endif

#ifndef CPUINFO_ARCH_RISCV64
#define CPUINFO_ARCH_RISCV64 0
#endif

#if CPUINFO_ARCH_X86 && defined(_MSC_VER)
#define CPUINFO_ABI __cdecl
#elif CPUINFO_ARCH_X86 && defined(__GNUC__)
#define CPUINFO_ABI __attribute__((__cdecl__))
#else
#define CPUINFO_ABI
#endif

#define CPUINFO_CACHE_UNIFIED 0x00000001
#define CPUINFO_CACHE_INCLUSIVE 0x00000002
#define CPUINFO_CACHE_COMPLEX_INDEXING 0x00000004

struct cpuinfo_cache {
	/** Cache size in bytes */
	uint32_t size;
	/** Number of ways of associativity */
	uint32_t associativity;
	/** Number of sets */
	uint32_t sets;
	/** Number of partitions */
	uint32_t partitions;
	/** Line size in bytes */
	uint32_t line_size;
	/**
	 * Binary characteristics of the cache (unified cache, inclusive cache,
	 * cache with complex indexing).
	 *
	 * @see CPUINFO_CACHE_UNIFIED, CPUINFO_CACHE_INCLUSIVE,
	 * CPUINFO_CACHE_COMPLEX_INDEXING
	 */
	uint32_t flags;
	/** Index of the first logical processor that shares this cache */
	uint32_t processor_start;
	/** Number of logical processors that share this cache */
	uint32_t processor_count;
};

struct cpuinfo_trace_cache {
	uint32_t uops;
	uint32_t associativity;
};

#define CPUINFO_PAGE_SIZE_4KB 0x1000
#define CPUINFO_PAGE_SIZE_1MB 0x100000
#define CPUINFO_PAGE_SIZE_2MB 0x200000
#define CPUINFO_PAGE_SIZE_4MB 0x400000
#define CPUINFO_PAGE_SIZE_16MB 0x1000000
#define CPUINFO_PAGE_SIZE_1GB 0x40000000

struct cpuinfo_tlb {
	uint32_t entries;
	uint32_t associativity;
	uint64_t pages;
};

/** Vendor of processor core design */
enum cpuinfo_vendor {
	/** Processor vendor is not known to the library, or the library failed
	   to get vendor information from the OS. */
	cpuinfo_vendor_unknown = 0,

	/* Active vendors of modern CPUs */

	/**
	 * Intel Corporation. Vendor of x86, x86-64, IA64, and ARM processor
	 * microarchitectures.
	 *
	 * Sold its ARM design subsidiary in 2006. The last ARM processor design
	 * was released in 2004.
	 */
	cpuinfo_vendor_intel = 1,
	/** Advanced Micro Devices, Inc. Vendor of x86 and x86-64 processor
	   microarchitectures. */
	cpuinfo_vendor_amd = 2,
	/** ARM Holdings plc. Vendor of ARM and ARM64 processor
	   microarchitectures. */
	cpuinfo_vendor_arm = 3,
	/** Qualcomm Incorporated. Vendor of ARM and ARM64 processor
	   microarchitectures. */
	cpuinfo_vendor_qualcomm = 4,
	/** Apple Inc. Vendor of ARM and ARM64 processor microarchitectures. */
	cpuinfo_vendor_apple = 5,
	/** Samsung Electronics Co., Ltd. Vendir if ARM64 processor
	   microarchitectures. */
	cpuinfo_vendor_samsung = 6,
	/** Nvidia Corporation. Vendor of ARM64-compatible processor
	   microarchitectures. */
	cpuinfo_vendor_nvidia = 7,
	/** MIPS Technologies, Inc. Vendor of MIPS processor microarchitectures.
	 */
	cpuinfo_vendor_mips = 8,
	/** International Business Machines Corporation. Vendor of PowerPC
	   processor microarchitectures. */
	cpuinfo_vendor_ibm = 9,
	/** Ingenic Semiconductor. Vendor of MIPS processor microarchitectures.
	 */
	cpuinfo_vendor_ingenic = 10,
	/**
	 * VIA Technologies, Inc. Vendor of x86 and x86-64 processor
	 * microarchitectures.
	 *
	 * Processors are designed by Centaur Technology, a subsidiary of VIA
	 * Technologies.
	 */
	cpuinfo_vendor_via = 11,
	/** Cavium, Inc. Vendor of ARM64 processor microarchitectures. */
	cpuinfo_vendor_cavium = 12,
	/** Broadcom, Inc. Vendor of ARM processor microarchitectures. */
	cpuinfo_vendor_broadcom = 13,
	/** Applied Micro Circuits Corporation (APM). Vendor of ARM64 processor
	   microarchitectures. */
	cpuinfo_vendor_apm = 14,
	/**
	 * Huawei Technologies Co., Ltd. Vendor of ARM64 processor
	 * microarchitectures.
	 *
	 * Processors are designed by HiSilicon, a subsidiary of Huawei.
	 */
	cpuinfo_vendor_huawei = 15,
	/**
	 * Hygon (Chengdu Haiguang Integrated Circuit Design Co., Ltd), Vendor
	 * of x86-64 processor microarchitectures.
	 *
	 * Processors are variants of AMD cores.
	 */
	cpuinfo_vendor_hygon = 16,
	/** SiFive, Inc. Vendor of RISC-V processor microarchitectures. */
	cpuinfo_vendor_sifive = 17,

	/* Active vendors of embedded CPUs */

	/** Texas Instruments Inc. Vendor of ARM processor microarchitectures.
	 */
	cpuinfo_vendor_texas_instruments = 30,
	/** Marvell Technology Group Ltd. Vendor of ARM processor
	 * microarchitectures.
	 */
	cpuinfo_vendor_marvell = 31,
	/** RDC Semiconductor Co., Ltd. Vendor of x86 processor
	   microarchitectures. */
	cpuinfo_vendor_rdc = 32,
	/** DM&P Electronics Inc. Vendor of x86 processor microarchitectures. */
	cpuinfo_vendor_dmp = 33,
	/** Motorola, Inc. Vendor of PowerPC and ARM processor
	   microarchitectures. */
	cpuinfo_vendor_motorola = 34,

	/* Defunct CPU vendors */

	/**
	 * Transmeta Corporation. Vendor of x86 processor microarchitectures.
	 *
	 * Now defunct. The last processor design was released in 2004.
	 * Transmeta processors implemented VLIW ISA and used binary translation
	 * to execute x86 code.
	 */
	cpuinfo_vendor_transmeta = 50,
	/**
	 * Cyrix Corporation. Vendor of x86 processor microarchitectures.
	 *
	 * Now defunct. The last processor design was released in 1996.
	 */
	cpuinfo_vendor_cyrix = 51,
	/**
	 * Rise Technology. Vendor of x86 processor microarchitectures.
	 *
	 * Now defunct. The last processor design was released in 1999.
	 */
	cpuinfo_vendor_rise = 52,
	/**
	 * National Semiconductor. Vendor of x86 processor microarchitectures.
	 *
	 * Sold its x86 design subsidiary in 1999. The last processor design was
	 * released in 1998.
	 */
	cpuinfo_vendor_nsc = 53,
	/**
	 * Silicon Integrated Systems. Vendor of x86 processor
	 * microarchitectures.
	 *
	 * Sold its x86 design subsidiary in 2001. The last processor design was
	 * released in 2001.
	 */
	cpuinfo_vendor_sis = 54,
	/**
	 * NexGen. Vendor of x86 processor microarchitectures.
	 *
	 * Now defunct. The last processor design was released in 1994.
	 * NexGen designed the first x86 microarchitecture which decomposed x86
	 * instructions into simple microoperations.
	 */
	cpuinfo_vendor_nexgen = 55,
	/**
	 * United Microelectronics Corporation. Vendor of x86 processor
	 * microarchitectures.
	 *
	 * Ceased x86 in the early 1990s. The last processor design was released
	 * in 1991. Designed U5C and U5D processors. Both are 486 level.
	 */
	cpuinfo_vendor_umc = 56,
	/**
	 * Digital Equipment Corporation. Vendor of ARM processor
	 * microarchitecture.
	 *
	 * Sold its ARM designs in 1997. The last processor design was released
	 * in 1997.
	 */
	cpuinfo_vendor_dec = 57,
};

/**
 * Processor microarchitecture
 *
 * Processors with different microarchitectures often have different instruction
 * performance characteristics, and may have dramatically different pipeline
 * organization.
 */
enum cpuinfo_uarch {
	/** Microarchitecture is unknown, or the library failed to get
	   information about the microarchitecture from OS */
	cpuinfo_uarch_unknown = 0,

	/** Pentium and Pentium MMX microarchitecture. */
	cpuinfo_uarch_p5 = 0x00100100,
	/** Intel Quark microarchitecture. */
	cpuinfo_uarch_quark = 0x00100101,

	/** Pentium Pro, Pentium II, and Pentium III. */
	cpuinfo_uarch_p6 = 0x00100200,
	/** Pentium M. */
	cpuinfo_uarch_dothan = 0x00100201,
	/** Intel Core microarchitecture. */
	cpuinfo_uarch_yonah = 0x00100202,
	/** Intel Core 2 microarchitecture on 65 nm process. */
	cpuinfo_uarch_conroe = 0x00100203,
	/** Intel Core 2 microarchitecture on 45 nm process. */
	cpuinfo_uarch_penryn = 0x00100204,
	/** Intel Nehalem and Westmere microarchitectures (Core i3/i5/i7 1st
	   gen). */
	cpuinfo_uarch_nehalem = 0x00100205,
	/** Intel Sandy Bridge microarchitecture (Core i3/i5/i7 2nd gen). */
	cpuinfo_uarch_sandy_bridge = 0x00100206,
	/** Intel Ivy Bridge microarchitecture (Core i3/i5/i7 3rd gen). */
	cpuinfo_uarch_ivy_bridge = 0x00100207,
	/** Intel Haswell microarchitecture (Core i3/i5/i7 4th gen). */
	cpuinfo_uarch_haswell = 0x00100208,
	/** Intel Broadwell microarchitecture. */
	cpuinfo_uarch_broadwell = 0x00100209,
	/** Intel Sky Lake microarchitecture (14 nm, including
	   Kaby/Coffee/Whiskey/Amber/Comet/Cascade/Cooper Lake). */
	cpuinfo_uarch_sky_lake = 0x0010020A,
	/** DEPRECATED (Intel Kaby Lake microarchitecture). */
	cpuinfo_uarch_kaby_lake = 0x0010020A,
	/** Intel Palm Cove microarchitecture (10 nm, Cannon Lake). */
	cpuinfo_uarch_palm_cove = 0x0010020B,
	/** Intel Sunny Cove microarchitecture (10 nm, Ice Lake). */
	cpuinfo_uarch_sunny_cove = 0x0010020C,

	/** Pentium 4 with Willamette, Northwood, or Foster cores. */
	cpuinfo_uarch_willamette = 0x00100300,
	/** Pentium 4 with Prescott and later cores. */
	cpuinfo_uarch_prescott = 0x00100301,

	/** Intel Atom on 45 nm process. */
	cpuinfo_uarch_bonnell = 0x00100400,
	/** Intel Atom on 32 nm process. */
	cpuinfo_uarch_saltwell = 0x00100401,
	/** Intel Silvermont microarchitecture (22 nm out-of-order Atom). */
	cpuinfo_uarch_silvermont = 0x00100402,
	/** Intel Airmont microarchitecture (14 nm out-of-order Atom). */
	cpuinfo_uarch_airmont = 0x00100403,
	/** Intel Goldmont microarchitecture (Denverton, Apollo Lake). */
	cpuinfo_uarch_goldmont = 0x00100404,
	/** Intel Goldmont Plus microarchitecture (Gemini Lake). */
	cpuinfo_uarch_goldmont_plus = 0x00100405,

	/** Intel Knights Ferry HPC boards. */
	cpuinfo_uarch_knights_ferry = 0x00100500,
	/** Intel Knights Corner HPC boards (aka Xeon Phi). */
	cpuinfo_uarch_knights_corner = 0x00100501,
	/** Intel Knights Landing microarchitecture (second-gen MIC). */
	cpuinfo_uarch_knights_landing = 0x00100502,
	/** Intel Knights Hill microarchitecture (third-gen MIC). */
	cpuinfo_uarch_knights_hill = 0x00100503,
	/** Intel Knights Mill Xeon Phi. */
	cpuinfo_uarch_knights_mill = 0x00100504,

	/** Intel/Marvell XScale series. */
	cpuinfo_uarch_xscale = 0x00100600,

	/** AMD K5. */
	cpuinfo_uarch_k5 = 0x00200100,
	/** AMD K6 and alike. */
	cpuinfo_uarch_k6 = 0x00200101,
	/** AMD Athlon and Duron. */
	cpuinfo_uarch_k7 = 0x00200102,
	/** AMD Athlon 64, Opteron 64. */
	cpuinfo_uarch_k8 = 0x00200103,
	/** AMD Family 10h (Barcelona, Istambul, Magny-Cours). */
	cpuinfo_uarch_k10 = 0x00200104,
	/**
	 * AMD Bulldozer microarchitecture
	 * Zambezi FX-series CPUs, Zurich, Valencia and Interlagos Opteron CPUs.
	 */
	cpuinfo_uarch_bulldozer = 0x00200105,
	/**
	 * AMD Piledriver microarchitecture
	 * Vishera FX-series CPUs, Trinity and Richland APUs, Delhi, Seoul, Abu
	 * Dhabi Opteron CPUs.
	 */
	cpuinfo_uarch_piledriver = 0x00200106,
	/** AMD Steamroller microarchitecture (Kaveri APUs). */
	cpuinfo_uarch_steamroller = 0x00200107,
	/** AMD Excavator microarchitecture (Carizzo APUs). */
	cpuinfo_uarch_excavator = 0x00200108,
	/** AMD Zen microarchitecture (12/14 nm Ryzen and EPYC CPUs). */
	cpuinfo_uarch_zen = 0x00200109,
	/** AMD Zen 2 microarchitecture (7 nm Ryzen and EPYC CPUs). */
	cpuinfo_uarch_zen2 = 0x0020010A,
	/** AMD Zen 3 microarchitecture. */
	cpuinfo_uarch_zen3 = 0x0020010B,
	/** AMD Zen 4 microarchitecture. */
	cpuinfo_uarch_zen4 = 0x0020010C,

	/** NSC Geode and AMD Geode GX and LX. */
	cpuinfo_uarch_geode = 0x00200200,
	/** AMD Bobcat mobile microarchitecture. */
	cpuinfo_uarch_bobcat = 0x00200201,
	/** AMD Jaguar mobile microarchitecture. */
	cpuinfo_uarch_jaguar = 0x00200202,
	/** AMD Puma mobile microarchitecture. */
	cpuinfo_uarch_puma = 0x00200203,

	/** ARM7 series. */
	cpuinfo_uarch_arm7 = 0x00300100,
	/** ARM9 series. */
	cpuinfo_uarch_arm9 = 0x00300101,
	/** ARM 1136, ARM 1156, ARM 1176, or ARM 11MPCore. */
	cpuinfo_uarch_arm11 = 0x00300102,

	/** ARM Cortex-A5. */
	cpuinfo_uarch_cortex_a5 = 0x00300205,
	/** ARM Cortex-A7. */
	cpuinfo_uarch_cortex_a7 = 0x00300207,
	/** ARM Cortex-A8. */
	cpuinfo_uarch_cortex_a8 = 0x00300208,
	/** ARM Cortex-A9. */
	cpuinfo_uarch_cortex_a9 = 0x00300209,
	/** ARM Cortex-A12. */
	cpuinfo_uarch_cortex_a12 = 0x00300212,
	/** ARM Cortex-A15. */
	cpuinfo_uarch_cortex_a15 = 0x00300215,
	/** ARM Cortex-A17. */
	cpuinfo_uarch_cortex_a17 = 0x00300217,

	/** ARM Cortex-A32. */
	cpuinfo_uarch_cortex_a32 = 0x00300332,
	/** ARM Cortex-A35. */
	cpuinfo_uarch_cortex_a35 = 0x00300335,
	/** ARM Cortex-A53. */
	cpuinfo_uarch_cortex_a53 = 0x00300353,
	/** ARM Cortex-A55 revision 0 (restricted dual-issue capabilities
	   compared to revision 1+). */
	cpuinfo_uarch_cortex_a55r0 = 0x00300354,
	/** ARM Cortex-A55. */
	cpuinfo_uarch_cortex_a55 = 0x00300355,
	/** ARM Cortex-A57. */
	cpuinfo_uarch_cortex_a57 = 0x00300357,
	/** ARM Cortex-A65. */
	cpuinfo_uarch_cortex_a65 = 0x00300365,
	/** ARM Cortex-A72. */
	cpuinfo_uarch_cortex_a72 = 0x00300372,
	/** ARM Cortex-A73. */
	cpuinfo_uarch_cortex_a73 = 0x00300373,
	/** ARM Cortex-A75. */
	cpuinfo_uarch_cortex_a75 = 0x00300375,
	/** ARM Cortex-A76. */
	cpuinfo_uarch_cortex_a76 = 0x00300376,
	/** ARM Cortex-A77. */
	cpuinfo_uarch_cortex_a77 = 0x00300377,
	/** ARM Cortex-A78. */
	cpuinfo_uarch_cortex_a78 = 0x00300378,

	/** ARM Neoverse N1. */
	cpuinfo_uarch_neoverse_n1 = 0x00300400,
	/** ARM Neoverse E1. */
	cpuinfo_uarch_neoverse_e1 = 0x00300401,
	/** ARM Neoverse V1. */
	cpuinfo_uarch_neoverse_v1 = 0x00300402,
	/** ARM Neoverse N2. */
	cpuinfo_uarch_neoverse_n2 = 0x00300403,
	/** ARM Neoverse V2. */
	cpuinfo_uarch_neoverse_v2 = 0x00300404,

	/** ARM Cortex-X1. */
	cpuinfo_uarch_cortex_x1 = 0x00300501,
	/** ARM Cortex-X2. */
	cpuinfo_uarch_cortex_x2 = 0x00300502,
	/** ARM Cortex-X3. */
	cpuinfo_uarch_cortex_x3 = 0x00300503,
	/** ARM Cortex-X4. */
	cpuinfo_uarch_cortex_x4 = 0x00300504,

	/** ARM Cortex-A510. */
	cpuinfo_uarch_cortex_a510 = 0x00300551,
	/** ARM Cortex-A520. */
	cpuinfo_uarch_cortex_a520 = 0x00300552,
	/** ARM Cortex-A710. */
	cpuinfo_uarch_cortex_a710 = 0x00300571,
	/** ARM Cortex-A715. */
	cpuinfo_uarch_cortex_a715 = 0x00300572,
	/** ARM Cortex-A720. */
	cpuinfo_uarch_cortex_a720 = 0x00300573,

	/** Qualcomm Scorpion. */
	cpuinfo_uarch_scorpion = 0x00400100,
	/** Qualcomm Krait. */
	cpuinfo_uarch_krait = 0x00400101,
	/** Qualcomm Kryo. */
	cpuinfo_uarch_kryo = 0x00400102,
	/** Qualcomm Falkor. */
	cpuinfo_uarch_falkor = 0x00400103,
	/** Qualcomm Saphira. */
	cpuinfo_uarch_saphira = 0x00400104,

	/** Nvidia Denver. */
	cpuinfo_uarch_denver = 0x00500100,
	/** Nvidia Denver 2. */
	cpuinfo_uarch_denver2 = 0x00500101,
	/** Nvidia Carmel. */
	cpuinfo_uarch_carmel = 0x00500102,

	/** Samsung Exynos M1 (Exynos 8890 big cores). */
	cpuinfo_uarch_exynos_m1 = 0x00600100,
	/** Samsung Exynos M2 (Exynos 8895 big cores). */
	cpuinfo_uarch_exynos_m2 = 0x00600101,
	/** Samsung Exynos M3 (Exynos 9810 big cores). */
	cpuinfo_uarch_exynos_m3 = 0x00600102,
	/** Samsung Exynos M4 (Exynos 9820 big cores). */
	cpuinfo_uarch_exynos_m4 = 0x00600103,
	/** Samsung Exynos M5 (Exynos 9830 big cores). */
	cpuinfo_uarch_exynos_m5 = 0x00600104,

	/* Deprecated synonym for Cortex-A76 */
	cpuinfo_uarch_cortex_a76ae = 0x00300376,
	/* Deprecated names for Exynos. */
	cpuinfo_uarch_mongoose_m1 = 0x00600100,
	cpuinfo_uarch_mongoose_m2 = 0x00600101,
	cpuinfo_uarch_meerkat_m3 = 0x00600102,
	cpuinfo_uarch_meerkat_m4 = 0x00600103,

	/** Apple A6 and A6X processors. */
	cpuinfo_uarch_swift = 0x00700100,
	/** Apple A7 processor. */
	cpuinfo_uarch_cyclone = 0x00700101,
	/** Apple A8 and A8X processor. */
	cpuinfo_uarch_typhoon = 0x00700102,
	/** Apple A9 and A9X processor. */
	cpuinfo_uarch_twister = 0x00700103,
	/** Apple A10 and A10X processor. */
	cpuinfo_uarch_hurricane = 0x00700104,
	/** Apple A11 processor (big cores). */
	cpuinfo_uarch_monsoon = 0x00700105,
	/** Apple A11 processor (little cores). */
	cpuinfo_uarch_mistral = 0x00700106,
	/** Apple A12 processor (big cores). */
	cpuinfo_uarch_vortex = 0x00700107,
	/** Apple A12 processor (little cores). */
	cpuinfo_uarch_tempest = 0x00700108,
	/** Apple A13 processor (big cores). */
	cpuinfo_uarch_lightning = 0x00700109,
	/** Apple A13 processor (little cores). */
	cpuinfo_uarch_thunder = 0x0070010A,
	/** Apple A14 / M1 processor (big cores). */
	cpuinfo_uarch_firestorm = 0x0070010B,
	/** Apple A14 / M1 processor (little cores). */
	cpuinfo_uarch_icestorm = 0x0070010C,
	/** Apple A15 / M2 processor (big cores). */
	cpuinfo_uarch_avalanche = 0x0070010D,
	/** Apple A15 / M2 processor (little cores). */
	cpuinfo_uarch_blizzard = 0x0070010E,

	/** Cavium ThunderX. */
	cpuinfo_uarch_thunderx = 0x00800100,
	/** Cavium ThunderX2 (originally Broadcom Vulkan). */
	cpuinfo_uarch_thunderx2 = 0x00800200,

	/** Marvell PJ4. */
	cpuinfo_uarch_pj4 = 0x00900100,

	/** Broadcom Brahma B15. */
	cpuinfo_uarch_brahma_b15 = 0x00A00100,
	/** Broadcom Brahma B53. */
	cpuinfo_uarch_brahma_b53 = 0x00A00101,

	/** Applied Micro X-Gene. */
	cpuinfo_uarch_xgene = 0x00B00100,

	/* Hygon Dhyana (a modification of AMD Zen for Chinese market). */
	cpuinfo_uarch_dhyana = 0x01000100,

	/** HiSilicon TaiShan v110 (Huawei Kunpeng 920 series processors). */
	cpuinfo_uarch_taishan_v110 = 0x00C00100,
};

struct cpuinfo_processor {
	/** SMT (hyperthread) ID within a core */
	uint32_t smt_id;
	/** Core containing this logical processor */
	const struct cpuinfo_core* core;
	/** Cluster of cores containing this logical processor */
	const struct cpuinfo_cluster* cluster;
	/** Physical package containing this logical processor */
	const struct cpuinfo_package* package;
#if defined(__linux__)
	/**
	 * Linux-specific ID for the logical processor:
	 * - Linux kernel exposes information about this logical processor in
	 * /sys/devices/system/cpu/cpu<linux_id>/
	 * - Bit <linux_id> in the cpu_set_t identifies this logical processor
	 */
	int linux_id;
#endif
#if defined(_WIN32) || defined(__CYGWIN__)
	/** Windows-specific ID for the group containing the logical processor.
	 */
	uint16_t windows_group_id;
	/**
	 * Windows-specific ID of the logical processor within its group:
	 * - Bit <windows_processor_id> in the KAFFINITY mask identifies this
	 * logical processor within its group.
	 */
	uint16_t windows_processor_id;
#endif
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	/** APIC ID (unique x86-specific ID of the logical processor) */
	uint32_t apic_id;
#endif
	struct {
		/** Level 1 instruction cache */
		const struct cpuinfo_cache* l1i;
		/** Level 1 data cache */
		const struct cpuinfo_cache* l1d;
		/** Level 2 unified or data cache */
		const struct cpuinfo_cache* l2;
		/** Level 3 unified or data cache */
		const struct cpuinfo_cache* l3;
		/** Level 4 unified or data cache */
		const struct cpuinfo_cache* l4;
	} cache;
};

struct cpuinfo_core {
	/** Index of the first logical processor on this core. */
	uint32_t processor_start;
	/** Number of logical processors on this core */
	uint32_t processor_count;
	/** Core ID within a package */
	uint32_t core_id;
	/** Cluster containing this core */
	const struct cpuinfo_cluster* cluster;
	/** Physical package containing this core. */
	const struct cpuinfo_package* package;
	/** Vendor of the CPU microarchitecture for this core */
	enum cpuinfo_vendor vendor;
	/** CPU microarchitecture for this core */
	enum cpuinfo_uarch uarch;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	/** Value of CPUID leaf 1 EAX register for this core */
	uint32_t cpuid;
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	/** Value of Main ID Register (MIDR) for this core */
	uint32_t midr;
#endif
	/** Clock rate (non-Turbo) of the core, in Hz */
	uint64_t frequency;
};

struct cpuinfo_cluster {
	/** Index of the first logical processor in the cluster */
	uint32_t processor_start;
	/** Number of logical processors in the cluster */
	uint32_t processor_count;
	/** Index of the first core in the cluster */
	uint32_t core_start;
	/** Number of cores on the cluster */
	uint32_t core_count;
	/** Cluster ID within a package */
	uint32_t cluster_id;
	/** Physical package containing the cluster */
	const struct cpuinfo_package* package;
	/** CPU microarchitecture vendor of the cores in the cluster */
	enum cpuinfo_vendor vendor;
	/** CPU microarchitecture of the cores in the cluster */
	enum cpuinfo_uarch uarch;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	/** Value of CPUID leaf 1 EAX register of the cores in the cluster */
	uint32_t cpuid;
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	/** Value of Main ID Register (MIDR) of the cores in the cluster */
	uint32_t midr;
#endif
	/** Clock rate (non-Turbo) of the cores in the cluster, in Hz */
	uint64_t frequency;
};

#define CPUINFO_PACKAGE_NAME_MAX 48

struct cpuinfo_package {
	/** SoC or processor chip model name */
	char name[CPUINFO_PACKAGE_NAME_MAX];
	/** Index of the first logical processor on this physical package */
	uint32_t processor_start;
	/** Number of logical processors on this physical package */
	uint32_t processor_count;
	/** Index of the first core on this physical package */
	uint32_t core_start;
	/** Number of cores on this physical package */
	uint32_t core_count;
	/** Index of the first cluster of cores on this physical package */
	uint32_t cluster_start;
	/** Number of clusters of cores on this physical package */
	uint32_t cluster_count;
};

struct cpuinfo_uarch_info {
	/** Type of CPU microarchitecture */
	enum cpuinfo_uarch uarch;
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	/** Value of CPUID leaf 1 EAX register for the microarchitecture */
	uint32_t cpuid;
#elif CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	/** Value of Main ID Register (MIDR) for the microarchitecture */
	uint32_t midr;
#endif
	/** Number of logical processors with the microarchitecture */
	uint32_t processor_count;
	/** Number of cores with the microarchitecture */
	uint32_t core_count;
};

#ifdef __cplusplus
extern "C" {
#endif

bool CPUINFO_ABI cpuinfo_initialize(void);

void CPUINFO_ABI cpuinfo_deinitialize(void);

#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
/* This structure is not a part of stable API. Use cpuinfo_has_x86_* functions
 * instead. */
struct cpuinfo_x86_isa {
#if CPUINFO_ARCH_X86
	bool rdtsc;
#endif
	bool rdtscp;
	bool rdpid;
	bool sysenter;
#if CPUINFO_ARCH_X86
	bool syscall;
#endif
	bool msr;
	bool clzero;
	bool clflush;
	bool clflushopt;
	bool mwait;
	bool mwaitx;
#if CPUINFO_ARCH_X86
	bool emmx;
#endif
	bool fxsave;
	bool xsave;
#if CPUINFO_ARCH_X86
	bool fpu;
	bool mmx;
	bool mmx_plus;
#endif
	bool three_d_now;
	bool three_d_now_plus;
#if CPUINFO_ARCH_X86
	bool three_d_now_geode;
#endif
	bool prefetch;
	bool prefetchw;
	bool prefetchwt1;
#if CPUINFO_ARCH_X86
	bool daz;
	bool sse;
	bool sse2;
#endif
	bool sse3;
	bool ssse3;
	bool sse4_1;
	bool sse4_2;
	bool sse4a;
	bool misaligned_sse;
	bool avx;
	bool avxvnni;
	bool fma3;
	bool fma4;
	bool xop;
	bool f16c;
	bool avx2;
	bool avx512f;
	bool avx512pf;
	bool avx512er;
	bool avx512cd;
	bool avx512dq;
	bool avx512bw;
	bool avx512vl;
	bool avx512ifma;
	bool avx512vbmi;
	bool avx512vbmi2;
	bool avx512bitalg;
	bool avx512vpopcntdq;
	bool avx512vnni;
	bool avx512bf16;
	bool avx512fp16;
	bool avx512vp2intersect;
	bool avx512_4vnniw;
	bool avx512_4fmaps;
	bool amx_bf16;
	bool amx_tile;
	bool amx_int8;
	bool amx_fp16;
	bool avx_vnni_int8;
	bool avx_vnni_int16;
	bool avx_ne_convert;
	bool hle;
	bool rtm;
	bool xtest;
	bool mpx;
#if CPUINFO_ARCH_X86
	bool cmov;
	bool cmpxchg8b;
#endif
	bool cmpxchg16b;
	bool clwb;
	bool movbe;
#if CPUINFO_ARCH_X86_64
	bool lahf_sahf;
#endif
	bool fs_gs_base;
	bool lzcnt;
	bool popcnt;
	bool tbm;
	bool bmi;
	bool bmi2;
	bool adx;
	bool aes;
	bool vaes;
	bool pclmulqdq;
	bool vpclmulqdq;
	bool gfni;
	bool rdrand;
	bool rdseed;
	bool sha;
	bool rng;
	bool ace;
	bool ace2;
	bool phe;
	bool pmm;
	bool lwp;
};

extern struct cpuinfo_x86_isa cpuinfo_isa;
#endif

static inline bool cpuinfo_has_x86_rdtsc(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.rdtsc;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_rdtscp(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.rdtscp;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_rdpid(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.rdpid;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_clzero(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.clzero;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_mwait(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.mwait;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_mwaitx(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.mwaitx;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_fxsave(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.fxsave;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_xsave(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.xsave;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_fpu(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.fpu;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_mmx(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.mmx;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_mmx_plus(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.mmx_plus;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_3dnow(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.three_d_now;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_3dnow_plus(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.three_d_now_plus;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_3dnow_geode(void) {
#if CPUINFO_ARCH_X86_64
	return false;
#elif CPUINFO_ARCH_X86
#if defined(__ANDROID__)
	return false;
#else
	return cpuinfo_isa.three_d_now_geode;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_prefetch(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.prefetch;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_prefetchw(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.prefetchw;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_prefetchwt1(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.prefetchwt1;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_daz(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.daz;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_sse(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.sse;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_sse2(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.sse2;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_sse3(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.sse3;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_ssse3(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.ssse3;
#endif
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_sse4_1(void) {
#if CPUINFO_ARCH_X86_64
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.sse4_1;
#endif
#elif CPUINFO_ARCH_X86
	return cpuinfo_isa.sse4_1;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_sse4_2(void) {
#if CPUINFO_ARCH_X86_64
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.sse4_2;
#endif
#elif CPUINFO_ARCH_X86
	return cpuinfo_isa.sse4_2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_sse4a(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.sse4a;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_misaligned_sse(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.misaligned_sse;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avxvnni(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avxvnni;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_fma3(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.fma3;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_fma4(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.fma4;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_xop(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.xop;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_f16c(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.f16c;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx2(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512f(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512f;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512pf(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512pf;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512er(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512er;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512cd(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512cd;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512dq(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512dq;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512bw(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512bw;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512vl(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512vl;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512ifma(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512ifma;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512vbmi(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512vbmi;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512vbmi2(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512vbmi2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512bitalg(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512bitalg;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512vpopcntdq(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512vpopcntdq;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512vnni(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512vnni;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512bf16(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512bf16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512fp16(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512fp16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512vp2intersect(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512vp2intersect;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512_4vnniw(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512_4vnniw;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_avx512_4fmaps(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx512_4fmaps;
#else
	return false;
#endif
}

/* [NOTE] Intel Advanced Matrix Extensions (AMX) detection
 *
 * I.  AMX is a new extensions to the x86 ISA to work on matrices, consists of
 *   1) 2-dimentional registers (tiles), hold sub-matrices from larger matrices in memory
 *   2) Accelerator called Tile Matrix Multiply (TMUL), contains instructions operating on tiles
 *
 * II. Platforms that supports AMX:
 * +-----------------+-----+----------+----------+----------+----------+
 * |    Platforms    | Gen | amx-bf16 | amx-tile | amx-int8 | amx-fp16 |
 * +-----------------+-----+----------+----------+----------+----------+
 * | Sapphire Rapids | 4th |   YES    |   YES    |   YES    |    NO    |
 * +-----------------+-----+----------+----------+----------+----------+
 * | Emerald Rapids  | 5th |   YES    |   YES    |   YES    |    NO    |
 * +-----------------+-----+----------+----------+----------+----------+
 * | Granite Rapids  | 6th |   YES    |   YES    |   YES    |   YES    |
 * +-----------------+-----+----------+----------+----------+----------+
 *
 * Reference: https://www.intel.com/content/www/us/en/products/docs
 *    /accelerator-engines/advanced-matrix-extensions/overview.html
 */
static inline bool cpuinfo_has_x86_amx_bf16(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.amx_bf16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_amx_tile(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.amx_tile;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_amx_int8(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.amx_int8;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_amx_fp16(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.amx_fp16;
#else
	return false;
#endif
}

/*
 * Intel AVX Vector Neural Network Instructions (VNNI) INT8
 * Supported Platfroms: Sierra Forest, Arrow Lake, Lunar Lake
 */
static inline bool cpuinfo_has_x86_avx_vnni_int8(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx_vnni_int8;
#else
	return false;
#endif
}

/*
 * Intel AVX Vector Neural Network Instructions (VNNI) INT16
 * Supported Platfroms: Arrow Lake, Lunar Lake
 */
static inline bool cpuinfo_has_x86_avx_vnni_int16(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx_vnni_int16;
#else
	return false;
#endif
}

/*
 * A new set of instructions, which can convert low precision floating point
 * like BF16/FP16 to high precision floating point FP32, as well as convert FP32
 * elements to BF16. This instruction allows the platform to have improved AI
 * capabilities and better compatibility.
 *
 * Supported Platforms: Sierra Forest, Arrow Lake, Lunar Lake
 */
static inline bool cpuinfo_has_x86_avx_ne_convert(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.avx_ne_convert;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_hle(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.hle;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_rtm(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.rtm;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_xtest(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.xtest;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_mpx(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.mpx;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_cmov(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
	return cpuinfo_isa.cmov;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_cmpxchg8b(void) {
#if CPUINFO_ARCH_X86_64
	return true;
#elif CPUINFO_ARCH_X86
	return cpuinfo_isa.cmpxchg8b;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_cmpxchg16b(void) {
#if CPUINFO_ARCH_X86_64
	return cpuinfo_isa.cmpxchg16b;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_clwb(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.clwb;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_movbe(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.movbe;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_lahf_sahf(void) {
#if CPUINFO_ARCH_X86
	return true;
#elif CPUINFO_ARCH_X86_64
	return cpuinfo_isa.lahf_sahf;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_lzcnt(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.lzcnt;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_popcnt(void) {
#if CPUINFO_ARCH_X86_64
#if defined(__ANDROID__)
	return true;
#else
	return cpuinfo_isa.popcnt;
#endif
#elif CPUINFO_ARCH_X86
	return cpuinfo_isa.popcnt;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_tbm(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.tbm;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_bmi(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.bmi;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_bmi2(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.bmi2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_adx(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.adx;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_aes(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.aes;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_vaes(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.vaes;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_pclmulqdq(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.pclmulqdq;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_vpclmulqdq(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.vpclmulqdq;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_gfni(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.gfni;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_rdrand(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.rdrand;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_rdseed(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.rdseed;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_x86_sha(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	return cpuinfo_isa.sha;
#else
	return false;
#endif
}

#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
/* This structure is not a part of stable API. Use cpuinfo_has_arm_* functions
 * instead. */
struct cpuinfo_arm_isa {
#if CPUINFO_ARCH_ARM
	bool thumb;
	bool thumb2;
	bool thumbee;
	bool jazelle;
	bool armv5e;
	bool armv6;
	bool armv6k;
	bool armv7;
	bool armv7mp;
	bool armv8;
	bool idiv;

	bool vfpv2;
	bool vfpv3;
	bool d32;
	bool fp16;
	bool fma;

	bool wmmx;
	bool wmmx2;
	bool neon;
#endif
#if CPUINFO_ARCH_ARM64
	bool atomics;
	bool bf16;
	bool sve;
	bool sve2;
	bool i8mm;
	bool sme;
	bool sme2;
	bool sme2p1;
	bool sme_i16i32;
	bool sme_bi32i32;
	bool sme_b16b16;
	bool sme_f16f16;
	uint32_t svelen;
#endif
	bool rdm;
	bool fp16arith;
	bool dot;
	bool jscvt;
	bool fcma;
	bool fhm;

	bool aes;
	bool sha1;
	bool sha2;
	bool pmull;
	bool crc32;
};

extern struct cpuinfo_arm_isa cpuinfo_isa;
#endif

static inline bool cpuinfo_has_arm_thumb(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.thumb;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_thumb2(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.thumb2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_v5e(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.armv5e;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_v6(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.armv6;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_v6k(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.armv6k;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_v7(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.armv7;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_v7mp(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.armv7mp;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_v8(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.armv8;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_idiv(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.idiv;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_vfpv2(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.vfpv2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_vfpv3(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.vfpv3;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_vfpv3_d32(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.vfpv3 && cpuinfo_isa.d32;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_vfpv3_fp16(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.vfpv3 && cpuinfo_isa.fp16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_vfpv3_fp16_d32(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.vfpv3 && cpuinfo_isa.fp16 && cpuinfo_isa.d32;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_vfpv4(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.vfpv3 && cpuinfo_isa.fma;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_vfpv4_d32(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.vfpv3 && cpuinfo_isa.fma && cpuinfo_isa.d32;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_fp16_arith(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.fp16arith;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_bf16(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.bf16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_wmmx(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.wmmx;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_wmmx2(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.wmmx2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_neon(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.neon;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_neon_fp16(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.neon && cpuinfo_isa.fp16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_neon_fma(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.neon && cpuinfo_isa.fma;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_neon_v8(void) {
#if CPUINFO_ARCH_ARM64
	return true;
#elif CPUINFO_ARCH_ARM
	return cpuinfo_isa.neon && cpuinfo_isa.armv8;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_atomics(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.atomics;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_neon_rdm(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.rdm;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_neon_fp16_arith(void) {
#if CPUINFO_ARCH_ARM
	return cpuinfo_isa.neon && cpuinfo_isa.fp16arith;
#elif CPUINFO_ARCH_ARM64
	return cpuinfo_isa.fp16arith;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_fhm(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.fhm;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_neon_dot(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.dot;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_neon_bf16(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.bf16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_jscvt(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.jscvt;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_fcma(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.fcma;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_i8mm(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.i8mm;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_aes(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.aes;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sha1(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sha1;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sha2(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sha2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_pmull(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.pmull;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_crc32(void) {
#if CPUINFO_ARCH_ARM || CPUINFO_ARCH_ARM64
	return cpuinfo_isa.crc32;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sve(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sve;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sve_bf16(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sve && cpuinfo_isa.bf16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sve2(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sve2;
#else
	return false;
#endif
}

// Function to get the max SVE vector length on ARM CPU's which support SVE.
static inline uint32_t cpuinfo_get_max_arm_sve_length(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.svelen * 8; // bytes * 8 = bit length(vector length)
#else
	return 0;
#endif
}

static inline bool cpuinfo_has_arm_sme(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sme;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sme2(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sme2;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sme2p1(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sme2p1;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sme_i16i32(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sme_i16i32;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sme_bi32i32(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sme_bi32i32;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sme_b16b16(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sme_b16b16;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_arm_sme_f16f16(void) {
#if CPUINFO_ARCH_ARM64
	return cpuinfo_isa.sme_f16f16;
#else
	return false;
#endif
}

#if CPUINFO_ARCH_RISCV32 || CPUINFO_ARCH_RISCV64
/* This structure is not a part of stable API. Use cpuinfo_has_riscv_* functions
 * instead. */
struct cpuinfo_riscv_isa {
	/**
	 * Keep fields in line with the canonical order as defined by
	 * Section 27.11 Subset Naming Convention.
	 */
	/* RV32I/64I/128I Base ISA. */
	bool i;
#if CPUINFO_ARCH_RISCV32
	/* RV32E Base ISA. */
	bool e;
#endif
	/* Integer Multiply/Divide Extension. */
	bool m;
	/* Atomic Extension. */
	bool a;
	/* Single-Precision Floating-Point Extension. */
	bool f;
	/* Double-Precision Floating-Point Extension. */
	bool d;
	/* Compressed Extension. */
	bool c;
	/* Vector Extension. */
	bool v;
};

extern struct cpuinfo_riscv_isa cpuinfo_isa;
#endif

static inline bool cpuinfo_has_riscv_i(void) {
#if CPUINFO_ARCH_RISCV32 || CPUINFO_ARCH_RISCV64
	return cpuinfo_isa.i;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_riscv_e(void) {
#if CPUINFO_ARCH_RISCV32
	return cpuinfo_isa.e;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_riscv_m(void) {
#if CPUINFO_ARCH_RISCV32 || CPUINFO_ARCH_RISCV64
	return cpuinfo_isa.m;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_riscv_a(void) {
#if CPUINFO_ARCH_RISCV32 || CPUINFO_ARCH_RISCV64
	return cpuinfo_isa.a;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_riscv_f(void) {
#if CPUINFO_ARCH_RISCV32 || CPUINFO_ARCH_RISCV64
	return cpuinfo_isa.f;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_riscv_d(void) {
#if CPUINFO_ARCH_RISCV32 || CPUINFO_ARCH_RISCV64
	return cpuinfo_isa.d;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_riscv_g(void) {
	// The 'G' extension is simply shorthand for 'IMAFD'.
	return cpuinfo_has_riscv_i() && cpuinfo_has_riscv_m() && cpuinfo_has_riscv_a() && cpuinfo_has_riscv_f() &&
		cpuinfo_has_riscv_d();
}

static inline bool cpuinfo_has_riscv_c(void) {
#if CPUINFO_ARCH_RISCV32 || CPUINFO_ARCH_RISCV64
	return cpuinfo_isa.c;
#else
	return false;
#endif
}

static inline bool cpuinfo_has_riscv_v(void) {
#if CPUINFO_ARCH_RISCV32 || CPUINFO_ARCH_RISCV64
	return cpuinfo_isa.v;
#else
	return false;
#endif
}

const struct cpuinfo_processor* CPUINFO_ABI cpuinfo_get_processors(void);
const struct cpuinfo_core* CPUINFO_ABI cpuinfo_get_cores(void);
const struct cpuinfo_cluster* CPUINFO_ABI cpuinfo_get_clusters(void);
const struct cpuinfo_package* CPUINFO_ABI cpuinfo_get_packages(void);
const struct cpuinfo_uarch_info* CPUINFO_ABI cpuinfo_get_uarchs(void);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l1i_caches(void);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l1d_caches(void);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l2_caches(void);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l3_caches(void);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l4_caches(void);

const struct cpuinfo_processor* CPUINFO_ABI cpuinfo_get_processor(uint32_t index);
const struct cpuinfo_core* CPUINFO_ABI cpuinfo_get_core(uint32_t index);
const struct cpuinfo_cluster* CPUINFO_ABI cpuinfo_get_cluster(uint32_t index);
const struct cpuinfo_package* CPUINFO_ABI cpuinfo_get_package(uint32_t index);
const struct cpuinfo_uarch_info* CPUINFO_ABI cpuinfo_get_uarch(uint32_t index);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l1i_cache(uint32_t index);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l1d_cache(uint32_t index);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l2_cache(uint32_t index);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l3_cache(uint32_t index);
const struct cpuinfo_cache* CPUINFO_ABI cpuinfo_get_l4_cache(uint32_t index);

uint32_t CPUINFO_ABI cpuinfo_get_processors_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_cores_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_clusters_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_packages_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_uarchs_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_l1i_caches_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_l1d_caches_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_l2_caches_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_l3_caches_count(void);
uint32_t CPUINFO_ABI cpuinfo_get_l4_caches_count(void);

/**
 * Returns upper bound on cache size.
 */
uint32_t CPUINFO_ABI cpuinfo_get_max_cache_size(void);

/**
 * Identify the logical processor that executes the current thread.
 *
 * There is no guarantee that the thread will stay on the same logical processor
 * for any time. Callers should treat the result as only a hint, and be prepared
 * to handle NULL return value.
 */
const struct cpuinfo_processor* CPUINFO_ABI cpuinfo_get_current_processor(void);

/**
 * Identify the core that executes the current thread.
 *
 * There is no guarantee that the thread will stay on the same core for any
 * time. Callers should treat the result as only a hint, and be prepared to
 * handle NULL return value.
 */
const struct cpuinfo_core* CPUINFO_ABI cpuinfo_get_current_core(void);

/**
 * Identify the microarchitecture index of the core that executes the current
 * thread. If the system does not support such identification, the function
 * returns 0.
 *
 * There is no guarantee that the thread will stay on the same type of core for
 * any time. Callers should treat the result as only a hint.
 */
uint32_t CPUINFO_ABI cpuinfo_get_current_uarch_index(void);

/**
 * Identify the microarchitecture index of the core that executes the current
 * thread. If the system does not support such identification, the function
 * returns the user-specified default value.
 *
 * There is no guarantee that the thread will stay on the same type of core for
 * any time. Callers should treat the result as only a hint.
 */
uint32_t CPUINFO_ABI cpuinfo_get_current_uarch_index_with_default(uint32_t default_uarch_index);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CPUINFO_H */
