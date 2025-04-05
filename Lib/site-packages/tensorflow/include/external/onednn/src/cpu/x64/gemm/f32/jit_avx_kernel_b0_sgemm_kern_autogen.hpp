/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef CPU_X64_GEMM_F32_JIT_AVX_KERNEL_B0_SGEMM_KERN_AUTOGEN_HPP
#define CPU_X64_GEMM_F32_JIT_AVX_KERNEL_B0_SGEMM_KERN_AUTOGEN_HPP

#ifndef _WIN32

#define M rdi
#define N rsi
#define K rdx
#define A r8
#define B r9
#define C rcx
#define LDC r10

#define AA r15
#define I r11
#define J r12
#define H rax
#define AO rbx
#define BO rbp
#define CO1 r13
#define CO2 r14

#define OLD_C (8 + stacksize + rsp)
#define OLD_LDC (16 + stacksize + rsp)

#else

#define M rcx
#define N rdx
#define K r8
#define A rdi
#define B rsi
#define C r9
#define LDC r10
#define AA r15
#define I r11
#define J r12
#define H rax
#define AO rbx
#define BO rbp
#define CO1 r13
#define CO2 r14

#define OLD_A 40 + stacksize + rsp
#define OLD_B 48 + stacksize + rsp
#define OLD_C 56 + stacksize + rsp
#define OLD_LDC 64 + stacksize + rsp

#endif

#endif
