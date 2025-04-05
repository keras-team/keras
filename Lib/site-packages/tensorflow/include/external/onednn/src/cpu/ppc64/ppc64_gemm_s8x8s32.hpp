/*******************************************************************************
* Copyright 2022 IBM Corporation
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

#include <altivec.h>
#include "cpu/simple_q10n.hpp"

namespace dnnl {
namespace impl {

uint64_t mker;

typedef __vector signed long long vec_i64 __attribute__((aligned(8)));
typedef __vector short vec_i16 __attribute__((aligned(2)));
typedef __vector unsigned char vec_t;
typedef __vector signed char vec_st;

int pack_N16_16bit(dim_t k, dim_t m, short *a, dim_t lda, short *ap) {
    int32_t i, j;
    int32_t kcell, cell, koff, moff, krows, mrows, block4, block2, mcell,
            chunk4count, k8, m4, m16;
    int32_t m_cap = (m + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    krows = (k + 1) >> 1;
    mrows = (m + 3) >> 2;
    block4 = 4 * krows;
    block2 = 2 * krows;
    k8 = (k >> 3) << 3;
    m4 = (m >> 2) << 2;
    m16 = (m >> 4) << 4;

    // MAIN BLOCK
    for (j = 0; j < m16; j += 4) {
        for (i = 0; i < k8; i += 8) {
            kcell = i >> 1; // 0, 1, 2, 3
            mcell = j >> 2;
            short *dest = &ap[32 * ((mcell >> 2) * krows + kcell)
                    + 8 * (mcell & 3)];

            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
            vec_t swizB = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29,
                    30, 31};
            vec_t swizL
                    = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            vec_t swizR = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29,
                    30, 31};

            V0 = *(vec_t *)&a[lda * (j + 0) + i];
            V1 = *(vec_t *)&a[lda * (j + 1) + i];
            V2 = *(vec_t *)&a[lda * (j + 2) + i];
            V3 = *(vec_t *)&a[lda * (j + 3) + i];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[32] = D1;
            *(vec_t *)&dest[64] = D2;
            *(vec_t *)&dest[96] = D3;
        }
    }

    for (j = m16; j < m4; ++j) {
        for (i = 0; i < k8; ++i) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            ap[8 * cell + 2 * moff + koff] = a[lda * j + i];
        }
    }

    // HIGH EDGE IN M DIRECTION
    for (j = m4; j < m_cap; ++j) {
        for (i = 0; i < k8; ++i) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (j < m)
                ap[8 * cell + 2 * moff + koff] = a[lda * j + i];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (j = 0; j < m4; ++j) {
        for (i = k8; i < k_cap; ++i) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (i < k)
                ap[8 * cell + 2 * moff + koff] = a[lda * j + i];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH M, HIGH K)
    for (j = m4; j < m_cap; ++j) {
        for (i = k8; i < k_cap; ++i) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (j < m && i < k)
                ap[8 * cell + 2 * moff + koff] = a[lda * j + i];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }
    return 0;
}

int pack_T16_16bit(dim_t k, dim_t m, short *a, dim_t lda, short *ap) {
    int32_t i, j;
    int32_t kcell, cell, koff, moff, krows, mrows, block4, block2, mcell,
            chunk4count, k4, m8, m16;
    int32_t m_cap = (m + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    krows = (k + 1) >> 1;
    mrows = (m + 3) >> 2;
    block4 = 4 * krows;
    block2 = 2 * krows;
    k4 = (k >> 2) << 2;
    m16 = (m >> 4) << 4;
    m8 = (m >> 3) << 3;

    // MAIN BLOCK
    for (i = 0; i < k4; i += 4) {
        for (j = 0; j < m16; j += 16) {
            short *src = &a[lda * i + j];
            short *dst = &ap[2 * j * krows + 16 * i];
            vec_t V0, V1, V2, V3, V4, V5, V6, V7;
            vec_t D0, D1, D2, D3, D4, D5, D6, D7;
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};

            V0 = *(vec_t *)&src[0];
            V1 = *(vec_t *)&src[lda];
            V2 = *(vec_t *)&src[8];
            V3 = *(vec_t *)&src[lda + 8];
            V4 = *(vec_t *)&src[2 * lda];
            V5 = *(vec_t *)&src[3 * lda];
            V6 = *(vec_t *)&src[2 * lda + 8];
            V7 = *(vec_t *)&src[3 * lda + 8];
            D0 = vec_perm(V0, V1, swizL);
            D1 = vec_perm(V0, V1, swizR);
            D2 = vec_perm(V2, V3, swizL);
            D3 = vec_perm(V2, V3, swizR);
            D4 = vec_perm(V4, V5, swizL);
            D5 = vec_perm(V4, V5, swizR);
            D6 = vec_perm(V6, V7, swizL);
            D7 = vec_perm(V6, V7, swizR);

            *(vec_t *)&dst[0] = D0;
            *(vec_t *)&dst[8] = D1;
            *(vec_t *)&dst[16] = D2;
            *(vec_t *)&dst[24] = D3;
            *(vec_t *)&dst[32] = D4;
            *(vec_t *)&dst[40] = D5;
            *(vec_t *)&dst[48] = D6;
            *(vec_t *)&dst[56] = D7;
        }
    }

    for (i = 0; i < k4; ++i) {
        for (j = m16; j < m8; ++j) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            ap[8 * cell + 2 * moff + koff] = a[lda * i + j];
        }
    }

    // HIGH EDGE IN M DIRECTION
    for (i = 0; i < k4; ++i) {
        for (j = m8; j < m_cap; ++j) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (j < m)
                ap[8 * cell + 2 * moff + koff] = a[lda * i + j];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k4; i < k_cap; ++i) {
        for (j = 0; j < m8; ++j) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (i < k)
                ap[8 * cell + 2 * moff + koff] = a[lda * i + j];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH M, HIGH K)
    for (i = k4; i < k_cap; ++i) {
        for (j = m8; j < m_cap; ++j) {
            kcell = i >> 1;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 1;
            moff = j & 3;
            if (i < k && j < m)
                ap[8 * cell + 2 * moff + koff] = a[lda * i + j];
            else
                ap[8 * cell + 2 * moff + koff] = 0;
        }
    }
    return 0;
}

int pack_T8_16bit(dim_t k, dim_t n, short *b, dim_t ldb, short *bp) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k4, n8, n16;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    krows = (k + 1) >> 1;
    k4 = (k >> 2) << 2;
    n8 = (n >> 3) << 3;
    n16 = (n >> 4) << 4;

    // MAIN BLOCK
    for (i = 0; i < k4; i += 4) {
        for (j = 0; j < n16; j += 16) {
            short *src = &b[ldb * i + j];
            short *dst0145 = &bp[2 * j * krows + 8 * i];
            short *dst2367 = &bp[2 * (j + 8) * krows + 8 * i];
            vec_t V0, V1, V2, V3, V4, V5, V6, V7;
            vec_t D0, D1, D2, D3, D4, D5, D6, D7;
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};

            V0 = *(vec_t *)&src[0];
            V1 = *(vec_t *)&src[ldb];
            V2 = *(vec_t *)&src[8];
            V3 = *(vec_t *)&src[ldb + 8];
            V4 = *(vec_t *)&src[2 * ldb];
            V5 = *(vec_t *)&src[3 * ldb];
            V6 = *(vec_t *)&src[2 * ldb + 8];
            V7 = *(vec_t *)&src[3 * ldb + 8];
            D0 = vec_perm(V0, V1, swizL);
            D1 = vec_perm(V0, V1, swizR);
            D2 = vec_perm(V2, V3, swizL);
            D3 = vec_perm(V2, V3, swizR);
            D4 = vec_perm(V4, V5, swizL);
            D5 = vec_perm(V4, V5, swizR);
            D6 = vec_perm(V6, V7, swizL);
            D7 = vec_perm(V6, V7, swizR);

            *(vec_t *)&dst0145[0] = D0;
            *(vec_t *)&dst0145[8] = D1;
            *(vec_t *)&dst2367[0] = D2;
            *(vec_t *)&dst2367[8] = D3;
            *(vec_t *)&dst0145[16] = D4;
            *(vec_t *)&dst0145[24] = D5;
            *(vec_t *)&dst2367[16] = D6;
            *(vec_t *)&dst2367[24] = D7;
        }
        for (j = n16; j < n8; j += 8) {
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            short *dst = &bp[8 * (columns_done * krows + (i & (~1)))];
            vec_t V0, V1, V2, V3;
            vec_t D0, D1, D2, D3;
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};

            V0 = *(vec_t *)&b[ldb * (i + 0) + j];
            V1 = *(vec_t *)&b[ldb * (i + 1) + j];
            V2 = *(vec_t *)&b[ldb * (i + 2) + j];
            V3 = *(vec_t *)&b[ldb * (i + 3) + j];
            D0 = vec_perm(V0, V1, swizL);
            D1 = vec_perm(V0, V1, swizR);
            D2 = vec_perm(V2, V3, swizL);
            D3 = vec_perm(V2, V3, swizR);

            *(vec_t *)&dst[0] = D0;
            *(vec_t *)&dst[8] = D1;
            *(vec_t *)&dst[16] = D2;
            *(vec_t *)&dst[24] = D3;
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (i = 0; i < k4; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (j < n)
                bp[8 * cell + 2 * noff + koff] = b[ldb * i + j];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k4; i < k_cap; ++i) {
        for (j = 0; j < n8; ++j) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (i < k)
                bp[8 * cell + 2 * noff + koff] = b[ldb * i + j];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (i = k4; i < k_cap; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (i < k && j < n)
                bp[8 * cell + 2 * noff + koff] = b[ldb * i + j];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }
    return 0;
}

int pack_N8_16bit(dim_t k, dim_t n, short *b, dim_t ldb, short *bp) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k8, k16, n4, n8;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    krows = (k + 1) >> 1;
    k8 = (k >> 3) << 3;
    k16 = (k >> 4) << 4;
    n4 = (n >> 2) << 2;
    n8 = (n >> 3) << 3;

    // MAIN BLOCK
    for (j = 0; j < n8; j += 4) {
        for (i = 0; i < k16; i += 16) {
            kcell = i >> 1; // 0, 1, 2, 3
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t j_hiflag = (j & 4) >> 2;
            koff = i & 1;
            noff = j & 3;
            short *dst = &bp[8 * (columns_done * krows + kcell * 2 + j_hiflag)];

            vec_t V0, V1, V2, V3, V4, V5, V6, V7;
            vec_t D01A, D01B, D23A, D23B, D45A, D45B, D67A, D67B;
            vec_t D0, D1, D2, D3, D4, D5, D6, D7;
            vec_t swizA
                    = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
            vec_t swizB = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29,
                    30, 31};
            vec_t swizL
                    = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            vec_t swizR = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29,
                    30, 31};

            V0 = *(vec_t *)&b[ldb * (j + 0) + i];
            V1 = *(vec_t *)&b[ldb * (j + 1) + i];
            V2 = *(vec_t *)&b[ldb * (j + 2) + i];
            V3 = *(vec_t *)&b[ldb * (j + 3) + i];
            V4 = *(vec_t *)&b[ldb * (j + 0) + i + 8];
            V5 = *(vec_t *)&b[ldb * (j + 1) + i + 8];
            V6 = *(vec_t *)&b[ldb * (j + 2) + i + 8];
            V7 = *(vec_t *)&b[ldb * (j + 3) + i + 8];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D45A = vec_perm(V4, V5, swizA);
            D45B = vec_perm(V4, V5, swizB);
            D67A = vec_perm(V6, V7, swizA);
            D67B = vec_perm(V6, V7, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            D4 = vec_perm(D45A, D67A, swizL);
            D5 = vec_perm(D45A, D67A, swizR);
            D6 = vec_perm(D45B, D67B, swizL);
            D7 = vec_perm(D45B, D67B, swizR);

            *(vec_t *)&dst[0] = D0;
            *(vec_t *)&dst[16] = D1;
            *(vec_t *)&dst[32] = D2;
            *(vec_t *)&dst[48] = D3;
            *(vec_t *)&dst[64] = D4;
            *(vec_t *)&dst[80] = D5;
            *(vec_t *)&dst[96] = D6;
            *(vec_t *)&dst[112] = D7;
        }
        for (i = k16; i < k8; i += 8) {
            kcell = i >> 1; // 0, 1, 2, 3
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t j_hiflag = (j & 4) >> 2;
            koff = i & 1;
            noff = j & 3;
            short *dst = &bp[8 * (columns_done * krows + kcell * 2 + j_hiflag)];

            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
            vec_t swizB = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29,
                    30, 31};
            vec_t swizL
                    = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            vec_t swizR = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29,
                    30, 31};

            V0 = *(vec_t *)&b[ldb * (j + 0) + i];
            V1 = *(vec_t *)&b[ldb * (j + 1) + i];
            V2 = *(vec_t *)&b[ldb * (j + 2) + i];
            V3 = *(vec_t *)&b[ldb * (j + 3) + i];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            *(vec_t *)&dst[0] = D0;
            *(vec_t *)&dst[16] = D1;
            *(vec_t *)&dst[32] = D2;
            *(vec_t *)&dst[48] = D3;
        }
    }

    for (j = n8; j < n4; ++j) {
        for (i = 0; i < k8; ++i) {
            kcell = i >> 1;
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            bp[8 * cell + 2 * noff + koff] = b[ldb * j + i];
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (j = n4; j < n_cap; ++j) {
        for (i = 0; i < k8; ++i) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (j < n)
                bp[8 * cell + 2 * noff + koff] = b[ldb * j + i];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (j = 0; j < n4; ++j) {
        for (i = k8; i < k_cap; ++i) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (i < k)
                bp[8 * cell + 2 * noff + koff] = b[ldb * j + i];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (j = n4; j < n_cap; ++j) {
        for (i = k8; i < k_cap; ++i) {
            kcell = i >> 1;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 1;
            noff = j & 3;
            if (j < n && i < k)
                bp[8 * cell + 2 * noff + koff] = b[ldb * j + i];
            else
                bp[8 * cell + 2 * noff + koff] = 0;
        }
    }
    return 0;
}

int pack_T16_8bit(dim_t k, dim_t m, const int8_t *a, dim_t lda, int8_t *ap) {
    int32_t i, j;
    int32_t m_cap = (m + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    int32_t kcell, cell, koff, moff, krows, mrows, block4, block2, mcell,
            chunk4count, m16, k4;
    m16 = (m >> 4) << 4;
    k4 = (k >> 2) << 2;
    krows = (k + 3) >> 2;
    mrows = (m + 3) >> 2;
    block4 = 4 * krows;
    block2 = 2 * krows;

    // MAIN BLOCK
    for (i = 0; i < k4; i += 4) {
        for (j = 0; j < m16; j += 16) {
            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
            vec_t swizB = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                    15, 31};
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};
            int8_t *dest;

            V0 = *(vec_t *)&a[lda * (i + 0) + j];
            V1 = *(vec_t *)&a[lda * (i + 1) + j];
            V2 = *(vec_t *)&a[lda * (i + 2) + j];
            V3 = *(vec_t *)&a[lda * (i + 3) + j];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            dest = &ap[16 * (((j >> 4) * block4) + i)];

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[16] = D1;
            *(vec_t *)&dest[32] = D2;
            *(vec_t *)&dest[48] = D3;
        }
    }

    // HIGH EDGE IN M DIRECTION
    for (i = 0; i < k4; ++i) {
        for (j = m16; j < m_cap; ++j) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (j < m)
                ap[16 * cell + 4 * moff + koff] = a[lda * i + j];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k4; i < k_cap; ++i) {
        for (j = 0; j < m16; ++j) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (i < k)
                ap[16 * cell + 4 * moff + koff] = a[lda * i + j];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH M, HIGH K)
    for (i = k4; i < k_cap; ++i) {
        for (j = m16; j < m_cap; ++j) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;
            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = chunk4count * block4;
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (i < k && j < m)
                ap[16 * cell + 4 * moff + koff] = a[lda * i + j];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    return 0;
}

int pack_N8_8bit(dim_t k, dim_t n, const uint8_t *b, dim_t ldb, uint8_t *bp) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k8, n8;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    krows = (k + 3) >> 2;
    k8 = k >> 3;
    n8 = n >> 3;

    // MAIN BLOCK
    for (j = 0; j < (n8 << 3); j += 8) {
        for (i = 0; i < (k8 << 3); i += 8) {
            vec_t V0, V1, V2, V3;
            vec_t D0, D1, D2, D3;
            vec_t swizA = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            vec_t swizB = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};
            const uint8_t *src = &b[ldb * j + i];
            uint8_t *dest = &bp[16 * (krows * (j >> 2) + (i >> 1))];

            *(signed long long *)&V0[0] = *(signed long long *)&src[ldb * 0];
            *(signed long long *)&V0[8] = *(signed long long *)&src[ldb * 1];
            *(signed long long *)&V1[0] = *(signed long long *)&src[ldb * 2];
            *(signed long long *)&V1[8] = *(signed long long *)&src[ldb * 3];
            *(signed long long *)&V2[0] = *(signed long long *)&src[ldb * 4];
            *(signed long long *)&V2[8] = *(signed long long *)&src[ldb * 5];
            *(signed long long *)&V3[0] = *(signed long long *)&src[ldb * 6];
            *(signed long long *)&V3[8] = *(signed long long *)&src[ldb * 7];

            D0 = vec_perm(V0, V1, swizA);
            D1 = vec_perm(V2, V3, swizA);
            D2 = vec_perm(V0, V1, swizB);
            D3 = vec_perm(V2, V3, swizB);

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[16] = D1;
            *(vec_t *)&dest[32] = D2;
            *(vec_t *)&dest[48] = D3;
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (j = (n8 << 3); j < n_cap; ++j) {
        for (i = 0; i < (k8 << 3); ++i) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (j < n)
                bp[16 * cell + 4 * noff + koff] = b[ldb * j + i];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (j = 0; j < (n8 << 3); ++j) {
        for (i = (k8 << 3); i < k_cap; ++i) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (i < k)
                bp[16 * cell + 4 * noff + koff] = b[ldb * j + i];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (j = (n8 << 3); j < n_cap; ++j) {
        for (i = (k8 << 3); i < k_cap; ++i) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (j < n && i < k)
                bp[16 * cell + 4 * noff + koff] = b[ldb * j + i];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    return 0;
}

int pack_N16_8bit(dim_t k, dim_t m, const int8_t *a, dim_t lda, int8_t *ap) {
    int32_t i, j;
    int32_t m_cap = (m + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    int32_t kcell, cell, koff, moff, krows, mrows, block4, block2, mcell,
            chunk4count, m16, k4, k16;
    m16 = (m >> 4) << 4;
    k4 = (k >> 2) << 2;
    k16 = (k >> 4) << 4;
    krows = (k + 3) >> 2;
    mrows = (m + 3) >> 2;
    block4 = 4 * krows;
    block2 = 2 * krows;

    // MAIN BLOCK
    for (j = 0; j < m16; j += 16) {
        for (i = 0; i < k16; i += 16) {
            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
            vec_t swizB = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29,
                    30, 31};
            vec_t swizL = {
                    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27};
            vec_t swizR = {
                    4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31};

            const int8_t *src = &a[lda * j + i];
            int8_t *dest = &ap[j * (krows << 2) + (i << 4)];

            V0 = *(vec_t *)&src[0 * lda];
            V1 = *(vec_t *)&src[1 * lda];
            V2 = *(vec_t *)&src[2 * lda];
            V3 = *(vec_t *)&src[3 * lda];
            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[64] = D1;
            *(vec_t *)&dest[128] = D2;
            *(vec_t *)&dest[192] = D3;

            V0 = *(vec_t *)&src[4 * lda];
            V1 = *(vec_t *)&src[5 * lda];
            V2 = *(vec_t *)&src[6 * lda];
            V3 = *(vec_t *)&src[7 * lda];
            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            *(vec_t *)&dest[16] = D0;
            *(vec_t *)&dest[80] = D1;
            *(vec_t *)&dest[144] = D2;
            *(vec_t *)&dest[208] = D3;

            V0 = *(vec_t *)&src[8 * lda];
            V1 = *(vec_t *)&src[9 * lda];
            V2 = *(vec_t *)&src[10 * lda];
            V3 = *(vec_t *)&src[11 * lda];
            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            *(vec_t *)&dest[32] = D0;
            *(vec_t *)&dest[96] = D1;
            *(vec_t *)&dest[160] = D2;
            *(vec_t *)&dest[224] = D3;

            V0 = *(vec_t *)&src[12 * lda];
            V1 = *(vec_t *)&src[13 * lda];
            V2 = *(vec_t *)&src[14 * lda];
            V3 = *(vec_t *)&src[15 * lda];
            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);
            *(vec_t *)&dest[48] = D0;
            *(vec_t *)&dest[112] = D1;
            *(vec_t *)&dest[176] = D2;
            *(vec_t *)&dest[240] = D3;
        }
        for (i = k16; i < k4; i += 4) {
            vec_t D0, D1, D2, D3;
            const int8_t *src = &a[lda * j + i];
            int8_t *dest = &ap[j * (krows << 2) + (i << 4)];

            *(int *)&D0[0] = *(int *)&src[0 * lda];
            *(int *)&D0[4] = *(int *)&src[1 * lda];
            *(int *)&D0[8] = *(int *)&src[2 * lda];
            *(int *)&D0[12] = *(int *)&src[3 * lda];
            *(int *)&D1[0] = *(int *)&src[4 * lda];
            *(int *)&D1[4] = *(int *)&src[5 * lda];
            *(int *)&D1[8] = *(int *)&src[6 * lda];
            *(int *)&D1[12] = *(int *)&src[7 * lda];
            *(int *)&D2[0] = *(int *)&src[8 * lda];
            *(int *)&D2[4] = *(int *)&src[9 * lda];
            *(int *)&D2[8] = *(int *)&src[10 * lda];
            *(int *)&D2[12] = *(int *)&src[11 * lda];
            *(int *)&D3[0] = *(int *)&src[12 * lda];
            *(int *)&D3[4] = *(int *)&src[13 * lda];
            *(int *)&D3[8] = *(int *)&src[14 * lda];
            *(int *)&D3[12] = *(int *)&src[15 * lda];

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[16] = D1;
            *(vec_t *)&dest[32] = D2;
            *(vec_t *)&dest[48] = D3;
        }
    }

    // HIGH EDGE IN M DIRECTION
    for (j = m16; j < m_cap; ++j) {
        for (i = 0; i < k4; ++i) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;

            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (j < m)
                ap[16 * cell + 4 * moff + koff] = a[lda * j + i];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (j = 0; j < m16; ++j) {
        for (i = k4; i < k_cap; ++i) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;

            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (i < k)
                ap[16 * cell + 4 * moff + koff] = a[lda * j + i];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH M, HIGH K)
    for (j = m16; j < m_cap; ++j) {
        for (i = k4; i < k_cap; ++i) {
            kcell = i >> 2;
            mcell = j >> 2;
            chunk4count = mcell >> 2;

            if (mcell < (mrows & ~3))
                cell = (chunk4count * block4) + (4 * kcell) + (mcell & 3);
            else {
                cell = (chunk4count * block4);
                if (m_cap & 8) {
                    switch (mcell & 3) {
                        case 0:
                        case 1: cell += 2 * kcell + (mcell & 1); break;
                        case 2: cell += block2 + kcell; break;
                    }
                } else if (m_cap & 4)
                    cell += kcell;
            }
            koff = i & 3;
            moff = j & 3;
            if (j < m && i < k)
                ap[16 * cell + 4 * moff + koff] = a[lda * j + i];
            else
                ap[16 * cell + 4 * moff + koff] = 0;
        }
    }

    return 0;
}

int pack_T8_8bit(dim_t k, dim_t n, const uint8_t *b, dim_t ldb, uint8_t *bp) {
    int32_t i, j;
    int32_t kcell, cell, koff, noff, krows, k8, n8;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    krows = (k + 3) >> 2;
    k8 = (k >> 3) << 3;
    n8 = (n >> 3) << 3;

    // MAIN BLOCK
    for (i = 0; i < k8; i += 8) {
        for (j = 0; j < n8; j += 8) {
            vec_t V0, V1, V2, V3;
            vec_t D01A, D01B, D23A, D23B;
            vec_t D0, D1, D2, D3;
            vec_t swizA
                    = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23};
            vec_t swizB = {8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30,
                    15, 31};
            vec_t swizL
                    = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
            vec_t swizR = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15,
                    30, 31};
            uint8_t *dest;

            *(signed long long *)&V0[0]
                    = *(signed long long *)&b[ldb * (i + 0) + j];
            *(signed long long *)&V1[0]
                    = *(signed long long *)&b[ldb * (i + 1) + j];
            *(signed long long *)&V2[0]
                    = *(signed long long *)&b[ldb * (i + 2) + j];
            *(signed long long *)&V3[0]
                    = *(signed long long *)&b[ldb * (i + 3) + j];
            *(signed long long *)&V0[8]
                    = *(signed long long *)&b[ldb * (i + 4) + j];
            *(signed long long *)&V1[8]
                    = *(signed long long *)&b[ldb * (i + 5) + j];
            *(signed long long *)&V2[8]
                    = *(signed long long *)&b[ldb * (i + 6) + j];
            *(signed long long *)&V3[8]
                    = *(signed long long *)&b[ldb * (i + 7) + j];

            D01A = vec_perm(V0, V1, swizA);
            D01B = vec_perm(V0, V1, swizB);
            D23A = vec_perm(V2, V3, swizA);
            D23B = vec_perm(V2, V3, swizB);
            D0 = vec_perm(D01A, D23A, swizL);
            D1 = vec_perm(D01A, D23A, swizR);
            D2 = vec_perm(D01B, D23B, swizL);
            D3 = vec_perm(D01B, D23B, swizR);

            dest = &bp[16 * ((j >> 2) * krows + (i >> 1))];

            *(vec_t *)&dest[0] = D0;
            *(vec_t *)&dest[16] = D1;
            *(vec_t *)&dest[32] = D2;
            *(vec_t *)&dest[48] = D3;
        }
    }

    // HIGH EDGE IN N DIRECTION
    for (i = 0; i < k8; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (j < n)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // HIGH EDGE IN K DIRECTION
    for (i = k8; i < k_cap; ++i) {
        for (j = 0; j < n8; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (i < k)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    // UPPER CORNER (HIGH N, HIGH K)
    for (i = k8; i < k_cap; ++i) {
        for (j = n8; j < n_cap; ++j) {
            kcell = i >> 2;
            // special handling if j is in a PARTIAL last "group of 8"
            int32_t maingroup = (j & (~7)) < (n & (~7));
            int32_t columns_done = ((j & (~7)) >> 3) << 1;
            int32_t groupwidth = (maingroup || ((n & 7) > 4)) ? 2 : 1;
            int32_t j_hiflag = (j & 4) >> 2;
            cell = columns_done * krows + kcell * groupwidth + j_hiflag;
            koff = i & 3;
            noff = j & 3;
            if (i < k && j < n)
                bp[16 * cell + 4 * noff + koff] = b[ldb * i + j];
            else
                bp[16 * cell + 4 * noff + koff] = 0;
        }
    }

    return 0;
}

typedef __vector int32_t v4si_t __attribute__((aligned(4)));

#define SWIZZLE_4x4 \
    { \
        result_i[0] = vec_perm(result[0], result[1], swizA); \
        result_i[1] = vec_perm(result[0], result[1], swizB); \
        result_i[2] = vec_perm(result[2], result[3], swizA); \
        result_i[3] = vec_perm(result[2], result[3], swizB); \
        result_t[0] = vec_perm(result_i[0], result_i[2], swizC); \
        result_t[1] = vec_perm(result_i[0], result_i[2], swizD); \
        result_t[2] = vec_perm(result_i[1], result_i[3], swizC); \
        result_t[3] = vec_perm(result_i[1], result_i[3], swizD); \
    }

#define SAVE_ACC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), 0); \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), 0); \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), 0);

#define SAVE_ACC1(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), 0); \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), 0); \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), 0);

#define SAVE_ACC_COND(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    if ((n_cap - n) < 3) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), \
                0); \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    if ((n_cap - n) < 2) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), \
                0); \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    if ((n_cap - n) < 1) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), \
                0);

#define SAVE_ACC1_COND(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = vec_cts( \
            beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[0], 0), 0); \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    if ((n_cap - n) < 3) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[1], 0), \
                0); \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    if ((n_cap - n) < 2) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[2], 0), \
                0); \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    if ((n_cap - n) < 1) \
        rowC[0] = vec_cts( \
                beta * vec_ctf(rowC[0], 0) + alpha * vec_ctf(result_t[3], 0), \
                0);

#define SAVE_ACC_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    rowC[0] = result_t[3];

#define SAVE_ACC1_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    rowC[0] = result_t[3];

#define SAVE_ACC_COND_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[0 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[1 * ldc + J]; \
    if ((n_cap - n) < 3) rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[2 * ldc + J]; \
    if ((n_cap - n) < 2) rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[3 * ldc + J]; \
    if ((n_cap - n) < 1) rowC[0] = result_t[3];

#define SAVE_ACC1_COND_ABSC(ACC, J) \
    __builtin_mma_disassemble_acc((void *)result, ACC); \
    SWIZZLE_4x4 rowC = (v4si_t *)&CO[4 * ldc + J]; \
    rowC[0] = result_t[0]; \
    rowC = (v4si_t *)&CO[5 * ldc + J]; \
    if ((n_cap - n) < 3) rowC[0] = result_t[1]; \
    rowC = (v4si_t *)&CO[6 * ldc + J]; \
    if ((n_cap - n) < 2) rowC[0] = result_t[2]; \
    rowC = (v4si_t *)&CO[7 * ldc + J]; \
    if ((n_cap - n) < 1) rowC[0] = result_t[3];

#define SET_ACC_ZERO4() \
    __builtin_mma_xxsetaccz(&acc0); \
    __builtin_mma_xxsetaccz(&acc1); \
    __builtin_mma_xxsetaccz(&acc2); \
    __builtin_mma_xxsetaccz(&acc3);

#define SET_ACC_ZERO8() \
    __builtin_mma_xxsetaccz(&acc0); \
    __builtin_mma_xxsetaccz(&acc1); \
    __builtin_mma_xxsetaccz(&acc2); \
    __builtin_mma_xxsetaccz(&acc3); \
    __builtin_mma_xxsetaccz(&acc4); \
    __builtin_mma_xxsetaccz(&acc5); \
    __builtin_mma_xxsetaccz(&acc6); \
    __builtin_mma_xxsetaccz(&acc7);

#define PREFETCH1(x, y) \
    asm volatile("dcbt %0, %1" : : "r"(x), "b"(y) : "memory");

#define MMA __builtin_mma_xvi16ger2pp

void gemm_kernel_16bit(dim_t m, dim_t n, dim_t k, float alpha, short *A,
        short *B, int32_t *C, float beta, dim_t ldc) {
    int32_t i;
    int32_t m_cap = (m + 3) & ~3;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 1) & ~1;
    int32_t m_skip;
    int32_t n_skip = (n & 8) != (n_cap & 8);
    int32_t fastpath;
    v4si_t result[4], result_i[4], result_t[4];
    vec_t swizA = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    vec_t swizB
            = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    vec_t swizC = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    vec_t swizD
            = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
    fastpath = ((alpha == 1.0) && (beta == 0.0));

    /* Loop for multiples of 8 */
    i = n_cap >> 3;
    while (i) {
        int32_t j;
        int32_t *CO;
        short *AO;
        CO = C;
        C += ldc << 3;
        AO = A;
        PREFETCH1(A, 128);
        PREFETCH1(A, 256);
        /* Loop for m >= 16. */
        j = m_cap >> 4;
        m_skip = (m >> 4) != (m_cap >> 4);
        while (j) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            int32_t l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                MMA(&acc4, rowA[2], rowB[0]);
                MMA(&acc5, rowA[2], rowB[1]);
                MMA(&acc6, rowA[3], rowB[0]);
                MMA(&acc7, rowA[3], rowB[1]);
                rowA += 4;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc3, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc5, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc6, 0);
                        SAVE_ACC1_COND_ABSC(&acc7, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC1_ABSC(&acc7, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc3, 0);
                } else {
                    SAVE_ACC1(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc5, 0);
                } else {
                    SAVE_ACC1(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!(n_skip)) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc6, 0);
                        SAVE_ACC1_COND(&acc7, 0);
                    }
                } else {
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC1(&acc7, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 4);
            BO += (k_cap << 3);
            --j;
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                rowA += 2;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc2, 0);
                        SAVE_ACC1_COND_ABSC(&acc3, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc2, 0);
                        SAVE_ACC1_COND(&acc3, 0);
                    }
                } else {
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC1(&acc3, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 3);
            BO += (k_cap << 3);
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l = 0;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                rowA += 1;
                rowB += 2;
            }

            if (fastpath) {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC_ABSC(&acc0, 0);
                        SAVE_ACC1_COND_ABSC(&acc1, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
            } else {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                            + alpha * result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                            + alpha * result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                            + alpha * result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC(&acc0, 0);
                        SAVE_ACC1_COND(&acc1, 0);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC1(&acc1, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 2);
            BO += (k_cap << 3);
        }

    endloop8:
        B += k_cap << 3;
        i -= 1;
    }

    if (n_cap & 4) {
        int32_t j;
        int32_t *CO;
        short *AO;
        CO = C;
        C += ldc << 2;
        AO = A;
        int32_t n_skip = (n != n_cap);
        /* Loop for m >= 32. */
        m_skip = (m >> 5) != (m_cap >> 5);
        for (j = 0; j < (m_cap >> 5); j++) {
            short *BO = B;
            short *A1 = AO + (16 * k_cap);
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowA1 = (vec_t *)A1;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                MMA(&acc4, rowA1[0], rowB[0]);
                MMA(&acc5, rowA1[1], rowB[0]);
                MMA(&acc6, rowA1[2], rowB[0]);
                MMA(&acc7, rowA1[3], rowB[0]);
                rowA += 4;
                rowA1 += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    SAVE_ACC_COND_ABSC(&acc3, 12);
                    SAVE_ACC_COND_ABSC(&acc4, 16);
                    SAVE_ACC_COND_ABSC(&acc5, 20);
                    SAVE_ACC_COND_ABSC(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc4, 0);
                    SAVE_ACC_ABSC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC_ABSC(&acc7, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    SAVE_ACC_COND(&acc3, 12);
                    SAVE_ACC_COND(&acc4, 16);
                    SAVE_ACC_COND(&acc5, 20);
                    SAVE_ACC_COND(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = beta * CO[0 * ldc + 28 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii]
                                        = beta * CO[1 * ldc + 28 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii]
                                        = beta * CO[2 * ldc + 28 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii]
                                        = beta * CO[3 * ldc + 28 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC(&acc4, 0);
                    SAVE_ACC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC(&acc7, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 5;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 16) != (m_cap & 16);

        if (m_cap & 16) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                rowA += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = beta * CO[0 * ldc + 12 + ii]
                                    + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii]
                                        = beta * CO[1 * ldc + 12 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii]
                                        = beta * CO[2 * ldc + 12 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii]
                                        = beta * CO[3 * ldc + 12 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 4;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                rowA += 2;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    if (m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc1, 4);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    if (m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = beta * CO[0 * ldc + 4 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii]
                                        = beta * CO[1 * ldc + 4 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii]
                                        = beta * CO[2 * ldc + 4 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii]
                                        = beta * CO[3 * ldc + 4 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc1, 4);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                }
            }
            CO += 8;
            AO += k_cap << 3;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            short *BO = B;
            v4si_t *rowC;
            __vector_quad acc0;
            __builtin_mma_xxsetaccz(&acc0);
            int32_t l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 2; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                rowA += 1;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    int32_t count = 4 - (m_cap - m);
                    int32_t ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                }
            } else {
                if (m_skip || n_skip) {
                    int32_t count = 4 - (m_cap - m);
                    int32_t ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = beta * CO[0 * ldc + ii] + alpha * result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                } else {
                    SAVE_ACC(&acc0, 0);
                }
            }
            CO += 4;
            AO += k_cap << 2;
            BO += k_cap << 2;
        }

    endloop4:
        B += k_cap << 2;
    }
    return;
}

#undef MMA
#define MMA __builtin_mma_xvi8ger4pp

void gemm_kernel_8bit(dim_t m, dim_t n, dim_t k, float alpha, int8_t *A,
        uint8_t *B, int32_t *C, float beta, dim_t ldc) {
    int32_t i;
    int32_t m_cap = (m + 3) & ~3;
    int32_t n_cap = (n + 3) & ~3;
    int32_t k_cap = (k + 3) & ~3;
    int32_t m_skip;
    int32_t n_skip = (n & 8) != (n_cap & 8);
    int32_t fastpath;
    v4si_t result[4], result_i[4], result_t[4];
    vec_t swizA = {0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23};
    vec_t swizB
            = {8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31};
    vec_t swizC = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23};
    vec_t swizD
            = {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31};
    fastpath = ((alpha == 1.0) && (beta == 0.0));

    /* Loop for multiples of 8 */
    i = n_cap >> 3;
    while (i) {
        int32_t j;
        int32_t *CO;
        int8_t *AO;
        CO = C;
        C += ldc << 3;
        AO = A;
        PREFETCH1(A, 128);
        PREFETCH1(A, 256);
        /* Loop for m >= 16. */
        j = m_cap >> 4;
        m_skip = (m >> 4) != (m_cap >> 4);
        while (j) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            int32_t l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                MMA(&acc4, rowA[2], rowB[0]);
                MMA(&acc5, rowA[2], rowB[1]);
                MMA(&acc6, rowA[3], rowB[0]);
                MMA(&acc7, rowA[3], rowB[1]);
                rowA += 4;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc3, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC_ABSC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc5, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc6, 0);
                        SAVE_ACC1_COND_ABSC(&acc7, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC1_ABSC(&acc7, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                SAVE_ACC(&acc2, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc3, 0);
                } else {
                    SAVE_ACC1(&acc3, 0);
                }
                CO += 4;
                SAVE_ACC(&acc4, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc5, 0);
                } else {
                    SAVE_ACC1(&acc5, 0);
                }
                CO += 4;
                if (((j == 1) && m_skip) || ((i == 1) && n_skip)) {
                    if ((j == 1) && m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc6);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc6, 0);
                        SAVE_ACC1_COND(&acc7, 0);
                    }
                } else {
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC1(&acc7, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 4);
            BO += (k_cap << 3);
            --j;
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                MMA(&acc2, rowA[1], rowB[0]);
                MMA(&acc3, rowA[1], rowB[1]);
                rowA += 2;
                rowB += 2;
            }

            if (fastpath) {
                SAVE_ACC_ABSC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND_ABSC(&acc1, 0);
                } else {
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_ABSC(&acc2, 0);
                        SAVE_ACC1_COND_ABSC(&acc3, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC1_ABSC(&acc3, 0);
                }
            } else {
                SAVE_ACC(&acc0, 0);
                if ((i == 1) && n_skip) {
                    SAVE_ACC1_COND(&acc1, 0);
                } else {
                    SAVE_ACC1(&acc1, 0);
                }
                CO += 4;
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc2);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                        if ((i > 1) || (!n_skip) || (n_cap & 4)
                                || (n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC(&acc2, 0);
                        SAVE_ACC1_COND(&acc3, 0);
                    }
                } else {
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC1(&acc3, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 3);
            BO += (k_cap << 3);
        }

        if (m_skip) goto endloop8;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l = 0;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[0], rowB[1]);
                rowA += 1;
                rowB += 2;
            }

            if (fastpath) {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC_ABSC(&acc0, 0);
                        SAVE_ACC1_COND_ABSC(&acc1, 0);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC1_ABSC(&acc1, 0);
                }
            } else {
                if (m_skip || ((i == 1) & n_skip)) {
                    if (m_skip) {
                        int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        __builtin_mma_disassemble_acc((void *)result, &acc0);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + ii]
                                = beta * CO[0 * ldc + ii]
                                + alpha * result_t[0][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[4 * ldc + ii]
                                = beta * CO[4 * ldc + ii]
                                + alpha * result_t[0][ii];
                        if ((i == 1) & n_skip) {
                            if ((n_cap & 4) || (n_cap - n) < 3)
                                for (ii = 0; ii < count; ++ii)
                                    CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                            + alpha * result_t[1][ii];
                            if ((n_cap & 4) || (n_cap - n) < 2)
                                for (ii = 0; ii < count; ++ii)
                                    CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                            + alpha * result_t[2][ii];
                            if ((n_cap & 4) || (n_cap - n) < 1)
                                for (ii = 0; ii < count; ++ii)
                                    CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                            + alpha * result_t[3][ii];
                        } else {
                            for (ii = 0; ii < count; ++ii)
                                CO[5 * ldc + ii] = beta * CO[5 * ldc + ii]
                                        + alpha * result_t[1][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[6 * ldc + ii] = beta * CO[6 * ldc + ii]
                                        + alpha * result_t[2][ii];
                            for (ii = 0; ii < count; ++ii)
                                CO[7 * ldc + ii] = beta * CO[7 * ldc + ii]
                                        + alpha * result_t[3][ii];
                        }
                    } else {
                        SAVE_ACC(&acc0, 0);
                        SAVE_ACC1_COND(&acc1, 0);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC1(&acc1, 0);
                }
            }
            CO += 4;
            AO += (k_cap << 2);
            BO += (k_cap << 3);
        }

    endloop8:
        B += k_cap << 3;
        i -= 1;
    }

    if (n_cap & 4) {
        int32_t j;
        int32_t *CO;
        int8_t *AO;
        CO = C;
        C += ldc << 2;
        AO = A;
        int32_t n_skip = (n != n_cap);
        /* Loop for m >= 32. */
        m_skip = (m >> 5) != (m_cap >> 5);
        for (j = 0; j < (m_cap >> 5); j++) {
            uint8_t *BO = B;
            int8_t *A1 = AO + (16 * k_cap);
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
            SET_ACC_ZERO8();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowA1 = (vec_t *)A1;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                MMA(&acc4, rowA1[0], rowB[0]);
                MMA(&acc5, rowA1[1], rowB[0]);
                MMA(&acc6, rowA1[2], rowB[0]);
                MMA(&acc7, rowA1[3], rowB[0]);
                rowA += 4;
                rowA1 += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    SAVE_ACC_COND_ABSC(&acc3, 12);
                    SAVE_ACC_COND_ABSC(&acc4, 16);
                    SAVE_ACC_COND_ABSC(&acc5, 20);
                    SAVE_ACC_COND_ABSC(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc4, 0);
                    SAVE_ACC_ABSC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc6, 0);
                    SAVE_ACC_ABSC(&acc7, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    SAVE_ACC_COND(&acc3, 12);
                    SAVE_ACC_COND(&acc4, 16);
                    SAVE_ACC_COND(&acc5, 20);
                    SAVE_ACC_COND(&acc6, 24);
                    if ((j == (m_cap >> 5) - 1) && m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc7);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 28 + ii]
                                = beta * CO[0 * ldc + 28 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 28 + ii]
                                        = beta * CO[1 * ldc + 28 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 28 + ii]
                                        = beta * CO[2 * ldc + 28 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 28 + ii]
                                        = beta * CO[3 * ldc + 28 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc7, 28);
                    }
                    CO += 32;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                    SAVE_ACC(&acc4, 0);
                    SAVE_ACC(&acc5, 4);
                    CO += 8;
                    SAVE_ACC(&acc6, 0);
                    SAVE_ACC(&acc7, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 5;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 16) != (m_cap & 16);

        if (m_cap & 16) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1, acc2, acc3;
            SET_ACC_ZERO4();
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                MMA(&acc2, rowA[2], rowB[0]);
                MMA(&acc3, rowA[3], rowB[0]);
                rowA += 4;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    SAVE_ACC_COND_ABSC(&acc1, 4);
                    SAVE_ACC_COND_ABSC(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC_ABSC(&acc2, 0);
                    SAVE_ACC_ABSC(&acc3, 4);
                    CO += 8;
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    SAVE_ACC_COND(&acc1, 4);
                    SAVE_ACC_COND(&acc2, 8);
                    if (m_skip) {
                        __builtin_mma_disassemble_acc((void *)result, &acc3);
                        SWIZZLE_4x4 int32_t count = 4 - (m_cap - m);
                        int32_t ii;
                        for (ii = 0; ii < count; ++ii)
                            CO[0 * ldc + 12 + ii] = beta * CO[0 * ldc + 12 + ii]
                                    + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 12 + ii]
                                        = beta * CO[1 * ldc + 12 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 12 + ii]
                                        = beta * CO[2 * ldc + 12 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 12 + ii]
                                        = beta * CO[3 * ldc + 12 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc3, 12);
                    }
                    CO += 16;
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                    CO += 8;
                    SAVE_ACC(&acc2, 0);
                    SAVE_ACC(&acc3, 4);
                    CO += 8;
                }
            }
            AO += k_cap << 4;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 8) != (m_cap & 8);

        if (m_cap & 8) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0, acc1;
            __builtin_mma_xxsetaccz(&acc0);
            __builtin_mma_xxsetaccz(&acc1);
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            int32_t l;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                MMA(&acc1, rowA[1], rowB[0]);
                rowA += 2;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND_ABSC(&acc0, 0);
                    if (m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii] = result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii] = result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii] = result_t[3][ii];
                    } else {
                        SAVE_ACC_COND_ABSC(&acc1, 4);
                    }
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                    SAVE_ACC_ABSC(&acc1, 4);
                }
            } else {
                if (m_skip || n_skip) {
                    SAVE_ACC_COND(&acc0, 0);
                    if (m_skip) {
                        int32_t ii;
                        int32_t count = 4 - (m_cap - m);
                        __builtin_mma_disassemble_acc((void *)result, &acc1);
                        SWIZZLE_4x4 for (ii = 0; ii < count; ++ii)
                                CO[0 * ldc + 4 + ii]
                                = beta * CO[0 * ldc + 4 + ii]
                                + alpha * result_t[0][ii];
                        if ((n_cap - n) < 3)
                            for (ii = 0; ii < count; ++ii)
                                CO[1 * ldc + 4 + ii]
                                        = beta * CO[1 * ldc + 4 + ii]
                                        + alpha * result_t[1][ii];
                        if ((n_cap - n) < 2)
                            for (ii = 0; ii < count; ++ii)
                                CO[2 * ldc + 4 + ii]
                                        = beta * CO[2 * ldc + 4 + ii]
                                        + alpha * result_t[2][ii];
                        if ((n_cap - n) < 1)
                            for (ii = 0; ii < count; ++ii)
                                CO[3 * ldc + 4 + ii]
                                        = beta * CO[3 * ldc + 4 + ii]
                                        + alpha * result_t[3][ii];
                    } else {
                        SAVE_ACC_COND(&acc1, 4);
                    }
                } else {
                    SAVE_ACC(&acc0, 0);
                    SAVE_ACC(&acc1, 4);
                }
            }
            CO += 8;
            AO += k_cap << 3;
            BO += k_cap << 2;
        }

        if (m_skip) goto endloop4;

        m_skip = (m & 4) != (m_cap & 4);

        if (m_cap & 4) {
            uint8_t *BO = B;
            v4si_t *rowC;
            __vector_quad acc0;
            __builtin_mma_xxsetaccz(&acc0);
            int32_t l;
            vec_t *rowA = (vec_t *)AO;
            vec_t *rowB = (vec_t *)BO;
            for (l = 0; l < k_cap / 4; l++) {
                MMA(&acc0, rowA[0], rowB[0]);
                rowA += 1;
                rowB += 1;
            }

            if (fastpath) {
                if (m_skip || n_skip) {
                    int32_t count = 4 - (m_cap - m);
                    int32_t ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = result_t[3][ii];
                } else {
                    SAVE_ACC_ABSC(&acc0, 0);
                }
            } else {
                if (m_skip || n_skip) {
                    int32_t count = 4 - (m_cap - m);
                    int32_t ii;
                    __builtin_mma_disassemble_acc((void *)result, &acc0);
                    SWIZZLE_4x4 for (ii = 0; ii < count; ++ii) CO[0 * ldc + ii]
                            = beta * CO[0 * ldc + ii] + alpha * result_t[0][ii];
                    if ((n_cap - n) < 3)
                        for (ii = 0; ii < count; ++ii)
                            CO[1 * ldc + ii] = beta * CO[1 * ldc + ii]
                                    + alpha * result_t[1][ii];
                    if ((n_cap - n) < 2)
                        for (ii = 0; ii < count; ++ii)
                            CO[2 * ldc + ii] = beta * CO[2 * ldc + ii]
                                    + alpha * result_t[2][ii];
                    if ((n_cap - n) < 1)
                        for (ii = 0; ii < count; ++ii)
                            CO[3 * ldc + ii] = beta * CO[3 * ldc + ii]
                                    + alpha * result_t[3][ii];
                } else {
                    SAVE_ACC(&acc0, 0);
                }
            }
            CO += 4;
            AO += k_cap << 2;
            BO += k_cap << 2;
        }

    endloop4:
        B += k_cap << 2;
    }
    return;
}

} // namespace impl
} // namespace dnnl
