#ifndef EIGEN_MISC_BLAS_H
#define EIGEN_MISC_BLAS_H

extern "C" {

#define BLASFUNC(FUNC) FUNC##_

/* Level 1 routines */

void BLASFUNC(saxpy)(const int *, const float *, const float *, const int *, float *, const int *);
void BLASFUNC(daxpy)(const int *, const double *, const double *, const int *, double *, const int *);
void BLASFUNC(caxpy)(const int *, const float *, const float *, const int *, float *, const int *);
void BLASFUNC(zaxpy)(const int *, const double *, const double *, const int *, double *, const int *);

/* Level 2 routines */

void BLASFUNC(sgemv)(const char *, const int *, const int *, const float *, const float *, const int *, const float *,
                     const int *, const float *, float *, const int *);
void BLASFUNC(dgemv)(const char *, const int *, const int *, const double *, const double *, const int *,
                     const double *, const int *, const double *, double *, const int *);
void BLASFUNC(cgemv)(const char *, const int *, const int *, const float *, const float *, const int *, const float *,
                     const int *, const float *, float *, const int *);
void BLASFUNC(zgemv)(const char *, const int *, const int *, const double *, const double *, const int *,
                     const double *, const int *, const double *, double *, const int *);

void BLASFUNC(strmv)(const char *, const char *, const char *, const int *, const float *, const int *, float *,
                     const int *);
void BLASFUNC(dtrmv)(const char *, const char *, const char *, const int *, const double *, const int *, double *,
                     const int *);
void BLASFUNC(ctrmv)(const char *, const char *, const char *, const int *, const float *, const int *, float *,
                     const int *);
void BLASFUNC(ztrmv)(const char *, const char *, const char *, const int *, const double *, const int *, double *,
                     const int *);

void BLASFUNC(ssymv)(const char *, const int *, const float *, const float *, const int *, const float *, const int *,
                     const float *, float *, const int *);
void BLASFUNC(dsymv)(const char *, const int *, const double *, const double *, const int *, const double *,
                     const int *, const double *, double *, const int *);

void BLASFUNC(chemv)(const char *, const int *, const float *, const float *, const int *, const float *, const int *,
                     const float *, float *, const int *);
void BLASFUNC(zhemv)(const char *, const int *, const double *, const double *, const int *, const double *,
                     const int *, const double *, double *, const int *);

/* Level 3 routines */

void BLASFUNC(sgemm)(const char *, const char *, const int *, const int *, const int *, const float *, const float *,
                     const int *, const float *, const int *, const float *, float *, const int *);
void BLASFUNC(dgemm)(const char *, const char *, const int *, const int *, const int *, const double *, const double *,
                     const int *, const double *, const int *, const double *, double *, const int *);
void BLASFUNC(cgemm)(const char *, const char *, const int *, const int *, const int *, const float *, const float *,
                     const int *, const float *, const int *, const float *, float *, const int *);
void BLASFUNC(zgemm)(const char *, const char *, const int *, const int *, const int *, const double *, const double *,
                     const int *, const double *, const int *, const double *, double *, const int *);

void BLASFUNC(strsm)(const char *, const char *, const char *, const char *, const int *, const int *, const float *,
                     const float *, const int *, float *, const int *);
void BLASFUNC(dtrsm)(const char *, const char *, const char *, const char *, const int *, const int *, const double *,
                     const double *, const int *, double *, const int *);
void BLASFUNC(ctrsm)(const char *, const char *, const char *, const char *, const int *, const int *, const float *,
                     const float *, const int *, float *, const int *);
void BLASFUNC(ztrsm)(const char *, const char *, const char *, const char *, const int *, const int *, const double *,
                     const double *, const int *, double *, const int *);

void BLASFUNC(strmm)(const char *, const char *, const char *, const char *, const int *, const int *, const float *,
                     const float *, const int *, float *, const int *);
void BLASFUNC(dtrmm)(const char *, const char *, const char *, const char *, const int *, const int *, const double *,
                     const double *, const int *, double *, const int *);
void BLASFUNC(ctrmm)(const char *, const char *, const char *, const char *, const int *, const int *, const float *,
                     const float *, const int *, float *, const int *);
void BLASFUNC(ztrmm)(const char *, const char *, const char *, const char *, const int *, const int *, const double *,
                     const double *, const int *, double *, const int *);

void BLASFUNC(ssymm)(const char *, const char *, const int *, const int *, const float *, const float *, const int *,
                     const float *, const int *, const float *, float *, const int *);
void BLASFUNC(dsymm)(const char *, const char *, const int *, const int *, const double *, const double *, const int *,
                     const double *, const int *, const double *, double *, const int *);

void BLASFUNC(ssyrk)(const char *, const char *, const int *, const int *, const float *, const float *, const int *,
                     const float *, float *, const int *);
void BLASFUNC(dsyrk)(const char *, const char *, const int *, const int *, const double *, const double *, const int *,
                     const double *, double *, const int *);

void BLASFUNC(chemm)(const char *, const char *, const int *, const int *, const float *, const float *, const int *,
                     const float *, const int *, const float *, float *, const int *);
void BLASFUNC(zhemm)(const char *, const char *, const int *, const int *, const double *, const double *, const int *,
                     const double *, const int *, const double *, double *, const int *);

void BLASFUNC(cherk)(const char *, const char *, const int *, const int *, const float *, const float *, const int *,
                     const float *, float *, const int *);
void BLASFUNC(zherk)(const char *, const char *, const int *, const int *, const double *, const double *, const int *,
                     const double *, double *, const int *);

#undef BLASFUNC
}

#endif
