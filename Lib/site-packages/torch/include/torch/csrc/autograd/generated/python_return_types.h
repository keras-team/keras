#pragma once

namespace torch {
namespace autograd {
namespace generated {

PyTypeObject* get__fake_quantize_per_tensor_affine_cachemask_tensor_qparams_structseq();
PyTypeObject* get__fused_moving_avg_obs_fq_helper_structseq();
PyTypeObject* get__linalg_det_structseq();
PyTypeObject* get__linalg_det_out_structseq();
PyTypeObject* get__linalg_eigh_structseq();
PyTypeObject* get__linalg_eigh_out_structseq();
PyTypeObject* get__linalg_slogdet_structseq();
PyTypeObject* get__linalg_slogdet_out_structseq();
PyTypeObject* get__linalg_solve_ex_structseq();
PyTypeObject* get__linalg_solve_ex_out_structseq();
PyTypeObject* get__linalg_svd_structseq();
PyTypeObject* get__linalg_svd_out_structseq();
PyTypeObject* get__lu_with_info_structseq();
PyTypeObject* get__scaled_dot_product_cudnn_attention_structseq();
PyTypeObject* get__scaled_dot_product_efficient_attention_structseq();
PyTypeObject* get__scaled_dot_product_flash_attention_structseq();
PyTypeObject* get__scaled_dot_product_flash_attention_for_cpu_structseq();
PyTypeObject* get__unpack_dual_structseq();
PyTypeObject* get_aminmax_structseq();
PyTypeObject* get_aminmax_out_structseq();
PyTypeObject* get_cummax_structseq();
PyTypeObject* get_cummax_out_structseq();
PyTypeObject* get_cummin_structseq();
PyTypeObject* get_cummin_out_structseq();
PyTypeObject* get_frexp_structseq();
PyTypeObject* get_frexp_out_structseq();
PyTypeObject* get_geqrf_out_structseq();
PyTypeObject* get_geqrf_structseq();
PyTypeObject* get_histogram_out_structseq();
PyTypeObject* get_histogram_structseq();
PyTypeObject* get_histogramdd_structseq();
PyTypeObject* get_kthvalue_structseq();
PyTypeObject* get_kthvalue_out_structseq();
PyTypeObject* get_linalg_cholesky_ex_structseq();
PyTypeObject* get_linalg_cholesky_ex_out_structseq();
PyTypeObject* get_linalg_eig_structseq();
PyTypeObject* get_linalg_eig_out_structseq();
PyTypeObject* get_linalg_eigh_structseq();
PyTypeObject* get_linalg_eigh_out_structseq();
PyTypeObject* get_linalg_inv_ex_structseq();
PyTypeObject* get_linalg_inv_ex_out_structseq();
PyTypeObject* get_linalg_ldl_factor_structseq();
PyTypeObject* get_linalg_ldl_factor_out_structseq();
PyTypeObject* get_linalg_ldl_factor_ex_structseq();
PyTypeObject* get_linalg_ldl_factor_ex_out_structseq();
PyTypeObject* get_linalg_lstsq_structseq();
PyTypeObject* get_linalg_lstsq_out_structseq();
PyTypeObject* get_linalg_lu_structseq();
PyTypeObject* get_linalg_lu_out_structseq();
PyTypeObject* get_linalg_lu_factor_structseq();
PyTypeObject* get_linalg_lu_factor_out_structseq();
PyTypeObject* get_linalg_lu_factor_ex_structseq();
PyTypeObject* get_linalg_lu_factor_ex_out_structseq();
PyTypeObject* get_linalg_qr_structseq();
PyTypeObject* get_linalg_qr_out_structseq();
PyTypeObject* get_linalg_slogdet_structseq();
PyTypeObject* get_linalg_slogdet_out_structseq();
PyTypeObject* get_linalg_solve_ex_structseq();
PyTypeObject* get_linalg_solve_ex_out_structseq();
PyTypeObject* get_linalg_svd_structseq();
PyTypeObject* get_linalg_svd_out_structseq();
PyTypeObject* get_lu_unpack_structseq();
PyTypeObject* get_lu_unpack_out_structseq();
PyTypeObject* get_max_structseq();
PyTypeObject* get_max_out_structseq();
PyTypeObject* get_median_structseq();
PyTypeObject* get_median_out_structseq();
PyTypeObject* get_min_structseq();
PyTypeObject* get_min_out_structseq();
PyTypeObject* get_mode_structseq();
PyTypeObject* get_mode_out_structseq();
PyTypeObject* get_nanmedian_structseq();
PyTypeObject* get_nanmedian_out_structseq();
PyTypeObject* get_qr_out_structseq();
PyTypeObject* get_qr_structseq();
PyTypeObject* get_slogdet_structseq();
PyTypeObject* get_slogdet_out_structseq();
PyTypeObject* get_sort_out_structseq();
PyTypeObject* get_sort_structseq();
PyTypeObject* get_svd_out_structseq();
PyTypeObject* get_svd_structseq();
PyTypeObject* get_topk_out_structseq();
PyTypeObject* get_topk_structseq();
PyTypeObject* get_triangular_solve_out_structseq();
PyTypeObject* get_triangular_solve_structseq();

}

void initReturnTypes(PyObject* module);

} // namespace autograd
} // namespace torch
