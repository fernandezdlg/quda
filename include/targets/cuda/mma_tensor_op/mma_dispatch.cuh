#pragma once

namespace quda
{
  namespace hmma
  {
    template <int m, int n, int k, class compute_t, class load_t> struct hmma_t {
    };
  } // namespace hmma
} // namespace quda

#include <mma_tensor_op/hmma_m16n16k4_sm70.cuh>
#include <mma_tensor_op/hmma_m16n8k8_sm80.cuh>

#include <mma_tensor_op/hmma_tfloat32_sm80.cuh>

#include <mma_tensor_op/smma_m16n8_sm80.cuh>
#include <mma_tensor_op/smma_m16n16k4_sm70.cuh>

#include <mma_tensor_op/simt.cuh>
#include <mma_tensor_op/simt_half.cuh>

#include <quda_define.h>

namespace quda
{
  namespace mma
  {
#if (__COMPUTE_CAPABILITY__ == 700)
    using hmma_t = hmma::hmma_t<16, 16, 4, half, half2>;
    using smma_half_t = smma::smma_t<half, 4, 1, 1>;
#else
    using hmma_t = hmma::hmma_t<16, 8, 8, half, half2>;
    using smma_half_t = smma::smma_t<half, 8, 1, 1>;
#endif

#if (__COMPUTE_CAPABILITY__ >= 800)
    template <class T> struct smma_dispatch {
    };

    template <> struct smma_dispatch<float> {
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
    };

    template <> struct smma_dispatch<short> {
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
    };
#else
    template <class T> struct smma_dispatch {
      using type = smma_half_t;
    };
#endif

    template <class T>
    struct mg_mma_dslash_t {
#if defined(QUDA_MULTIGRID_MMA_DSLASH_SIMT)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif defined(QUDA_MULTIGRID_MMA_DSLASH_SMMA)
      using type = typename smma_dispatch<T>::type;
#elif defined(QUDA_MULTIGRID_MMA_DSLASH_1xFP16)
      using type = hmma_t;
#elif defined(QUDA_MULTIGRID_MMA_DSLASH_3xFP16)
      using type = smma_half_t;
#elif defined(QUDA_MULTIGRID_MMA_DSLASH_1xTF32)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif defined(QUDA_MULTIGRID_MMA_DSLASH_3xTF32)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif defined(QUDA_MULTIGRID_MMA_DSLASH_3xBF16)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#else
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<T>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };

    template <class T>
    struct mg_mma_prolongator_t {
#if defined(QUDA_MULTIGRID_MMA_PROLONGATOR_SIMT)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif defined(QUDA_MULTIGRID_MMA_PROLONGATOR_SMMA)
      using type = typename smma_dispatch<T>::type;
#elif defined(QUDA_MULTIGRID_MMA_PROLONGATOR_1xFP16)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
  #else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
  #endif
#elif defined(QUDA_MULTIGRID_MMA_PROLONGATOR_3xFP16)
      using type = smma_half_t;
#elif defined(QUDA_MULTIGRID_MMA_PROLONGATOR_1xTF32)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif defined(QUDA_MULTIGRID_MMA_PROLONGATOR_3xTF32)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif defined(QUDA_MULTIGRID_MMA_PROLONGATOR_3xBF16)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#else
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<T>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };

    template <class T>
    struct mg_mma_restrictor_t {
#if defined(QUDA_MULTIGRID_MMA_RESTRICTOR_SIMT)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif defined(QUDA_MULTIGRID_MMA_RESTRICTOR_SMMA)
      using type = typename smma_dispatch<T>::type;
#elif defined(QUDA_MULTIGRID_MMA_RESTRICTOR_1xFP16)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
  #else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
  #endif
#elif defined(QUDA_MULTIGRID_MMA_RESTRICTOR_3xFP16)
      using type = smma_half_t;
#elif defined(QUDA_MULTIGRID_MMA_RESTRICTOR_1xTF32)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif defined(QUDA_MULTIGRID_MMA_RESTRICTOR_3xTF32)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif defined(QUDA_MULTIGRID_MMA_RESTRICTOR_3xBF16)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#else
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<T>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };

    template <class T> struct mg_mma_setup_t {
#if defined(QUDA_MULTIGRID_MMA_SETUP_SIMT)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif defined(QUDA_MULTIGRID_MMA_SETUP_SMMA)
      using type = typename smma_dispatch<T>::type;
#elif defined(QUDA_MULTIGRID_MMA_SETUP_1xFP16)
      using type = hmma_t;
#elif defined(QUDA_MULTIGRID_MMA_SETUP_3xFP16)
      using type = smma_half_t;
#elif defined(QUDA_MULTIGRID_MMA_SETUP_1xTF32)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif defined(QUDA_MULTIGRID_MMA_SETUP_3xTF32)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif defined(QUDA_MULTIGRID_MMA_SETUP_3xBF16)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#else
      using type = hmma_t;
#endif
    };

  } // namespace mma
} // namespace quda
