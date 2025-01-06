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

// QUDA_MULTIGRID_MMA_PROLONGATOR modulo 8
// - 0 DEFAULT
// - 1 SIMT
// - 2 SMMA
// - 3 1xFP16
// - 4 3xFP16
// - 5 1xTF32
// - 6 3xTF32
// - 7 3xBF16

#if defined(QUDA_MULTIGRID_MMA_DSLASH)
    template <class T>
    struct mg_mma_dslash_t { };

    // @brief half precision specialization
    template <>
    struct mg_mma_dslash_t<short> {
#if ((QUDA_MULTIGRID_MMA_DSLASH & 7) == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif ((QUDA_MULTIGRID_MMA_DSLASH & 7) == 2)
      using type = typename smma_dispatch<short>::type;
#elif ((QUDA_MULTIGRID_MMA_DSLASH & 7) == 3)
      using type = hmma_t;
#elif ((QUDA_MULTIGRID_MMA_DSLASH & 7) == 4)
      using type = smma_half_t;
#elif ((QUDA_MULTIGRID_MMA_DSLASH & 7) == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_DSLASH & 7) == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_DSLASH & 7) == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_DSLASH & 7) == 0)
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<short>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };

    // @brief single precision specialization
    template <>
    struct mg_mma_dslash_t<float> {
#if ((QUDA_MULTIGRID_MMA_DSLASH >> 3) == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif ((QUDA_MULTIGRID_MMA_DSLASH >> 3) == 2)
      using type = typename smma_dispatch<float>::type;
#elif ((QUDA_MULTIGRID_MMA_DSLASH >> 3) == 3)
      using type = hmma_t;
#elif ((QUDA_MULTIGRID_MMA_DSLASH >> 3) == 4)
      using type = smma_half_t;
#elif ((QUDA_MULTIGRID_MMA_DSLASH >> 3) == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_DSLASH >> 3) == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_DSLASH >> 3) == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_DSLASH >> 3) == 0)
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<float>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };
#endif

#if defined(QUDA_MULTIGRID_MMA_PROLONGATOR)
    template <class T>
    struct mg_mma_prolongator_t { };

    // @brief half precision specialization
    template <>
    struct mg_mma_prolongator_t<short> {
#if ((QUDA_MULTIGRID_MMA_PROLONGATOR & 7) == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR & 7) == 2)
      using type = typename smma_dispatch<short>::type;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR & 7) == 3)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = hmma::hmma_x_t<16, 8, 8, half, half2>;
  #else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
  #endif
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR & 7) == 4)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
  #else
      using type = smma_half_t;
  #endif
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR & 7) == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR & 7) == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR & 7) == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR & 7) == 0)
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<short>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };

    // @brief single precision specialization
    template <>
    struct mg_mma_prolongator_t<float> {
#if ((QUDA_MULTIGRID_MMA_PROLONGATOR >> 3) == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR >> 3) == 2)
      using type = typename smma_dispatch<float>::type;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR >> 3) == 3)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = hmma::hmma_x_t<16, 8, 8, half, half2>;
  #else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
  #endif
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR >> 3) == 4)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
  #else
      using type = smma_half_t;
  #endif
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR >> 3) == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR >> 3) == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR >> 3) == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_PROLONGATOR >> 3) == 0)
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<float>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };
#endif

#if defined(QUDA_MULTIGRID_MMA_RESTRICTOR)
    template <class T>
    struct mg_mma_restrictor_t { };

    // @brief half precision specialization
    template <>
    struct mg_mma_restrictor_t<short> {
#if ((QUDA_MULTIGRID_MMA_RESTRICTOR & 7) == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR & 7) == 2)
      using type = typename smma_dispatch<short>::type;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR & 7) == 3)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = hmma::hmma_x_t<16, 8, 8, half, half2>;
  #else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
  #endif
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR & 7) == 4)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
  #else
      using type = smma_half_t;
  #endif
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR & 7) == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR & 7) == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR & 7) == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR & 7) == 0)
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<short>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };

    // @brief single precision specialization
    template <>
    struct mg_mma_restrictor_t<float> {
#if ((QUDA_MULTIGRID_MMA_RESTRICTOR >> 3) == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR >> 3) == 2)
      using type = typename smma_dispatch<float>::type;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR >> 3) == 3)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = hmma::hmma_x_t<16, 8, 8, half, half2>;
  #else
      using type = hmma::hmma_t<16, 8, 8, half, half2>;
  #endif
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR >> 3) == 4)
  #if (__COMPUTE_CAPABILITY__ == 700)
      using type = smma::smma_x_t<mma::half, 8, 1, 1>;
  #else
      using type = smma_half_t;
  #endif
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR >> 3) == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR >> 3) == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR >> 3) == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_RESTRICTOR >> 3) == 0)
  #if (__COMPUTE_CAPABILITY__ >= 800)
      using type = typename smma_dispatch<float>::type;
  #else
      using type = simt::simt_t<float, 8, 4, 2, 2>;
  #endif
#endif
    };
#endif

#if defined(QUDA_MULTIGRID_MMA_SETUP)
    template <class T> struct mg_mma_setup_t { };

    // @brief half precision specialization
    template <> struct mg_mma_setup_t<short> {
#if ((QUDA_MULTIGRID_MMA_SETUP & 7) == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif ((QUDA_MULTIGRID_MMA_SETUP & 7) == 2)
      using type = typename smma_dispatch<short>::type;
#elif ((QUDA_MULTIGRID_MMA_SETUP & 7) == 3)
      using type = hmma_t;
#elif ((QUDA_MULTIGRID_MMA_SETUP & 7) == 4)
      using type = smma_half_t;
#elif ((QUDA_MULTIGRID_MMA_SETUP & 7) == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_SETUP & 7) == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_SETUP & 7) == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_SETUP & 7) == 0)
      using type = hmma_t;
#endif
    };

    // @brief single precision specialization
    template <> struct mg_mma_setup_t<float> {
#if ((QUDA_MULTIGRID_MMA_SETUP >> 3) == 1)
      using type = simt::simt_t<float, 8, 4, 2, 2>;
#elif ((QUDA_MULTIGRID_MMA_SETUP >> 3) == 2)
      using type = typename smma_dispatch<float>::type;
#elif ((QUDA_MULTIGRID_MMA_SETUP >> 3) == 3)
      using type = hmma_t;
#elif ((QUDA_MULTIGRID_MMA_SETUP >> 3) == 4)
      using type = smma_half_t;
#elif ((QUDA_MULTIGRID_MMA_SETUP >> 3) == 5)
      using type = hmma::hmma_tfloat32_t<4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_SETUP >> 3) == 6)
      using type = smma::smma_t<mma::tfloat32, 4, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_SETUP >> 3) == 7)
      using type = smma::smma_t<mma::bfloat16, 8, 1, 1>;
#elif ((QUDA_MULTIGRID_MMA_SETUP >> 3) == 0)
      using type = hmma_t;
#endif
    };
#endif

  } // namespace mma
} // namespace quda
