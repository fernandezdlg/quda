#include <cuda.h>
#include <unordered_map>

using barrier_t = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

namespace quda
{

  struct tma_descriptor_t {
    CUtensorMap map;
  };

  template <int kRank> struct tma_descriptor_key_t {
    std::array<size_t, kRank> tensor_dims;
    std::array<size_t, kRank> box_dims;
    void *ptr;

    bool operator==(const tma_descriptor_key_t &other) const
    {
      for (size_t i = 0; i < kRank; i++) {
        if (tensor_dims[i] != other.tensor_dims[i] || box_dims[i] != other.box_dims[i]) { return false; }
      }
      if (ptr != other.ptr) { return false; }
      return true;
    }
  };

  template <int kRank> struct tma_descriptor_hash_t {
    std::size_t operator()(const tma_descriptor_key_t<kRank> &key) const
    {
      std::size_t hash = 0;
      for (size_t i = 0; i < kRank; i++) {
        hash = (hash << 1) ^ std::hash<std::size_t> {}(key.tensor_dims[i]);
        hash = (hash << 1) ^ std::hash<std::size_t> {}(key.box_dims[i]);
      }
      hash = (hash << 1) ^ std::hash<void *> {}(key.ptr);
      return hash;
    }
  };

  template <class T, int kRank> tma_descriptor_t make_tma_descriptor(tma_descriptor_key_t<kRank> key)
  {
    CUtensorMap ret_value;

    cuuint64_t tensor_size[kRank];
    for (int i = 0; i < kRank; i++) { tensor_size[i] = static_cast<cuuint64_t>(key.tensor_dims[i]); }
    cuuint64_t tensor_stride[kRank - 1];
    tensor_stride[0] = tensor_size[0] * sizeof(T);
    for (int i = 1; i < kRank - 1; i++) { tensor_stride[i] = tensor_stride[i - 1] * tensor_size[i]; }
    cuuint32_t box_size[kRank];
    for (int i = 0; i < kRank; i++) { box_size[i] = static_cast<cuuint32_t>(key.box_dims[i]); }
    cuuint32_t elem_str[kRank];
    for (int i = 0; i < kRank; i++) { elem_str[i] = 1; }

    CUtensorMapDataType data_type;
    if constexpr (std::is_same_v<T, float>) {
      data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else if constexpr (std::is_same_v<T, short>) {
      data_type = CU_TENSOR_MAP_DATA_TYPE_UINT16;
    } else {
      errorQuda("Unexpected data type for TMA descriptor creation.");
    }

    CUresult error = cuTensorMapEncodeTiled(&ret_value, data_type, kRank, key.ptr, tensor_size, tensor_stride, box_size,
                                            elem_str, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                            CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (CUDA_SUCCESS != error) {
      const char *str;
      cuGetErrorName(error, &str);
      errorQuda("TMA descriptor creation returned %s\n", str);
    }

    return {ret_value};
  }

  template <class T, int kRank> tma_descriptor_t get_tma_descriptor(tma_descriptor_key_t<kRank> key)
  {
    static std::unordered_map<tma_descriptor_key_t<kRank>, tma_descriptor_t, tma_descriptor_hash_t<kRank>> _cache;
    if (_cache.find(key) == _cache.end()) { _cache[key] = make_tma_descriptor<T, kRank>(key); }
    return _cache[key];
  }

  constexpr int keep_divide_by_two_until(int i, int limit)
  {
    if (i <= limit) {
      return i;
    } else {
      return keep_divide_by_two_until(i / 2, limit);
    }
  };

  constexpr int tma_box_limit = 256; // TMA box sizes cannot be larger than 256

  /**
    So if TMA box sizes are larger than 256, we break them into smaller pieces by keep dividing by 2
   */
  template <int box_a, int box_b, class T>
  inline tma_descriptor_t get_tma_descriptor_5d_box_2d(std::array<size_t, 5> tensor_size, complex<T> *ptr)
  {
    constexpr int box_a_reduced = keep_divide_by_two_until(box_a, tma_box_limit);
    constexpr int box_b_reduced = keep_divide_by_two_until(box_b, tma_box_limit);
    tma_descriptor_key_t<5> key = {tensor_size, std::array<size_t, 5> {box_a_reduced, box_b_reduced, 1, 1, 1}, ptr};
    return get_tma_descriptor<T, 5>(key);
  }

  template <int box_a, int box_b, class T>
  inline tma_descriptor_t get_tma_descriptor_4d_box_2d(std::array<size_t, 4> tensor_size, complex<T> *ptr)
  {
    constexpr int box_a_reduced = keep_divide_by_two_until(box_a, tma_box_limit);
    constexpr int box_b_reduced = keep_divide_by_two_until(box_b, tma_box_limit);
    tma_descriptor_key_t<4> key = {tensor_size, std::array<size_t, 4> {box_a_reduced, box_b_reduced, 1, 1}, ptr};
    return get_tma_descriptor<T, 4>(key);
  }

  template <int box_a, int box_b, class T>
  __device__ void inline tma_load_gmem_5d_box_2d(complex<T> *smem_ptr, const CUtensorMap *map, int offset_a,
                                                 int offset_b, int offset_c, int offset_d, int offset_e, barrier_t *bar)
  {
    constexpr int box_a_reduced = keep_divide_by_two_until(box_a, tma_box_limit);
    constexpr int box_b_reduced = keep_divide_by_two_until(box_b, tma_box_limit);
#pragma unroll
    for (int i_a = 0; i_a < box_a; i_a += box_a_reduced) {
#pragma unroll
      for (int i_b = 0; i_b < box_b; i_b += box_b_reduced) {
        T *smem_ptr_reduced = reinterpret_cast<T *>(smem_ptr) + (i_a + i_b * box_a);
        cde::cp_async_bulk_tensor_5d_global_to_shared(smem_ptr_reduced, map, offset_a + i_a, offset_b + i_b, offset_c,
                                                      offset_d, offset_e, *bar);
      }
    }
  }

  template <int box_a, int box_b, class T>
  __device__ void inline tma_load_gmem_4d_box_2d(complex<T> *smem_ptr, const CUtensorMap *map, int offset_a,
                                                 int offset_b, int offset_c, int offset_d, barrier_t *bar)
  {
    constexpr int box_a_reduced = keep_divide_by_two_until(box_a, tma_box_limit);
    constexpr int box_b_reduced = keep_divide_by_two_until(box_b, tma_box_limit);
#pragma unroll
    for (int i_a = 0; i_a < box_a; i_a += box_a_reduced) {
#pragma unroll
      for (int i_b = 0; i_b < box_b; i_b += box_b_reduced) {
        T *smem_ptr_reduced = reinterpret_cast<T *>(smem_ptr) + (i_a + i_b * box_a);
        cde::cp_async_bulk_tensor_4d_global_to_shared(smem_ptr_reduced, map, offset_a + i_a, offset_b + i_b, offset_c,
                                                      offset_d, *bar);
      }
    }
  }

} // namespace quda
