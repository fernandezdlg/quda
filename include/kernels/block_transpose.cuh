#pragma once

#include <kernel_helper.h>
#include <color_spinor_field_order.h>
#include <color_spinor_field.h>
#include <shared_memory_cache_helper.h>

namespace quda
{

  /**
      Kernel argument struct
  */
  template <class v_t_, class b_t_, bool is_device_, typename vFloat, typename vAccessor, typename bFloat,
            typename bAccessor, int nSpin_, int nColor_, int nVec_, bool from_to_non_rel_>
  struct BlockTransposeArg : kernel_param<use_kernel_arg_p::TRUE, false> { // no bound checks
    using real = typename mapper<vFloat>::type;
    static constexpr bool is_device = is_device_;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    static constexpr int nVec = nVec_;

    static constexpr int from_to_non_rel = from_to_non_rel_;

    using v_t = v_t_;
    using b_t = b_t_;

    vAccessor V;
    bAccessor B[nVec];
    int_fastdiv block_y;
    int volume_cb;
    int actual_nvec;

    BlockTransposeArg(v_t &V, cvector_ref<b_t> &B_, int block_x, int block_y) :
      // We need full threadblock
      kernel_param(dim3((V.VolumeCB() + block_x - 1) / block_x * block_x, (nVec + block_y - 1) / block_y * block_y,
                        V.SiteSubset() * nColor)),
      V(V),
      block_y(block_y),
      volume_cb(V.VolumeCB()),
      actual_nvec(B_.size())
    {
      for (auto i = 0u; i < B_.size(); i++) { B[i] = bAccessor(B_[i]); }
    }
  };

  template <typename Arg> struct BlockTransposeKernel {
    const Arg &arg;
    constexpr BlockTransposeKernel(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    struct CacheDims {
      static constexpr dim3 dims(dim3 block)
      {
        block.x += 1;
        block.z = 1;
        return block;
      }
    };

    /**
      @brief Transpose between the two different orders of batched colorspinor fields:
        - B: nVec -> spatial/N -> spin/color -> N, where N is for that in floatN
        - V: spatial -> spin/color -> nVec
        The transpose uses shared memory to avoid strided memory accesses.
     */
    __device__ __host__ inline void operator()(int x_cb, int)
    {
      int parity_color = target::block_idx().z;
      int color = parity_color % Arg::nColor;
      int parity = parity_color / Arg::nColor;
      using color_spinor_t = ColorSpinor<typename Arg::real, 1, Arg::nSpin>;

      SharedMemoryCache<color_spinor_t, CacheDims> cache;

      int x_offset = target::block_dim().x * target::block_idx().x;
      int v_offset = target::block_dim().y * target::block_idx().y;

      if constexpr (std::is_const_v<typename Arg::v_t>) {
        int thread_idx = target::thread_idx().x + target::block_dim().x * target::thread_idx().y;
        {
          int v_ = thread_idx % arg.block_y;
          int x_ = thread_idx / arg.block_y;
          if (x_ + x_offset < arg.volume_cb && v_ + v_offset < arg.actual_nvec) {
            color_spinor_t color_spinor;
#pragma unroll
            for (int spin = 0; spin < Arg::nSpin; spin++) {
              color_spinor(spin, 0) = arg.V(parity, x_ + x_offset, spin, color, v_ + v_offset);
            }
            cache.save(color_spinor, x_, v_);
          }
        }
        cache.sync();
        {
          int v = target::thread_idx().y;
          int x = target::thread_idx().x;
          if (x_cb < arg.volume_cb && v + v_offset < arg.actual_nvec) {
            color_spinor_t color_spinor = cache.load(x, v);
            if constexpr (Arg::from_to_non_rel && Arg::nSpin == 4) {
              color_spinor.toNonRel();
              color_spinor *= rsqrt(static_cast<typename Arg::real>(2.0));
            }
#pragma unroll
            for (int spin = 0; spin < Arg::nSpin; spin++) {
              arg.B[v + v_offset](parity, x_cb, spin, color) = color_spinor(spin, 0);
            }
          }
        }
      } else {
        int thread_idx = target::thread_idx().x + target::block_dim().x * target::thread_idx().y;
        {
          int v = target::thread_idx().y;
          int x = target::thread_idx().x;
          if (x_cb < arg.volume_cb && v + v_offset < arg.actual_nvec) {
            color_spinor_t color_spinor;
#pragma unroll
            for (int spin = 0; spin < Arg::nSpin; spin++) {
              color_spinor(spin, 0) = arg.B[v + v_offset](parity, x_cb, spin, color);
            }
            cache.save(color_spinor, x, v);
          }
        }
        cache.sync();
        {
          int v_ = thread_idx % arg.block_y;
          int x_ = thread_idx / arg.block_y;
          if (x_ + x_offset < arg.volume_cb && v_ + v_offset < arg.actual_nvec) {
            color_spinor_t color_spinor = cache.load(x_, v_);
            if constexpr (Arg::nSpin == 4 && Arg::from_to_non_rel) {
              color_spinor.toRel();
              color_spinor *= rsqrt(static_cast<typename Arg::real>(2.0));
            }
#pragma unroll
            for (int spin = 0; spin < Arg::nSpin; spin++) {
              arg.V(parity, x_ + x_offset, spin, color, v_ + v_offset) = color_spinor(spin, 0);
            }
          }
        }
      }
    }
  };

} // namespace quda
