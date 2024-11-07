#include <tune_quda.h>
#include <int_factor_array.hpp>

namespace quda {

/**
    @brief Thie helper class instantiates the following mapping:
        tp.aux.x -> Bx in x_atom_size * [factors of (x + x_atom_size - 1) / x_atom_size];
        tp.aux.y -> By in y_atom_size * [factors of (y + y_atom_size - 1) / y_atom_size];
        tp.aux.z -> Bz in z_atom_size * [factors of (z + z_atom_size - 1) / z_atom_size];
        tp.aux.w -> Bw in w_atom_size * [factors of (w + w_atom_size - 1) / w_atom_size].
      See `void expand(TuneParam &tp, const qudaStream_t &stream)`
 */
template <class Callable, int x, int x_atom_size, int y, int y_atom_size, int z, int z_atom_size, int w, int w_atom_size>
class expand_aux_t {

  Callable &_callable;

  template <int Bx, int By, int Bz, size_t W, size_t... Ws>
    void span_w(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<W, Ws...>)
    {
      if (tp.aux.w == W) {
        constexpr IntFactorArray<(w + w_atom_size - 1) / w_atom_size> w_factors;
        _callable.template launch_mma<Bx, By, Bz, w_atom_size * w_factors[W]>(tp, stream);
      } else {
        if constexpr (sizeof...(Ws) > 0) {
          span_w<Bx, By, Bz>(tp, stream, std::index_sequence<Ws...>());
        } else {
          errorQuda("Invalid tp.aux.w");
        }
      }
    }

  template <int Bx, int By, size_t Z, size_t... Zs>
    void span_z(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<Z, Zs...>)
    {
      if (tp.aux.z == Z) {
        constexpr IntFactorArray<(z + z_atom_size - 1) / z_atom_size> z_factors;
        std::make_index_sequence<IntFactorArray<(w + w_atom_size - 1) / w_atom_size>().size()> w_indices;
        span_w<Bx, By, z_atom_size * z_factors[Z]>(tp, stream, w_indices);
      } else {
        if constexpr (sizeof...(Zs) > 0) {
          span_z<Bx, By>(tp, stream, std::index_sequence<Zs...>());
        } else {
          errorQuda("Invalid tp.aux.z");
        }
      }
    }

  template <int Bx, size_t Y, size_t... Ys>
    void span_y(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<Y, Ys...>)
    {
      if (tp.aux.y == Y) {
        constexpr IntFactorArray<(y + y_atom_size - 1) / y_atom_size> y_factors;
        std::make_index_sequence<IntFactorArray<(z + z_atom_size - 1) / z_atom_size>().size()> z_indices;
        span_z<Bx, y_atom_size * y_factors[Y]>(tp, stream, z_indices);
      } else {
        if constexpr (sizeof...(Ys) > 0) {
          span_y<Bx>(tp, stream, std::index_sequence<Ys...>());
        } else {
          errorQuda("Invalid tp.aux.y");
        }
      }
    }

  template <size_t X, size_t... Xs>
    void span_x(TuneParam &tp, const qudaStream_t &stream, std::index_sequence<X, Xs...>)
    {
      if (tp.aux.x == X) {
        constexpr IntFactorArray<(x + x_atom_size - 1) / x_atom_size> x_factors;
        std::make_index_sequence<IntFactorArray<(y + y_atom_size - 1) / y_atom_size>().size()> y_indices;
        span_y<x_atom_size * x_factors[X]>(tp, stream, y_indices);
      } else {
        if constexpr (sizeof...(Xs) > 0) {
          span_x(tp, stream, std::index_sequence<Xs...>());
        } else {
          errorQuda("Invalid tp.aux.x");
        }
      }
    }

 public:

    /**
        @brief invoke `_callable.template launch_mma<Bx, By, Bz, Bw>(tp, stream);` based on the tp.aux values
            tp.aux.x -> Bx in x_atom_size * [factors of (x + x_atom_size - 1) / x_atom_size];
            tp.aux.y -> By in y_atom_size * [factors of (y + y_atom_size - 1) / y_atom_size];
            tp.aux.z -> Bz in z_atom_size * [factors of (z + z_atom_size - 1) / z_atom_size];
            tp.aux.w -> Bw in w_atom_size * [factors of (w + w_atom_size - 1) / w_atom_size].
          For example, if x_atom_size = 8, x = 48, then Bx can take values in [8, 16, 24, 48]; when tp.aux.x == 0,
          Bx = 8; when tp.aux.x == 1, Bx = 16; when tp.aux.x == 2, Bx = 24; when tp.aux.x == 3, Bx = 48.
        @param tp The TuneParam parameter
        @param stream The stream parameter
     */
    void expand(TuneParam &tp, const qudaStream_t &stream)
    {
      std::make_index_sequence<IntFactorArray<(x + x_atom_size - 1) / x_atom_size>().size()> x_indices;
      span_x(tp, stream, x_indices);
    }

    expand_aux_t(Callable &callable): _callable(callable) { }

    /**
        @brief Get the Bx value
        @param tp The TuneParam parameter
     */
    int get_x(const TuneParam &tp) const {
      if (static_cast<unsigned int>(tp.aux.x) >= IntFactorArray<(x + x_atom_size - 1) / x_atom_size>().size()) {
        errorQuda("Invalid tp.aux.x = %d\n", tp.aux.x);
      }
      return x_atom_size * get_int_factor_array((x + x_atom_size - 1) / x_atom_size)[tp.aux.x];
    }

    /**
        @brief Get the By value
        @param tp The TuneParam parameter
     */
    int get_y(const TuneParam &tp) const {
      if (static_cast<unsigned int>(tp.aux.y) >= IntFactorArray<(y + y_atom_size - 1) / y_atom_size>().size()) {
        errorQuda("Invalid tp.aux.y = %d\n", tp.aux.y);
      }
      return y_atom_size * get_int_factor_array((y + y_atom_size - 1) / y_atom_size)[tp.aux.y];
    }

    /**
        @brief Get the Bz value
        @param tp The TuneParam parameter
     */
    int get_z(const TuneParam &tp) const {
      if (static_cast<unsigned int>(tp.aux.z) >= IntFactorArray<(z + z_atom_size - 1) / z_atom_size>().size()) {
        errorQuda("Invalid tp.aux.z = %d\n", tp.aux.z);
      }
      return z_atom_size * get_int_factor_array((z + z_atom_size - 1) / z_atom_size)[tp.aux.z];
    }

    /**
        @brief Get the Bw value
        @param tp The TuneParam parameter
     */
    int get_w(const TuneParam &tp) const {
      if (static_cast<unsigned int>(tp.aux.w) >= IntFactorArray<(w + w_atom_size - 1) / w_atom_size>().size()) {
        errorQuda("Invalid tp.aux.w = %d\n", tp.aux.w);
      }
      return w_atom_size * get_int_factor_array((w + w_atom_size - 1) / w_atom_size)[tp.aux.w];
    }

    /**
        @brief Advance to the next possible aux value and return true; return false we have gone to the last
          possible value
        @return whether or not an advance is performed
        @param tp The TuneParam parameter
     */
    bool advance_aux(TuneParam &param) const
    {
      auto advancer = [&](int &i, int limit) -> bool {
        if (i < limit) {
          i++;
          return _callable.set_mma_param(param);
        } else {
          return false;
        }
      };

      if (advancer(param.aux.x, numFactors((x + x_atom_size - 1) / x_atom_size) - 1)) {
        return true;
      } else {
        param.aux.x = 0;
        if (advancer(param.aux.y, numFactors((y + y_atom_size - 1) / y_atom_size) - 1)) {
          return true;
        } else {
          param.aux.y = 0;
          if (advancer(param.aux.z, numFactors((z + z_atom_size - 1) / z_atom_size) - 1)) {
            return true;
          } else {
            param.aux.z = 0;
            if (advancer(param.aux.w, numFactors((w + w_atom_size - 1) / w_atom_size) - 1)) {
              return true;
            } else {
              param.aux.w = 0;
              return false;
            }
          }
        }
      }
    }

    /**
        @brief Initialize aux
        @param tp The TuneParam parameter
     */
    void init_aux(TuneParam &param) const {
      param.aux.x = 0;
      param.aux.y = 0;
      param.aux.z = 0;
      param.aux.w = 0;
    }

};

}
