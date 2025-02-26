#include <color_spinor_field_order.h>
#include <kernel.h>

#define PRESERVE_SPINOR_NORM

#ifdef PRESERVE_SPINOR_NORM // Preserve the norm regardless of basis
#define kP (1.0 / sqrt(2.0))
#define kU (1.0 / sqrt(2.0))
#else // More numerically accurate not to preserve the norm between basis
#define kP (0.5)
#define kU (1.0)
#endif

namespace quda
{

  using namespace colorspinor;

  /**
   * @brief A helper function to figure out what parity to use for input and output.
   * @details Pick parity from input field site order. Addditionally QDPJIT fields
   *  may need a relative parity flip compared to what is expected when dealing with
   *  only the odd parity since the pointer is always to the top of the full spinort.
   * @param[in] f Reference to the field for parity computation
   * @return the computed parity
   */
  inline int computeParity(const ColorSpinorField &f)
  {

    // Account for odd-even vs. even-odd site orders
    int ret_val = f.SiteOrder() == QUDA_ODD_EVEN_SITE_ORDER ? 1 : 0;

    // Account for potential parity flip to access single parity subset QDP-JIT fields
    // The Flip is only needed fir offsetting into Odd Parity Fields
    if (f.FieldOrder() == QUDA_QDPJIT_FIELD_ORDER && f.SiteSubset() == QUDA_PARITY_SITE_SUBSET
        && f.SuggestedParity() == QUDA_ODD_PARITY) {
      ret_val = 1 - ret_val;
    }

    return ret_val;
  }

  template <typename FloatOut, typename FloatIn, int nSpin_, int nColor_, typename Out, typename In,
            template <int, int> class Basis_>
  struct CopyColorSpinorArg : kernel_param<> {
    using Basis = Basis_<nSpin_, nColor_>;
    using realOut = typename mapper<FloatOut>::type;
    using realIn = typename mapper<std::remove_const_t<FloatIn>>::type;
    static constexpr int nSpin = nSpin_;
    static constexpr int nColor = nColor_;
    Out out;
    const In in;
    const int outParity;
    const int inParity;
    CopyColorSpinorArg(ColorSpinorField &out, const ColorSpinorField &in, FloatOut *Out_, const FloatIn *In_) :
      kernel_param(dim3(in.VolumeCB(), in.SiteSubset(), 1)),
      out(out, 1, Out_),
      in(in, 1, const_cast<FloatIn *>(In_)),
      outParity(computeParity(out)),
      inParity(computeParity(in))
    {
    }
  };

  /** Straight copy with no basis change */
  template <int Ns, int Nc> struct PreserveBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      for (int s = 0; s < Ns; s++)
        for (int c = 0; c < Nc; c++) out[s * Nc + c] = in[s * Nc + c];
    }
  };

  /** Transform from relativistic Degrand-Rossi into non-relativistic UKQCD basis */
  template <int Ns, int Nc> struct NonRelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      int s1[4] = {1, 2, 3, 0};
      int s2[4] = {3, 0, 1, 2};
      FloatOut K1[4] = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP),
                        static_cast<FloatOut>(-kP)};
      FloatOut K2[4]
        = {static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) {
          out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c])
            + K2[s] * static_cast<complex<FloatOut>>(in[s2[s] * Nc + c]);
        }
      }
    }
  };

  /** Transform from non-relativistic UKQCD into relativistic Degrand-Rossi basis */
  template <int Ns, int Nc> struct RelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      int s1[4] = {1, 2, 3, 0};
      int s2[4] = {3, 0, 1, 2};
      FloatOut K1[4]
        = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU)};
      FloatOut K2[4] = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(-kU),
                        static_cast<FloatOut>(-kU)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) {
          out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c])
            + K2[s] * static_cast<complex<FloatOut>>(in[s2[s] * Nc + c]);
        }
      }
    }
  };

  /** Transform from non-relativistic Dirac-Pauli into relativistic Degrand-Rossi basis */
  template <int Ns, int Nc> struct DegrandRossiToDiracPaulBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      int s1[4] = {1, 2, 1, 0};
      int s2[4] = {3, 0, 3, 2};
      FloatOut K1[4]
        = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(-kU)};
      FloatOut K2[4]
        = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) {
          out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c])
            + K2[s] * static_cast<complex<FloatOut>>(in[s2[s] * Nc + c]);
        }
      }
    }
  };

  /** Transform from relativistic Degrand-Rossi into non-relativistic Dirac-Pauli basis */
  template <int Ns, int Nc> struct DiracPaulToDegrandRossiBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      int s1[4] = {1, 2, 1, 0};
      int s2[4] = {3, 0, 3, 2};
      FloatOut K1[4]
        = {static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(-kP)};
      FloatOut K2[4] = {static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP),
                        static_cast<FloatOut>(-kP)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) {
          out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c])
            + K2[s] * static_cast<complex<FloatOut>>(in[s2[s] * Nc + c]);
        }
      }
    }
  };

  /** Transform from chiral into UKQCD non-relativistic basis */
  template <int Ns, int Nc> struct ChiralToNonRelBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      int s1[4] = {0, 1, 0, 1};
      int s2[4] = {2, 3, 2, 3};
      FloatOut K1[4]
        = {static_cast<FloatOut>(-kP), static_cast<FloatOut>(-kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      FloatOut K2[4]
        = {static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP), static_cast<FloatOut>(kP)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) {
          out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c])
            + K2[s] * static_cast<complex<FloatOut>>(in[s2[s] * Nc + c]);
        }
      }
    }
  };

  /** Transform from UKQCD non-relativistic into chiral basis */
  template <int Ns, int Nc> struct NonRelToChiralBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      int s1[4] = {0, 1, 0, 1};
      int s2[4] = {2, 3, 2, 3};
      FloatOut K1[4]
        = {static_cast<FloatOut>(-kU), static_cast<FloatOut>(-kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU)};
      FloatOut K2[4]
        = {static_cast<FloatOut>(kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU), static_cast<FloatOut>(kU)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) {
          out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c])
            + K2[s] * static_cast<complex<FloatOut>>(in[s2[s] * Nc + c]);
        }
      }
    }
  };

  /** Transform from chiral into DeGrand-Rossi basis or from DeGrand-Rossi into chiral basis */
  template <int Ns, int Nc> struct ChiralToFromDegrandRossiBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      int s1[4] = {3, 2, 1, 0};
      FloatOut K1[4] = {static_cast<FloatOut>(-1.0), static_cast<FloatOut>(1.0), static_cast<FloatOut>(1.0),
                        static_cast<FloatOut>(-1.0)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
      }
    }
  };

  /** Transform from UKQCD to Dirac-Pauli and from Dirac-Pauli into UKQCD basis */
  template <int Ns, int Nc> struct UKQCDToFromDiracPauliBasis {
    template <typename FloatOut, typename FloatIn>
    __device__ __host__ inline void operator()(complex<FloatOut> out[Ns * Nc], const complex<FloatIn> in[Ns * Nc]) const
    {
      int s1[4] = {0, 1, 2, 3};
      FloatOut K1[4] = {static_cast<FloatOut>(-1.0), static_cast<FloatOut>(-1.0), static_cast<FloatOut>(1.0),
                        static_cast<FloatOut>(1.0)};
      for (int s = 0; s < Ns; s++) {
        for (int c = 0; c < Nc; c++) { out[s * Nc + c] = K1[s] * static_cast<complex<FloatOut>>(in[s1[s] * Nc + c]); }
      }
    }
  };

  template <typename Arg> struct CopyColorSpinor_ {
    const Arg &arg;
    constexpr CopyColorSpinor_(const Arg &arg) : arg(arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; }

    __device__ __host__ inline void operator()(int x_cb, int parity)
    {
      ColorSpinor<typename Arg::realIn, Arg::nColor, Arg::nSpin> in = arg.in(x_cb, (parity + arg.inParity) & 1);
      ColorSpinor<typename Arg::realOut, Arg::nColor, Arg::nSpin> out;
      typename Arg::Basis basis;
      basis(out.data, in.data);
      arg.out(x_cb, (parity + arg.outParity) & 1) = out;
    }
  };

} // namespace quda
