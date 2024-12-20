#include <util_quda.h>
#include <reference_wrapper_helper.h>
#include <color_spinor_field.h>
#include <array>
#include <algorithm>
#include <int_list.hpp>

namespace quda {

  template <int... Values>
  auto sort_values() {
    std::array<int, sizeof...(Values)> arr = {Values...};
    // std::sort is NOT constexpr until C++20
    std::sort(arr.begin(), arr.end());
    return arr;
  }

  inline int round_to_nearest_instantiated_nVec(int input_nVec) {
    // clang-format off
    auto sorted_nVecs = sort_values<@QUDA_MULTIGRID_MRHS_LIST@>();
    // clang-format on
    for (int nVec: sorted_nVecs) {
      if (input_nVec <= nVec) {
        return nVec;
      }
    }
    errorQuda("No instantiated nVec able to contain input nVec = %d", input_nVec);
    return 0;
  }

  template <class F> auto create_color_spinor_copy(cvector_ref<F> &fs, QudaFieldOrder order)
  {
    ColorSpinorParam param(fs[0]);
    int nVec = round_to_nearest_instantiated_nVec(fs.size());
    param.nColor = fs[0].Ncolor() * nVec;
    param.nVec = nVec;
    param.nVec_actual = fs.size();
    param.create = QUDA_NULL_FIELD_CREATE;
    param.fieldOrder = order;
    return getFieldTmp<ColorSpinorField>(param);
  }

  inline auto create_color_spinor_copy(const ColorSpinorField &f, QudaFieldOrder order)
  {
    ColorSpinorParam param(f);
    param.create = QUDA_NULL_FIELD_CREATE;
    param.fieldOrder = order;
    return getFieldTmp<ColorSpinorField>(param);
  }

}
