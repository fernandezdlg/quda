#include <string.h>
#include <multigrid.h>
#include <tune_quda.h>
#include <algorithm>
#include <transfer.h>
#include <blas_quda.h>

namespace quda {

  DiracCoarse::DiracCoarse(const DiracParam &param, bool gpu_setup) :
    Dirac(param),
    mass(param.mass),
    mu(param.mu),
    mu_factor(param.mu_factor),
    transfer(param.transfer),
    dirac(param.dirac),
    need_bidirectional(param.need_bidirectional),
    allow_truncation(param.allow_truncation),
    setup_use_mma(param.setup_use_mma),
    dslash_use_mma(param.dslash_use_mma),
    need_aos_gauge_copy(setup_use_mma || dslash_use_mma),
    enable_gpu(false),
    enable_cpu(false),
    gpu_setup(gpu_setup),
    init_gpu(gpu_setup),
    init_cpu(!gpu_setup)
  {
    initializeCoarse();
  }

  DiracCoarse::DiracCoarse(const DiracParam &param, std::shared_ptr<GaugeField> Y_h, std::shared_ptr<GaugeField> X_h,
                           std::shared_ptr<GaugeField> Xinv_h,
                           std::shared_ptr<GaugeField> Yhat_h, // cpu link fields
                           std::shared_ptr<GaugeField> Y_d, std::shared_ptr<GaugeField> X_d,
                           std::shared_ptr<GaugeField> Xinv_d,
                           std::shared_ptr<GaugeField> Yhat_d) // gpu link field
    :
    Dirac(param),
    mass(param.mass),
    mu(param.mu),
    mu_factor(param.mu_factor),
    transfer(nullptr),
    dirac(nullptr),
    need_bidirectional(false),
    allow_truncation(param.allow_truncation),
    setup_use_mma(param.setup_use_mma),
    dslash_use_mma(param.dslash_use_mma),
    need_aos_gauge_copy(setup_use_mma || dslash_use_mma),
    Y_h(Y_h),
    X_h(X_h),
    Xinv_h(Xinv_h),
    Yhat_h(Yhat_h),
    Y_d(Y_d),
    X_d(X_d),
    Xinv_d(Xinv_d),
    Yhat_d(Yhat_d),
    enable_gpu(Y_d ? true : false),
    enable_cpu(Y_h ? true : false),
    gpu_setup(true),
    init_gpu(enable_gpu ? false : true),
    init_cpu(enable_cpu ? false : true)
  {

    constexpr QudaGaugeFieldOrder gOrder = QUDA_MILC_GAUGE_ORDER;

    auto create_gauge_copy = [](const GaugeField &X) -> auto
    {
      GaugeFieldParam param(X);
      param.order = gOrder;
      auto output = std::shared_ptr<GaugeField>(GaugeField::Create(param));
      output->copy(X);
      return output;
    };

    if (need_aos_gauge_copy) {
      if (Y_d) {
        Y_aos_d = create_gauge_copy(*Y_d);
        Y_aos_d->exchangeGhost(QUDA_LINK_BIDIRECTIONAL);
      }
      if (X_d) X_aos_d = create_gauge_copy(*X_d);
      if (Xinv_d) Xinv_aos_d = create_gauge_copy(*Xinv_d);
      if (Yhat_d) {
        Yhat_aos_d = create_gauge_copy(*Yhat_d);
        Yhat_aos_d->exchangeGhost(QUDA_LINK_BIDIRECTIONAL);
      }
    }
  }

  DiracCoarse::DiracCoarse(const DiracCoarse &dirac, const DiracParam &param) :
    Dirac(param),
    mass(param.mass),
    mu(param.mu),
    mu_factor(param.mu_factor),
    transfer(param.transfer),
    dirac(param.dirac),
    need_bidirectional(param.need_bidirectional),
    allow_truncation(param.allow_truncation),
    setup_use_mma(param.setup_use_mma),
    dslash_use_mma(param.dslash_use_mma),
    need_aos_gauge_copy(setup_use_mma || dslash_use_mma),
    Y_h(dirac.Y_h),
    X_h(dirac.X_h),
    Xinv_h(dirac.Xinv_h),
    Yhat_h(dirac.Yhat_h),
    Y_d(dirac.Y_d),
    X_d(dirac.X_d),
    Y_aos_d(dirac.Y_aos_d),
    X_aos_d(dirac.X_aos_d),
    Xinv_d(dirac.Xinv_d),
    Yhat_d(dirac.Yhat_d),
    Xinv_aos_d(dirac.Xinv_aos_d),
    Yhat_aos_d(dirac.Yhat_aos_d),
    enable_gpu(dirac.enable_gpu),
    enable_cpu(dirac.enable_cpu),
    gpu_setup(dirac.gpu_setup),
    init_gpu(enable_gpu ? false : true),
    init_cpu(enable_cpu ? false : true)
  {
  }

  void DiracCoarse::createY(bool gpu) const
  {
    int ndim = transfer->Vectors().Ndim();
    lat_dim_t x;
    const int *geo_bs = transfer->Geo_bs(); // Number of coarse sites.
    for (int i = 0; i < ndim; i++) x[i] = transfer->Vectors().X(i)/geo_bs[i];
    int Nc_c = transfer->nvec(); // Coarse Color
    // Coarse Spin
    int Ns_c = (transfer->Spin_bs() == 0) ? 2 : transfer->Vectors().Nspin() / transfer->Spin_bs();
    GaugeFieldParam gParam;
    gParam.x = x;
    gParam.location = gpu ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION;
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = gpu ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    // use null-space precision for coarse links on gpu
    gParam.setPrecision( transfer->NullPrecision(gpu ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION) );
    gParam.nDim = ndim;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gParam.nFace = 1;
    gParam.geometry = QUDA_COARSE_GEOMETRY;

    int pad = std::max( { (x[0]*x[1]*x[2])/2, (x[1]*x[2]*x[3])/2, (x[0]*x[2]*x[3])/2, (x[0]*x[1]*x[3])/2 } );
    gParam.pad = gpu ? gParam.nFace * pad * 2 : 0; // factor of 2 since we have to store bi-directional ghost zone

    if (gpu) {
      Y_d = std::make_shared<GaugeField>(gParam);
      GaugeFieldParam milcParam(*Y_d);
      milcParam.order = QUDA_MILC_GAUGE_ORDER;
      if (need_aos_gauge_copy) { Y_aos_d = std::make_shared<GaugeField>(milcParam); }
    } else
      Y_h = std::make_shared<GaugeField>(gParam);

    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    gParam.pad = 0;

    if (gpu) {
      X_d = std::make_shared<GaugeField>(gParam);
      GaugeFieldParam milcParam(*X_d);
      milcParam.order = QUDA_MILC_GAUGE_ORDER;
      if (need_aos_gauge_copy) { X_aos_d = std::make_shared<GaugeField>(milcParam); }
    } else
      X_h = std::make_shared<GaugeField>(gParam);
  }

  void DiracCoarse::createYhat(bool gpu) const
  {
    int ndim = transfer->Vectors().Ndim();
    if (ndim == 5 && transfer->Vectors().Nspin() != 4) ndim = 4; // forced case for staggered, coarsened staggered
    lat_dim_t x;
    const int *geo_bs = transfer->Geo_bs(); // Number of coarse sites.
    for (int i = 0; i < ndim; i++) x[i] = transfer->Vectors().X(i)/geo_bs[i];
    int Nc_c = transfer->nvec();     // Coarse Color
    int Ns_c = (transfer->Spin_bs() == 0) ? 2 : transfer->Vectors().Nspin() / transfer->Spin_bs();

    GaugeFieldParam gParam;
    gParam.x = x;
    gParam.location = gpu ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION;
    gParam.nColor = Nc_c*Ns_c;
    gParam.reconstruct = QUDA_RECONSTRUCT_NO;
    gParam.order = gpu ? QUDA_FLOAT2_GAUGE_ORDER : QUDA_QDP_GAUGE_ORDER;
    gParam.link_type = QUDA_COARSE_LINKS;
    gParam.t_boundary = QUDA_PERIODIC_T;
    gParam.create = QUDA_ZERO_FIELD_CREATE;
    // use null-space precision for preconditioned links on gpu
    gParam.setPrecision( transfer->NullPrecision(gpu ? QUDA_CUDA_FIELD_LOCATION : QUDA_CPU_FIELD_LOCATION) );
    gParam.nDim = ndim;
    gParam.siteSubset = QUDA_FULL_SITE_SUBSET;
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_PAD;
    gParam.nFace = 1;
    gParam.geometry = QUDA_COARSE_GEOMETRY;

    int pad = std::max( { (x[0]*x[1]*x[2])/2, (x[1]*x[2]*x[3])/2, (x[0]*x[2]*x[3])/2, (x[0]*x[1]*x[3])/2 } );
    gParam.pad = gpu ? gParam.nFace * pad * 2 : 0; // factor of 2 since we have to store bi-directional ghost zone

    if (gpu) {
      Yhat_d = std::make_shared<GaugeField>(gParam);
      GaugeFieldParam milcParam(*Yhat_d);
      milcParam.order = QUDA_MILC_GAUGE_ORDER;
      if (need_aos_gauge_copy) { Yhat_aos_d = std::make_shared<GaugeField>(milcParam); }
    } else
      Yhat_h = std::make_shared<GaugeField>(gParam);

    gParam.setPrecision(gpu ? X_d->Precision() : X_h->Precision());
    gParam.ghostExchange = QUDA_GHOST_EXCHANGE_NO;
    gParam.nFace = 0;
    gParam.geometry = QUDA_SCALAR_GEOMETRY;
    gParam.pad = 0;

    if (gpu) {
      Xinv_d = std::make_shared<GaugeField>(gParam);
      GaugeFieldParam milcParam(*Xinv_d);
      milcParam.order = QUDA_MILC_GAUGE_ORDER;
      if (need_aos_gauge_copy) { Xinv_aos_d = std::make_shared<GaugeField>(milcParam); }
    } else
      Xinv_h = std::make_shared<GaugeField>(gParam);
  }

  void DiracCoarse::initializeCoarse()
  {
    createY(gpu_setup);

    if (!gpu_setup) {

      dirac->createCoarseOp(*Y_h, *X_h, *transfer, kappa, mass, Mu(), MuFactor(), AllowTruncation());
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("About to build the preconditioned coarse clover\n");

      createYhat(gpu_setup);

      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Finished building the preconditioned coarse clover\n");
      if (getVerbosity() >= QUDA_VERBOSE) printfQuda("About to create the preconditioned coarse op\n");

      calculateYhat(*Yhat_h, *Xinv_h, *Y_h, *X_h, setup_use_mma);

    } else {

      // The following fancy copies reduce the number of gauge field
      // copies (from and to QUDA_MILC_GAUGE_ORDER) by 2: one for X
      // and one for Y, both to QUDA_MILC_GAUGE_ORDER.
      if (setup_use_mma && dirac->isCoarse()) {

        dirac->createCoarseOp(*Y_aos_d, *X_aos_d, *transfer, kappa, mass, Mu(), MuFactor(), AllowTruncation());
        X_d->copy(*X_aos_d);

        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("About to build the preconditioned coarse clover\n");

        createYhat(gpu_setup);

        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Finished building the preconditioned coarse clover\n");
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("About to create the preconditioned coarse op\n");

        calculateYhat(*Yhat_d, *Xinv_d, *Y_aos_d, *X_aos_d, setup_use_mma);
        // TODO: we could pass in Yhat_aos_d and Xinv_aos_d directly
        Yhat_aos_d->copy(*Yhat_d);
        Yhat_aos_d->exchangeGhost(QUDA_LINK_BIDIRECTIONAL);
        Xinv_aos_d->copy(*Xinv_d);

        Y_d->copy(*Y_aos_d);

        // this extra exchange shouldn't be needed, but at present the
        // copy from Y_order to Y_d doesn't preserve the
        // bi-directional halo (in_offset isn't set in the copy
        // routine)
        Y_d->exchangeGhost(QUDA_LINK_BIDIRECTIONAL);

      } else {
        dirac->createCoarseOp(*Y_d, *X_d, *transfer, kappa, mass, Mu(), MuFactor(), AllowTruncation());

        if (need_aos_gauge_copy) {
          Y_aos_d->copy(*Y_d);
          Y_aos_d->exchangeGhost(QUDA_LINK_BIDIRECTIONAL);
          X_aos_d->copy(*X_d);
        }

        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("About to build the preconditioned coarse clover\n");

        createYhat(gpu_setup);

        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Finished building the preconditioned coarse clover\n");
        if (getVerbosity() >= QUDA_VERBOSE) printfQuda("About to create the preconditioned coarse op\n");

        calculateYhat(*Yhat_d, *Xinv_d, *Y_d, *X_d, setup_use_mma);

        if (need_aos_gauge_copy) {
          // TODO: we could pass in Yhat_aos_d and Xinv_aos_d directly
          Yhat_aos_d->copy(*Yhat_d);
          Yhat_aos_d->exchangeGhost(QUDA_LINK_BIDIRECTIONAL);
          Xinv_aos_d->copy(*Xinv_d);
        }
      }
    }

    if (getVerbosity() >= QUDA_VERBOSE) printfQuda("Finished creating the preconditioned coarse op\n");

    if (gpu_setup) {
      enable_gpu = true;
      init_gpu = true;
    } else {
      enable_cpu = true;
      init_cpu = true;
    }
  }

  // we only copy to host or device lazily on demand
  void DiracCoarse::initializeLazy(QudaFieldLocation location) const
  {
    if (!enable_cpu && !enable_gpu) errorQuda("Neither CPU or GPU coarse fields initialized");
    switch(location) {
    case QUDA_CUDA_FIELD_LOCATION:
      if (enable_gpu) return;
      createY(true);
      createYhat(true);
      Y_d->copy(*Y_h);
      if (need_aos_gauge_copy) { Y_aos_d->copy(*Y_d); }
      Yhat_d->copy(*Yhat_h);
      if (need_aos_gauge_copy) {
        Yhat_aos_d->copy(*Yhat_d);
        Yhat_aos_d->exchangeGhost(QUDA_LINK_BIDIRECTIONAL);
      }
      X_d->copy(*X_h);
      if (need_aos_gauge_copy) { X_aos_d->copy(*X_d); }
      Xinv_d->copy(*Xinv_h);
      if (need_aos_gauge_copy) { Xinv_aos_d->copy(*Xinv_d); }
      enable_gpu = true;
      init_gpu = true;
      break;
    case QUDA_CPU_FIELD_LOCATION:
      if (enable_cpu) return;
      createY(false);
      createYhat(false);
      Y_h->copy(*Y_d);
      Yhat_h->copy(*Yhat_d);
      X_h->copy(*X_d);
      Xinv_h->copy(*Xinv_d);
      enable_cpu = true;
      init_cpu = true;
      break;
    default:
      errorQuda("Unknown location");
    }
  }

  bool DiracCoarse::apply_mma(cvector_ref<ColorSpinorField> &f, bool use_mma) { return (f.size() > 1) && use_mma; }

  void DiracCoarse::createPreconditionedCoarseOp(GaugeField &Yhat, GaugeField &Xinv, const GaugeField &Y, const GaugeField &X) {
    calculateYhat(Yhat, Xinv, Y, X, setup_use_mma);
  }

  void DiracCoarse::Clover(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           QudaParity parity) const
  {
    QudaFieldLocation location = checkLocation(out[0], in[0]);
    initializeLazy(location);
    if (location == QUDA_CUDA_FIELD_LOCATION) {
      auto Y = apply_mma(out, dslash_use_mma) ? Y_aos_d : Y_d;
      auto X = apply_mma(out, dslash_use_mma) ? X_aos_d : X_d;
      ApplyCoarse(out, in, in, *Y, *X, kappa, parity, false, true, dagger, commDim.data, QUDA_INVALID_PRECISION,
                  dslash_use_mma);
    } else if (location == QUDA_CPU_FIELD_LOCATION) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, parity, false, true, dagger, commDim.data, QUDA_INVALID_PRECISION,
                  dslash_use_mma);
    }
  }

  void DiracCoarse::CloverInv(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                              QudaParity parity) const
  {
    QudaFieldLocation location = checkLocation(out[0], in[0]);
    initializeLazy(location);
    if ( location  == QUDA_CUDA_FIELD_LOCATION ) {
      auto Y = apply_mma(out, dslash_use_mma) ? Y_aos_d : Y_d;
      auto X = apply_mma(out, dslash_use_mma) ? Xinv_aos_d : Xinv_d;
      ApplyCoarse(out, in, in, *Y, *X, kappa, parity, false, true, dagger, commDim.data, QUDA_INVALID_PRECISION,
                  dslash_use_mma);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *Xinv_h, kappa, parity, false, true, dagger, commDim.data, QUDA_INVALID_PRECISION,
                  dslash_use_mma);
    }
  }

  void DiracCoarse::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                           QudaParity parity) const
  {
    QudaFieldLocation location = checkLocation(out[0], in[0]);
    initializeLazy(location);

    if ( location == QUDA_CUDA_FIELD_LOCATION ) {
      auto Y = apply_mma(out, dslash_use_mma) ? Y_aos_d : Y_d;
      auto X = apply_mma(out, dslash_use_mma) ? X_aos_d : X_d;
      ApplyCoarse(out, in, in, *Y, *X, kappa, parity, true, false, dagger, commDim.data, halo_precision, dslash_use_mma);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, parity, true, false, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    }
  }

  void DiracCoarse::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                               QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    if (k!=1.0) errorQuda("%s not supported for k!=1.0", __func__);

    QudaFieldLocation location = checkLocation(out[0], in[0]);
    initializeLazy(location);
    if ( location == QUDA_CUDA_FIELD_LOCATION ) {
      auto Y = apply_mma(out, dslash_use_mma) ? Y_aos_d : Y_d;
      auto X = apply_mma(out, dslash_use_mma) ? X_aos_d : X_d;
      ApplyCoarse(out, in, x, *Y, *X, kappa, parity, true, true, dagger, commDim.data, halo_precision, dslash_use_mma);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, x, *Y_h, *X_h, kappa, parity, true, true, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    }
  }

  void DiracCoarse::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    QudaFieldLocation location = checkLocation(out[0], in[0]);
    initializeLazy(location);
    if ( location == QUDA_CUDA_FIELD_LOCATION ) {
      auto Y = apply_mma(out, dslash_use_mma) ? Y_aos_d : Y_d;
      auto X = apply_mma(out, dslash_use_mma) ? X_aos_d : X_d;
      ApplyCoarse(out, in, in, *Y, *X, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    } else if ( location == QUDA_CPU_FIELD_LOCATION ) {
      ApplyCoarse(out, in, in, *Y_h, *X_h, kappa, QUDA_INVALID_PARITY, true, true, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    }
  }

  void DiracCoarse::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracCoarse::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                            cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                            const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      errorQuda("Preconditioned solution requires a preconditioned solve_type");
    }

    create_alias(src, b);
    create_alias(sol, x);
  }

  void DiracCoarse::reconstruct(cvector_ref<ColorSpinorField> &, cvector_ref<const ColorSpinorField> &,
                                const QudaSolutionType) const
  {
    /* do nothing */
  }

  //Make the coarse operator one level down.  Pass both the coarse gauge field and coarse clover field.
  void DiracCoarse::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double, double mu,
                                   double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Coarse operators only support aggregation coarsening");

    double a = 2.0 * kappa * mu * T.Vectors().TwistFlavor();
    if (checkLocation(Y, X) == QUDA_CPU_FIELD_LOCATION) {
      initializeLazy(QUDA_CPU_FIELD_LOCATION);
      CoarseCoarseOp(Y, X, T, *(this->Y_h), *(this->X_h), *(this->Xinv_h), kappa, mass, a, mu_factor, QUDA_COARSE_DIRAC,
                     QUDA_MATPC_INVALID, need_bidirectional);
    } else {
      initializeLazy(QUDA_CUDA_FIELD_LOCATION);
      if (Y.Order() != X.Order()) { errorQuda("Y/X order mismatch in createCoarseOp: %d %d\n", Y.Order(), X.Order()); }
      bool use_mma = Y.Order() == QUDA_MILC_GAUGE_ORDER;
      CoarseCoarseOp(Y, X, T, *(this->Y_d), *(this->X_d), *(this->Xinv_d), kappa, mass, a, mu_factor, QUDA_COARSE_DIRAC,
                     QUDA_MATPC_INVALID, need_bidirectional, use_mma);
    }
  }

  void DiracCoarse::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    if (Y_d) Y_d->prefetch(mem_space, stream);
    if (X_d) X_d->prefetch(mem_space, stream);
  }

  DiracCoarsePC::DiracCoarsePC(const DiracParam &param, bool gpu_setup) : DiracCoarse(param, gpu_setup)
  {
    /* do nothing */
  }

  DiracCoarsePC::DiracCoarsePC(const DiracParam &param, std::shared_ptr<GaugeField> Y_h, std::shared_ptr<GaugeField> X_h,
                               std::shared_ptr<GaugeField> Xinv_h, std::shared_ptr<GaugeField> Yhat_h,
                               std::shared_ptr<GaugeField> Y_d, std::shared_ptr<GaugeField> X_d,
                               std::shared_ptr<GaugeField> Xinv_d, std::shared_ptr<GaugeField> Yhat_d) :
    DiracCoarse(param, Y_h, X_h, Xinv_h, Yhat_h, Y_d, X_d, Xinv_d, Yhat_d)
  {
  }

  DiracCoarsePC::DiracCoarsePC(const DiracCoarse &dirac, const DiracParam &param) : DiracCoarse(dirac, param)
  {
    /* do nothing */
  }

  void DiracCoarsePC::Dslash(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                             QudaParity parity) const
  {
    QudaFieldLocation location = checkLocation(out[0], in[0]);
    initializeLazy(location);

    if (location == QUDA_CUDA_FIELD_LOCATION) {
      auto Y = apply_mma(out, dslash_use_mma) ? Yhat_aos_d : Yhat_d;
      auto X = apply_mma(out, dslash_use_mma) ? X_aos_d : X_d;
      ApplyCoarse(out, in, in, *Y, *X, kappa, parity, true, false, dagger, commDim.data, halo_precision, dslash_use_mma);
    } else if (location == QUDA_CPU_FIELD_LOCATION) {
      ApplyCoarse(out, in, in, *Yhat_h, *X_h, kappa, parity, true, false, dagger, commDim.data, halo_precision,
                  dslash_use_mma);
    }
  }

  void DiracCoarsePC::DslashXpay(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in,
                                 QudaParity parity, cvector_ref<const ColorSpinorField> &x, double k) const
  {
    // FIXME emulated for now
    Dslash(out, in, parity);
    blas::xpay(x, k, out);
  }

  void DiracCoarsePC::M(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out);

    if (in.SiteSubset() == QUDA_FULL_SITE_SUBSET || out.SiteSubset() == QUDA_FULL_SITE_SUBSET)
      errorQuda("Cannot apply preconditioned operator to full field (subsets = %d %d)", in.SiteSubset(),
                out.SiteSubset());

    if (matpcType == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC) {
      // DiracCoarsePC::Dslash applies A^{-1}Dslash
      Dslash(tmp, in, QUDA_ODD_PARITY);
      // DiracCoarse::DslashXpay applies (A - D) // FIXME this ignores the -1
      DiracCoarse::Dslash(out, tmp, QUDA_EVEN_PARITY);
      Clover(tmp, in, QUDA_EVEN_PARITY);
      blas::xpay(tmp, -1.0, out);
    } else if (matpcType == QUDA_MATPC_ODD_ODD_ASYMMETRIC) {
      // DiracCoarsePC::Dslash applies A^{-1}Dslash
      Dslash(tmp, in, QUDA_EVEN_PARITY);
      // DiracCoarse::DslashXpay applies (A - D) // FIXME this ignores the -1
      DiracCoarse::Dslash(out, tmp, QUDA_ODD_PARITY);
      Clover(tmp, in, QUDA_ODD_PARITY);
      blas::xpay(tmp, -1.0, out);
    } else if (matpcType == QUDA_MATPC_EVEN_EVEN) {
      Dslash(tmp, in, QUDA_ODD_PARITY);
      DslashXpay(out, tmp, QUDA_EVEN_PARITY, in, -1.0);
    } else if (matpcType == QUDA_MATPC_ODD_ODD) {
      Dslash(tmp, in, QUDA_EVEN_PARITY);
      DslashXpay(out, tmp, QUDA_ODD_PARITY, in, -1.0);
    } else {
      errorQuda("MatPCType %d not valid for DiracCoarsePC", matpcType);
    }
  }

  void DiracCoarsePC::MdagM(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in) const
  {
    auto tmp = getFieldTmp(out);
    M(tmp, in);
    Mdag(out, tmp);
  }

  void DiracCoarsePC::prepare(cvector_ref<ColorSpinorField> &sol, cvector_ref<ColorSpinorField> &src,
                              cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                              const QudaSolutionType solType) const
  {
    // we desire solution to preconditioned system
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) {
      create_alias(src, b);
      create_alias(sol, x);
      return;
    }

    create_alias(src, x(other_parity));
    create_alias(sol, x(this_parity));

    auto tmp = getFieldTmp(x.Even());
    // we desire solution to full system
    if (symmetric) {
      // src = A_ee^-1 (b_e - D_eo A_oo^-1 b_o)
#if 0
      CloverInv(src, b(other_parity), other_parity);
      DiracCoarse::Dslash(tmp, src, this_parity);
      blas::xpay(b(this_parity), -1.0, tmp);
      CloverInv(src, tmp, this_parity);
#else
      // src = A_ee^{-1} b_e - (A_ee^{-1} D_eo) A_oo^{-1} b_o
      CloverInv(src, b(other_parity), other_parity);
      Dslash(tmp, src, this_parity);
      CloverInv(src, b(this_parity), this_parity);
      blas::axpy(-1.0, tmp, src);
#endif
    } else {
      // src = b_e - D_eo A_oo^-1 b_o
      CloverInv(tmp, b(other_parity), other_parity);
      DiracCoarse::Dslash(src, tmp, this_parity);
      blas::xpay(b(this_parity), -1.0, src);
    }
  }

  void DiracCoarsePC::reconstruct(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b,
                                  const QudaSolutionType solType) const
  {
    if (solType == QUDA_MATPC_SOLUTION || solType == QUDA_MATPCDAG_MATPC_SOLUTION) return;

    checkFullSpinor(x, b);
    auto tmp = getFieldTmp(x.Even());
#if 0
    // x_o = A_oo^-1 (b_o - D_oe x_e)
    DiracCoarse::Dslash(tmp, x(this_parity), other_parity);
    blas::xpay(b(other_parity), -1.0, tmp);
    CloverInv(x(other_parity), tmp, other_parity);
#else
    // x_o = A_oo^{-1} b_o - (A_oo^{-1} D_oe) x_e
    Dslash(tmp, x(this_parity), other_parity);
    CloverInv(x(other_parity), b(other_parity), other_parity);
    blas::axpy(-1.0, tmp, x(other_parity));
#endif
  }

  //Make the coarse operator one level down.  For the preconditioned
  //operator we are coarsening the Yhat links, not the Y links.  We
  //pass the fine clover fields, though they are actually ignored.
  void DiracCoarsePC::createCoarseOp(GaugeField &Y, GaugeField &X, const Transfer &T, double kappa, double, double mu,
                                     double mu_factor, bool) const
  {
    if (T.getTransferType() != QUDA_TRANSFER_AGGREGATE)
      errorQuda("Coarse operators only support aggregation coarsening");

    double a = -2.0 * kappa * mu * T.Vectors().TwistFlavor();
    if (checkLocation(Y, X) == QUDA_CPU_FIELD_LOCATION) {
      initializeLazy(QUDA_CPU_FIELD_LOCATION);
      CoarseCoarseOp(Y, X, T, *(this->Yhat_h), *(this->X_h), *(this->Xinv_h), kappa, mass, a, -mu_factor,
                     QUDA_COARSEPC_DIRAC, matpcType, true);
    } else {
      initializeLazy(QUDA_CUDA_FIELD_LOCATION);
      if (Y.Order() != X.Order()) { errorQuda("Y/X order mismatch in createCoarseOp: %d %d\n", Y.Order(), X.Order()); }
      bool use_mma = Y.Order() == QUDA_MILC_GAUGE_ORDER;
      CoarseCoarseOp(Y, X, T, *(this->Yhat_d), *(this->X_d), *(this->Xinv_d), kappa, mass, a, -mu_factor,
                     QUDA_COARSEPC_DIRAC, matpcType, true, use_mma);
    }
  }

  void DiracCoarsePC::prefetch(QudaFieldLocation mem_space, qudaStream_t stream) const
  {
    Dirac::prefetch(mem_space, stream);
    if (Xinv_d) Xinv_d->prefetch(mem_space, stream);
    if (Yhat_d) Yhat_d->prefetch(mem_space, stream);
  }
}
