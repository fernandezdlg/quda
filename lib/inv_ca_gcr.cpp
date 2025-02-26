#include <invert_quda.h>
#include <blas_quda.h>
#include <eigen_helper.h>
#include <solver.hpp>

namespace quda
{

  CAGCR::CAGCR(const DiracMatrix &mat, const DiracMatrix &matSloppy, const DiracMatrix &matPrecon,
               const DiracMatrix &matEig, SolverParam &param) :
    Solver(mat, matSloppy, matPrecon, matEig, param),
    matMdagM(matEig.Expose()),
    init(false),
    lambda_init(false),
    basis(param.ca_basis)
  {
  }

  CAGCR::~CAGCR()
  {
    destroyDeflationSpace();
  }

  void CAGCR::create(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    Solver::create(x, b);

    if (!init || r.size() != b.size()) {
      if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_INIT);

      alpha.resize(b.size());
      for (auto i = 0u; i < alpha.size(); i++) alpha[i].resize(param.Nkrylov);

      bool mixed = param.precision != param.precision_sloppy;

      ColorSpinorParam csParam(b[0]);
      csParam.setPrecision(param.precision_sloppy);

      if (basis == QUDA_POWER_BASIS) {
        // in power basis q[k] = p[k+1], so we don't need a separate q array
        p.resize(param.Nkrylov + 1);
        q.resize(param.Nkrylov);
        for (int i = 0; i < param.Nkrylov + 1; i++) {
          resize(p[i], b.size(), csParam);
          if (i > 0) create_alias(q[i - 1], p[i]);
        }
      } else {
        p.resize(param.Nkrylov);
        q.resize(param.Nkrylov);
        for (int i = 0; i < param.Nkrylov; i++) {
          resize(p[i], b.size(), csParam);
          resize(q[i], b.size(), csParam);
        }
      }

      csParam.setPrecision(param.precision);
      if (mixed)
        resize(r, b.size(), csParam);
      else
        create_alias(r, p[0]);

      if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_INIT);
      init = true;
    } // init
  }

  void CAGCR::solve(std::vector<Complex> &psi_, cvector_ref<ColorSpinorField> &q, ColorSpinorField &b)
  {
    typedef Matrix<Complex, Dynamic, Dynamic> matrix;
    typedef Matrix<Complex, Dynamic, 1> vector;

    const int N = q.size();
    vector phi(N), psi(N);
    matrix A(N, N);

#if 1
    // only a single reduction but requires using the full dot product
    // compute rhs vector phi = Q* b = (q_i, b)

    // Construct the matrix Q* Q = (A P)* (A P) = (q_i, q_j) = (A p_i, A p_j)
    std::vector<Complex> A_(N * (N + 1));

    blas::block::cDotProduct(A_, q, {q, b});
    for (int i = 0; i < N; i++) {
      phi(i) = A_[i * (N + 1) + N];
      for (int j = 0; j < N; j++) { A(i, j) = A_[i * (N + 1) + j]; }
    }
#else
    // two reductions but uses the Hermitian block dot product
    // compute rhs vector phi = Q* b = (q_i, b)
    std::vector<Complex> phi_(N);
    blas::block::cDotProduct(phi_, q, b);
    for (int i = 0; i < N; i++) phi(i) = phi_[i];

    // Construct the matrix Q* Q = (A P)* (A P) = (q_i, q_j) = (A p_i, A p_j)
    std::vector<Complex> A_(N * N);
    blas::block::hDotProduct(A_, q, q);
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) A(i, j) = A_[i * N + j];
#endif

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      getProfile().TPSTART(QUDA_PROFILE_EIGEN);
    }

    // use Cholesky LDL since this seems plenty stable
    LDLT<matrix> cholesky(A);
    psi = cholesky.solve(phi);

    for (int i = 0; i < N; i++) psi_[i] = psi(i);

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_EIGEN);
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    }
  }

  cvector_ref<const ColorSpinorField> CAGCR::get_residual()
  {
    if (!init) errorQuda("No residual vector present");
    if (!param.return_residual) errorQuda("SolverParam::return_residual not enabled");
    return r;
  }

  /*
    The main CA-GCR algorithm, which consists of three main steps:
    1. Build basis vectors q_k = A p_k for k = 1..Nkrlylov
    2. Minimize the residual in this basis
    3. Update solution and residual vectors
    4. (Optional) restart if convergence or maxiter not reached
  */
  void CAGCR::operator()(cvector_ref<ColorSpinorField> &x, cvector_ref<const ColorSpinorField> &b)
  {
    const int n_krylov = param.Nkrylov;

    if (param.maxiter == 0 || n_krylov == 0) {
      if (param.use_init_guess == QUDA_USE_INIT_GUESS_NO) blas::zero(x);
      return;
    }

    create(x, b);

    if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);
    if (param.is_preconditioner) commGlobalReductionPush(param.global_reduction);

    // compute b2, but only if we need to
    bool fixed_iteration = param.sloppy_converge && n_krylov == param.maxiter && !param.compute_true_res;
    auto b2 = !fixed_iteration ? blas::norm2(b) : vector<double>(b.size(), 1.0);
    std::vector<double> r2(b.size(), 0.0); // if zero source then we will exit immediately doing no work

    if (param.deflate) {
      // Construct the eigensolver and deflation space if requested.
      if (param.eig_param.eig_type == QUDA_EIG_TR_LANCZOS || param.eig_param.eig_type == QUDA_EIG_BLK_TR_LANCZOS) {
        constructDeflationSpace(b[0], matMdagM);
      } else {
        // Use Arnoldi to inspect the space only and turn off deflation
        constructDeflationSpace(b[0], mat);
        param.deflate = false;
      }
      if (deflate_compute) {
        // compute the deflation space.
        if (!param.is_preconditioner) getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
        (*eig_solve)(evecs, evals);
        if (param.deflate) {
          // double the size of the Krylov space
          extendSVDDeflationSpace();
          // populate extra memory with L/R singular vectors
          eig_solve->computeSVD(evecs, evals);
        }
        if (!param.is_preconditioner) getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);
        deflate_compute = false;
      }
      if (recompute_evals) {
        eig_solve->computeEvals(evecs, evals);
        eig_solve->computeSVD(evecs, evals);
        recompute_evals = false;
      }
    }

    // compute intitial residual depending on whether we have an initial guess or not
    if (param.use_init_guess == QUDA_USE_INIT_GUESS_YES) {
      mat(r, x);
      // r = b - Ax0
      if (!fixed_iteration) {
        r2 = blas::xmyNorm(b, r);
        for (auto i = 0u; i < b.size(); i++)
          if (b2[i] == 0) b2[i] = r2[i];
      } else {
        blas::xpay(b, -1.0, r);
        r2 = b2; // dummy setting
      }
    } else {
      r2 = b2;
      blas::copy(r, b);
      blas::zero(x);
    }

    if (param.deflate && param.maxiter > 1) {
      // Deflate: Hardcoded to SVD. If maxiter == 1, this is a dummy solve
      eig_solve->deflateSVD(x, r, evecs, evals, true);

      // Compute r_defl = RHS - A * LHS
      mat(r, x);
      if (!fixed_iteration) {
        r2 = blas::xmyNorm(b, r);
      } else {
        blas::xpay(b, -1.0, r);
        r2 = b2; // dummy setting
      }
    }

    // Use power iterations to approx lambda_max
    auto &lambda_min = param.ca_lambda_min;
    auto &lambda_max = param.ca_lambda_max;

    if (basis == QUDA_CHEBYSHEV_BASIS && n_krylov > 1 && lambda_max < lambda_min && !lambda_init) {
      if (!param.is_preconditioner) {
        getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
        getProfile().TPSTART(QUDA_PROFILE_INIT);
      }

      // Perform 100 power iterations, normalizing every 10 mat-vecs, using r_ as an initial seed
      // and q[0]/q[1] as temporaries for the power iterations. Technically illegal if n_krylov == 1, but in that case lambda_max isn't used anyway.
      lambda_max = 1.1 * Solver::performPowerIterations(matSloppy, r[0], q[0][0], q[1][0], 100, 10);
      logQuda(QUDA_SUMMARIZE, "CA-GCR Approximate lambda max = 1.1 x %e\n", lambda_max / 1.1);

      lambda_init = true;

      if (!param.is_preconditioner) {
        getProfile().TPSTOP(QUDA_PROFILE_INIT);
        getProfile().TPSTART(QUDA_PROFILE_PREAMBLE);
      }
    }

    // Factors which map linear operator onto [-1,1]
    double m_map = 2. / (lambda_max - lambda_min);
    double b_map = -(lambda_max + lambda_min) / (lambda_max - lambda_min);

    // Check to see that we're not trying to invert on a zero-field source
    if (is_zero_src(x, b, b2)) {
      getProfile().TPSTOP(QUDA_PROFILE_INIT);
      return;
    }

    auto stop = !fixed_iteration ? stopping(param.tol, b2, param.residual_type) :
                                   std::vector(b.size(), 0.0); // stopping condition of solver

    const bool use_heavy_quark_res = (param.residual_type & QUDA_HEAVY_QUARK_RESIDUAL) ? true : false;

    // this parameter determines how many consective reliable update
    // reisudal increases we tolerate before terminating the solver,
    // i.e., how long do we want to keep trying to converge
    const int maxResIncrease = param.max_res_increase; // check if we reached the limit of our tolerance
    const int maxResIncreaseTotal = param.max_res_increase_total;

    std::vector<double> heavy_quark_res(b.size(), 0.0); // heavy quark residual
    if (use_heavy_quark_res) {
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
    }

    int resIncrease = 0;
    int resIncreaseTotal = 0;

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_PREAMBLE);
      getProfile().TPSTART(QUDA_PROFILE_COMPUTE);
    }
    int total_iter = 0;
    int restart = 0;
    auto r2_old = r2;
    auto maxr_deflate = sqrt(r2[0]);
    bool l2_converge = false;

    blas::copy(p[0], r); // no op if uni-precision

    auto get_i = [](std::vector<std::vector<ColorSpinorField>> &p, int i) {
      vector_ref<ColorSpinorField> p_i;
      p_i.reserve(p.size());
      for (auto &pi : p) p_i.push_back(pi[i]);
      return p_i;
    };

    PrintStats("CA-GCR", total_iter, r2, b2, heavy_quark_res);
    while (!convergenceL2(r2, stop) && total_iter < param.maxiter) {

      // build up a space of size n_krylov
      computeCAKrylovSpace(matSloppy, q, p, n_krylov, basis, m_map, b_map);

      for (auto i = 0u; i < b.size(); i++) solve(alpha[i], get_i(q, i), p[0][i]);

      // need to make sure P is only length n_krylov
      for (auto i = 0u; i < b.size(); i++) {
        auto pi = get_i(p, i);
        blas::block::caxpy(alpha[i], {pi.begin(), pi.begin() + n_krylov}, {x[i]});
      }

      // no need to compute residual vector if not returning
      // residual vector and only doing a single fixed iteration
      if (!fixed_iteration || param.return_residual) {
        for (auto i = 0u; i < b.size(); i++) {
          // update the residual vector
          for (int j = 0; j < n_krylov; j++) alpha[i][j] = -alpha[i][j];
          blas::block::caxpy(alpha[i], get_i(q, i), r[i]);
        }
      }

      total_iter += n_krylov;
      if (!fixed_iteration || getVerbosity() >= QUDA_DEBUG_VERBOSE) {
        // only compute the residual norm if we need to
        r2 = blas::norm2(r);
      }

      PrintStats("CA-GCR", total_iter, r2, b2, heavy_quark_res);

      // update since n_krylov or maxiter reached, converged or reliable update required
      // note that the heavy quark residual will by definition only be checked every n_krylov steps
      if (total_iter >= param.maxiter || (r2 < stop && !l2_converge) || sqrt(r2[0] / r2_old[0]) < param.delta) {

        if ((r2 < stop || total_iter >= param.maxiter) && param.sloppy_converge) break;
        mat(r, x);
        r2 = blas::xmyNorm(b, r);

        if (param.deflate && sqrt(r2[0]) < maxr_deflate * param.tol_restart) {
          // Deflate and accumulate to solution vector
          eig_solve->deflateSVD(x, r, evecs, evals, true);

          // Compute r_defl = RHS - A * LHS
          mat(r, x);
          r2 = blas::xmyNorm(b, r);

          maxr_deflate = sqrt(r2[0]);
        }

        if (use_heavy_quark_res) {
          auto hq = blas::HeavyQuarkResidualNorm(x, r);
          for (auto i = 0u; i < b.size(); i++) heavy_quark_res[i] = sqrt(hq[i].z);
        }

        // break-out check if we have reached the limit of the precision
        if (r2 > r2_old) {
          resIncrease++;
          resIncreaseTotal++;
          warningQuda(
            "CA-GCR: new reliable residual norm %e is greater than previous reliable residual norm %e (total #inc %i)",
            sqrt(r2[0]), sqrt(r2_old[0]), resIncreaseTotal);
          if (resIncrease > maxResIncrease or resIncreaseTotal > maxResIncreaseTotal) {
            warningQuda("CA-GCR: solver exiting due to too many true residual norm increases");
            break;
          }
        } else {
          resIncrease = 0;
        }

        r2_old = r2;
      }

      // No matter what, if we haven't converged, we do a restart.
      if (!convergenceL2(r2, stop)) {
        restart++; // restarting if residual is still too great

        PrintStats("CA-GCR (restart)", restart, r2, b2, heavy_quark_res);
        blas::copy(p[0], r); // no-op if uni-precision

        r2_old = r2;

        // prevent ending the Krylov space prematurely if other convergence criteria not met
        if (r2 < stop) l2_converge = true;
      }
    }

    if (total_iter > param.maxiter && getVerbosity() >= QUDA_SUMMARIZE)
      warningQuda("Exceeded maximum iterations %d", param.maxiter);

    logQuda(QUDA_VERBOSE, "CA-GCR: number of restarts = %d\n", restart);

    if (param.compute_true_res) {
      // Calculate the true residual
      mat(r, x);
      auto true_r2 = blas::xmyNorm(b, r);
      auto hq = blas::HeavyQuarkResidualNorm(x, r);
      for (auto i = 0u; i < b.size(); i++) {
        param.true_res[i] = sqrt(true_r2[i] / b2[i]);
        param.true_res_hq[i] = sqrt(hq[i].z);
      }
    }

    if (!param.is_preconditioner) {
      getProfile().TPSTOP(QUDA_PROFILE_COMPUTE);
      param.iter += total_iter;
    }

    PrintSummary("CA-GCR", total_iter, r2, b2, stop);

    if (param.is_preconditioner) commGlobalReductionPop();
  }

} // namespace quda
