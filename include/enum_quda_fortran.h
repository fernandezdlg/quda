#/*
# enum_quda_fortran.h
#
#   This  is  Fortran  version  of  enum_quda.h.   This  is  currently
#   generated by hand,  and so must be matched  against an appropriate
#   version  of QUDA.   It would  be nice  to auto-generate  this from
#   enum_quda.h and this should probably use Fortran enumerate types 
#   instead (this requires Fortran 2003, but this is covered by 
#   gfortran).
#*/

#define QUDA_INVALID_ENUM (-Z'7fffffff' - 1)

#define QudaLinkType integer(4)

#define QUDA_SUCCESS 0
#define QUDA_ERROR 1
#define QUDA_ERROR_UNINITIALIZED 2

#define QUDA_MEMORY_DEVICE 0
#define QUDA_MEMORY_PINNED 1
#define QUDA_MEMORY_MAPPED 2
#define QUDA_MEMORY_INVALID QUDA_INVALID_ENUM

#define QUDA_SU3_LINKS      0
#define QUDA_GENERAL_LINKS  1
#define QUDA_THREE_LINKS    2
#define QUDA_MOMENTUM_LINKS 3
#define QUDA_COARSE_LINKS   4
#define QUDA_SMEARED_LINKS  5
#define QUDA_TWOLINK_LINKS_LINKS 6

#define QUDA_WILSON_LINKS         QUDA_SU3_LINKS
#define QUDA_ASQTAD_FAT_LINKS     QUDA_GENERAL_LINKS
#define QUDA_ASQTAD_LONG_LINKS    QUDA_THREE_LINKS
#define QUDA_ASQTAD_MOM_LINKS     QUDA_MOMENTUM_LINKS
#define QUDA_ASQTAD_GENERAL_LINKS QUDA_GENERAL_LINKS
#define QUDA_INVALID_LINKS        QUDA_INVALID_ENUM

#define QudaGaugeFieldOrder integer(4)
#define QUDA_FLOAT_GAUGE_ORDER 1
#define QUDA_FLOAT2_GAUGE_ORDER 2 //no reconstruct and double precision
#define QUDA_FLOAT4_GAUGE_ORDER 4 // 8 reconstruct single, and 12 reconstruct single, half, quarter
#define QUDA_FLOAT8_GAUGE_ORDER 8 // 8 reconstruct half and quarter
#define QUDA_NATIVE_GAUGE_ORDER 9 // used to denote one of the above types in a trait, not used directly
#define QUDA_QDP_GAUGE_ORDER 10   // expect *gauge[4] even-odd spacetime row-column color
#define QUDA_QDPJIT_GAUGE_ORDER 11     // expect *gauge[4] even-odd spacetime row-column color
#define QUDA_CPS_WILSON_GAUGE_ORDER 12 // expect *gauge even-odd spacetime column-row color
#define QUDA_MILC_GAUGE_ORDER 13       // expect *gauge even-odd mu spacetime row-column order
#define QUDA_MILC_SITE_GAUGE_ORDER                                                                                     \
  14                             // packed into MILC site AoS [even-odd][spacetime] array, and [dir][row][col] inside
#define QUDA_BQCD_GAUGE_ORDER 15 // expect *gauge mu even-odd spacetime+halos row-column order
#define QUDA_TIFR_GAUGE_ORDER 16
#define QUDA_TIFR_PADDED_GAUGE_ORDER 17
#define QUDA_INVALID_GAUGE_ORDER QUDA_INVALID_ENUM

#define QudaTboundary integer(4)
#define QUDA_ANTI_PERIODIC_T -1
#define QUDA_PERIODIC_T 1
#define QUDA_INVALID_T_BOUNDARY QUDA_INVALID_ENUM

#define QudaPrecision integer(4)
#define QUDA_QUARTER_PRECISION 1
#define QUDA_HALF_PRECISION 2
#define QUDA_SINGLE_PRECISION 4
#define QUDA_DOUBLE_PRECISION 8
#define QUDA_INVALID_PRECISION QUDA_INVALID_ENUM

#define QudaReconstructType integer(4)
#define QUDA_RECONSTRUCT_NO 18
#define QUDA_RECONSTRUCT_12 12
#define QUDA_RECONSTRUCT_8 8  
#define QUDA_RECONSTRUCT_9 9 
#define QUDA_RECONSTRUCT_13 13
#define QUDA_RECONSTRUCT_10 10
#define QUDA_RECONSTRUCT_INVALID QUDA_INVALID_ENUM

#define QudaGaugeFixed integer(4)
#define QUDA_GAUGE_FIXED_NO  0
#define QUDA_GAUGE_FIXED_YES 1 // gauge field stored in temporal gauge
#define QUDA_GAUGE_FIXED_INVALID QUDA_INVALID_ENUM

! Types used in QudaInvertParam

#define QudaDslashType integer(4)
#define QUDA_WILSON_DSLASH 0 
#define QUDA_CLOVER_WILSON_DSLASH 1
#define QUDA_CLOVER_HASENBUSCH_TWIST_DSLASH 2
#define QUDA_DOMAIN_WALL_DSLASH 3
#define QUDA_DOMAIN_WALL_4D_DSLASH 4
#define QUDA_MOBIUS_DWF_DSLASH 5
#define QUDA_MOBIUS_DWF_EOFA_DSLASH 6
#define QUDA_STAGGERED_DSLASH 7
#define QUDA_ASQTAD_DSLASH 8
#define QUDA_TWISTED_MASS_DSLASH 9
#define QUDA_TWISTED_CLOVER_DSLASH 10
#define QUDA_LAPLACE_DSLASH 11
#define QUDA_COVDEV_DSLASH 12
#define QUDA_INVALID_DSLASH QUDA_INVALID_ENUM

#define QudaInverterType integer(4)
#define QUDA_CG_INVERTER 0
#define QUDA_BICGSTAB_INVERTER 1
#define QUDA_GCR_INVERTER 2
#define QUDA_MR_INVERTER 3
#define QUDA_SD_INVERTER 4
#define QUDA_PCG_INVERTER 5
#define QUDA_EIGCG_INVERTER 6
#define QUDA_INC_EIGCG_INVERTER 7
#define QUDA_GMRESDR_INVERTER 8
#define QUDA_GMRESDR_PROJ_INVERTER 9
#define QUDA_GMRESDR_SH_INVERTER 10
#define QUDA_FGMRESDR_INVERTER 11
#define QUDA_MG_INVERTER 12
#define QUDA_BICGSTABL_INVERTER 13
#define QUDA_CGNE_INVERTER 14
#define QUDA_CGNR_INVERTER 15
#define QUDA_CG3_INVERTER 16
#define QUDA_CG3NE_INVERTER 17
#define QUDA_CG3NR_INVERTER 18
#define QUDA_CA_CG_INVERTER 19
#define QUDA_CA_CGNE_INVERTER 20
#define QUDA_CA_CGNR_INVERTER 21
#define QUDA_CA_GCR_INVERTER 22
#define QUDA_INVALID_INVERTER QUDA_INVALID_ENUM

#define QudaEigType integer(4)
#define QUDA_EIG_TR_LANCZOS 0 // Thick Restarted Lanczos Solver
#define QUDA_EIG_BLK_IR_LANCZOS 1 // Block Thick Restarted Lanczos Solver
#define QUDA_EIG_IR_ARNOLDI 2 // Implicitly restarted Arnoldi solver
#define QUDA_EIG_BLK_IR_ARNOLDI 3 // Block Implicitly restarted Arnoldi solver (not yet implemented)
#define QUDA_EIG_INVALID QUDA_INVALID_ENUM

#define QudaEigSpectrumType integer(4)
#define QUDA_SPECTRUM_LM_EIG 0
#define QUDA_SPECTRUM_SM_EIG 1
#define QUDA_SPECTRUM_LR_EIG 2
#define QUDA_SPECTRUM_SR_EIG 3
#define QUDA_SPECTRUM_LI_EIG 4
#define QUDA_SPECTRUM_SI_EIG 5
#define QUDA_SPECTRUM_INVALID QUDA_INVALID_ENUM

#define QudaSolutionType integer(4)
#define QUDA_MAT_SOLUTION 0 
#define QUDA_MATDAG_MAT_SOLUTION 1
#define QUDA_MATPC_SOLUTION 2
#define QUDA_MATPC_DAG_SOLUTION 3
#define QUDA_MATPCDAG_MATPC_SOLUTION 4
#define QUDA_MATPCDAG_MATPC_SHIFT_SOLUTION 5
#define QUDA_INVALID_SOLUTION QUDA_INVALID_ENUM

#define QudaSolveType integer(4)
#define QUDA_DIRECT_SOLVE 0
#define QUDA_NORMOP_SOLVE 1
#define QUDA_DIRECT_PC_SOLVE 2
#define QUDA_NORMOP_PC_SOLVE 3
#define QUDA_NORMERR_SOLVE 4
#define QUDA_NORMERR_PC_SOLVE 5
#define QUDA_NORMEQ_SOLVE QUDA_NORMOP_SOLVE // deprecated
#define QUDA_NORMEQ_PC_SOLVE QUDA_NORMOP_PC_SOLVE // deprecated
#define QUDA_INVALID_SOLVE QUDA_INVALID_ENUM

#define QudaMultigridCycleType integer(4)
#define QUDA_MG_CYCLE_VCYCLE 0
#define QUDA_MG_CYCLE_FCYCLE 1
#define QUDA_MG_CYCLE_WCYCLE 2
#define QUDA_MG_CYCLE_RECURSIVE 3
#define QUDA_MG_CYCLE_INVALID QUDA_INVALID_ENUM

#define QudaSchwarzType integer(4)
#define QUDA_ADDITIVE_SCHWARZ 0 
#define QUDA_MULTIPLICATIVE_SCHWARZ 1
#define QUDA_INVALID_SCHWARZ QUDA_INVALID_ENUM

#define QudaAcceleratorType integer(4)
#define QUDA_MADWF_ACCELERATOR 0
#define QUDA_INVALID_ACCELERATOR QUDA_INVALID_ENUM

#define QudaResidualType integer(4)
#define QUDA_L2_RELATIVE_RESIDUAL 1
#define QUDA_L2_ABSOLUTE_RESIDUAL 2
#define QUDA_HEAVY_QUARK_RESIDUAL 4
#define QUDA_INVALID_RESIDUAL QUDA_INVALID_ENUM

#define QudaCABasis integer(4)
#define QUDA_POWER_BASIS 0
#define QUDA_CHEBYSHEV_BASIS 1
#define QUDA_INVALID_BASIS QUDA_INVALID_ENUM

#/*
   # Whether the preconditioned matrix is (1-k^2 Deo Doe) or (1-k^2 Doe Deo)
   #
   # For the clover-improved Wilson Dirac operator QUDA_MATPC_EVEN_EVEN
   # defaults to the "symmetric" form (1 - k^2 A_ee^-1 D_eo A_oo^-1 D_oe)
   # and likewise for QUDA_MATPC_ODD_ODD.
   #
   # For the "asymmetric" form (A_ee - k^2 D_eo A_oo^-1 D_oe) select
   # QUDA_MATPC_EVEN_EVEN_ASYMMETRIC.
   # */

#define QudaMatPCType integer(4)
#define QUDA_MATPC_EVEN_EVEN 0
#define QUDA_MATPC_ODD_ODD 1
#define QUDA_MATPC_EVEN_EVEN_ASYMMETRIC 2
#define QUDA_MATPC_ODD_ODD_ASYMMETRIC 3
#define QUDA_MATPC_INVALID QUDA_INVALID_ENUM

#define QudaDagType integer(4)
#define QUDA_DAG_NO 0 
#define QUDA_DAG_YES 1
#define QUDA_DAG_INVALID QUDA_INVALID_ENUM
  
#define QudaMassNormalization integer(4)
#define QUDA_KAPPA_NORMALIZATION 0 
#define QUDA_MASS_NORMALIZATION 1
#define QUDA_ASYMMETRIC_MASS_NORMALIZATION 2
#define QUDA_INVALID_NORMALIZATION QUDA_INVALID_ENUM

#define QudaSolverNormalization integer(4)
#define QUDA_DEFAULT_NORMALIZATION 0 // leave source and solution untouched
#define QUDA_SOURCE_NORMALIZATION  1 // normalize such that || src || = 1

#define QudaPreserveSource integer(4)
#define QUDA_PRESERVE_SOURCE_NO  0 // use the source for the residual
#define QUDA_PRESERVE_SOURCE_YES 1
#define QUDA_PRESERVE_SOURCE_INVALID QUDA_INVALID_ENUM

#define QudaDiracFieldOrder integer(4)
#define QUDA_INTERNAL_DIRAC_ORDER 0    // internal dirac order used by QUDA varies depending on precision and dslash type
#define QUDA_DIRAC_ORDER 1
#define QUDA_QDP_DIRAC_ORDER 2         // even-odd spin inside color
#define QUDA_QDPJIT_DIRAC_ORDER 3      // even-odd, complex-color-spin-spacetime
#define QUDA_CPS_WILSON_DIRAC_ORDER 4  // odd-even color inside spin
#define QUDA_LEX_DIRAC_ORDER 5         // lexicographical order color inside spin
#define QUDA_TIFR_PADDED_DIRAC_ORDER 6
#define QUDA_INVALID_DIRAC_ORDER QUDA_INVALID_ENUM

#define QudaCloverFieldOrder integer(4)
#define QUDA_FLOAT_CLOVER_ORDER 1   // even-odd float ordering 
#define QUDA_FLOAT2_CLOVER_ORDER 2   // even-odd float2 ordering
#define QUDA_FLOAT4_CLOVER_ORDER 4   // even-odd float4 ordering
#define QUDA_FLOAT8_CLOVER_ORDER 8   // even-odd float8 ordering
#define QUDA_PACKED_CLOVER_ORDER 9   // even-odd packed
#define QUDA_QDPJIT_CLOVER_ORDER 10  // lexicographical order packed
#define QUDA_BQCD_CLOVER_ORDER 11    // BQCD order which is a packed super-diagonal form
#define QUDA_INVALID_CLOVER_ORDER QUDA_INVALID_ENUM

#define QudaVerbosity integer(4)
#define QUDA_SILENT 0
#define QUDA_SUMMARIZE 1
#define QUDA_VERBOSE 2
#define QUDA_DEBUG_VERBOSE 3
#define QUDA_INVALID_VERBOSITY QUDA_INVALID_ENUM

#define QudaTune integer(4)
#define QUDA_TUNE_NO 0
#define QUDA_TUNE_YES 1
#define QUDA_TUNE_INVALID QUDA_INVALID_ENUM

#define QudaPreserveDirac integer(4)
#define QUDA_PRESERVE_DIRAC_NO 0
#define QUDA_PRESERVE_DIRAC_YES 1
#define QUDA_PRESERVE_DIRAC_INVALID QUDA_INVALID_ENUM

! Type used for "parity" argument to dslashQuda()

#define QudaParity integer(4)
#define QUDA_EVEN_PARITY 0
#define QUDA_ODD_PARITY 1
#define QUDA_INVALID_PARITY QUDA_INVALID_ENUM

! Types used only internally

#define QudaDiracType integer(4)
#define QUDA_WILSON_DIRAC 0
#define QUDA_WILSONPC_DIRAC 1
#define QUDA_CLOVER_DIRAC 2
#define QUDA_CLOVERPC_DIRAC 3
#define QUDA_CLOVER_HASENBUSCH_TWIST_DIRAC 4
#define QUDA_CLOVER_HASENBUSCH_TWISTPC_DIRAC 5
#define QUDA_DOMAIN_WALL_DIRAC 6
#define QUDA_DOMAIN_WALLPC_DIRAC 7
#define QUDA_DOMAIN_WALL_4D_DIRAC 8
#define QUDA_DOMAIN_WALL_4DPC_DIRAC 9
#define QUDA_MOBIUS_DOMAIN_WALL_DIRAC 10
#define QUDA_MOBIUS_DOMAIN_WALLPC_DIRAC 11
#define QUDA_MOBIUS_DOMAIN_WALL_EOFA_DIRAC 12
#define QUDA_MOBIUS_DOMAIN_WALLPC_EOFA_DIRAC 13
#define QUDA_STAGGERED_DIRAC 14
#define QUDA_STAGGEREDPC_DIRAC 15
#define QUDA_STAGGEREDKD_DIRAC 16
#define QUDA_ASQTAD_DIRAC 17
#define QUDA_ASQTADPC_DIRAC 18
#define QUDA_ASQTADKD_DIRAC 19
#define QUDA_TWISTED_MASS_DIRAC 20
#define QUDA_TWISTED_MASSPC_DIRAC 21
#define QUDA_TWISTED_CLOVER_DIRAC 22
#define QUDA_TWISTED_CLOVERPC_DIRAC 23
#define QUDA_COARSE_DIRAC 24
#define QUDA_COARSEPC_DIRAC 25
#define QUDA_GAUGE_LAPLACE_DIRAC 26
#define QUDA_GAUGE_LAPLACEPC_DIRAC 27
#define QUDA_GAUGE_COVDEV_DIRAC 28
#define QUDA_INVALID_DIRAC QUDA_INVALID_ENUM

! Where the field is stored
#define QudaFieldLocation integer(4)
#define QUDA_CPU_FIELD_LOCATION 1
#define QUDA_CUDA_FIELD_LOCATION 2
#define QUDA_INVALID_FIELD_LOCATION QUDA_INVALID_ENUM
  
! Which sites are included
#define QudaSiteSubset integer(4)
#define QUDA_PARITY_SITE_SUBSET 1
#define QUDA_FULL_SITE_SUBSET 2
#define QUDA_INVALID_SITE_SUBSET QUDA_INVALID_ENUM
  
! Site ordering (always t-z-y-x with rightmost varying fastest)
#define QudaSiteOrder integer(4)
#define QUDA_LEXICOGRAPHIC_SITE_ORDER 0 // lexicographic ordering
#define QUDA_EVEN_ODD_SITE_ORDER 1 // QUDA and QDP use this
#define QUDA_ODD_EVEN_SITE_ORDER 2 // CPS uses this
#define QUDA_INVALID_SITE_ORDER QUDA_INVALID_ENUM
  
! Degree of freedom ordering
#define QudaFieldOrder integer(4)
#define QUDA_FLOAT_FIELD_ORDER 1 // spin-color-complex-space
#define QUDA_FLOAT2_FIELD_ORDER 2 // (spin-color-complex)/2-space-(spin-color-complex)%2
#define QUDA_FLOAT4_FIELD_ORDER 4 // (spin-color-complex)/4-space-(spin-color-complex)%4
#define QUDA_FLOAT8_FIELD_ORDER 8 // (spin-color-complex)/8-space-(spin-color-complex)%8
#define QUDA_SPACE_SPIN_COLOR_FIELD_ORDER 9         // CPS/QDP++ ordering
#define QUDA_SPACE_COLOR_SPIN_FIELD_ORDER 10        // QLA ordering (spin inside color)
#define QUDA_QDPJIT_FIELD_ORDER 11                  // QDP field ordering (complex-color-spin-spacetime)
#define QUDA_QOP_DOMAIN_WALL_FIELD_ORDER 12         // QOP domain-wall ordering
#define QUDA_PADDED_SPACE_SPIN_COLOR_FIELD_ORDER 13 // TIFR RHMC ordering
#define QUDA_INVALID_FIELD_ORDER QUDA_INVALID_ENUM
  
#define QudaFieldCreate integer(4)
#define QUDA_NULL_FIELD_CREATE 0      // new field
#define QUDA_ZERO_FIELD_CREATE 1      // new field and zero it
#define QUDA_COPY_FIELD_CREATE 2      // copy to field
#define QUDA_REFERENCE_FIELD_CREATE 3 // reference to field
#define QUDA_GHOST_FIELD_CREATE 4     // dummy field used only for ghost storage
#define QUDA_INVALID_FIELD_CREATE QUDA_INVALID_ENUM

#define QudaGammaBasis integer(4)
#define QUDA_DEGRAND_ROSSI_GAMMA_BASIS 0
#define QUDA_UKQCD_GAMMA_BASIS 1
#define QUDA_CHIRAL_GAMMA_BASIS 2
#define QUDA_INVALID_GAMMA_BASIS QUDA_INVALID_ENUM

#define QudaSourceType integer(4)
#define QUDA_POINT_SOURCE 0
#define QUDA_RANDOM_SOURCE 1
#define QUDA_CONSTANT_SOURCE 2
#define QUDA_SINUSOIDAL_SOURCE 3
#define QUDA_CORNER_SOURCE 4
#define QUDA_INVALID_SOURCE QUDA_INVALID_ENUM

#define QudaNoiseType integer(4)
#define QUDA_NOISE_GAUSS 0
#define QUDA_NOISE_UNIFORM 1
#define QUDA_NOISE_INVALID QUDA_INVALID_ENUM

#define QudaDilutionType integer(4)
#define QUDA_DILUTION_SPIN 0
#define QUDA_DILUTION_COLOR 1
#define QUDA_DILUTION_SPIN_COLOR 2
#define QUDA_DILUTION_SPIN_COLOR_EVEN_ODD 3
#define QUDA_DILUTION_INVALID QUDA_INVALID_ENUM

#define QudaProjectionType integer(4)
#define QUDA_MINRES_PROJECTION 0
#define QUDA_GALERKIN_PROJECTION 1
#define QUDA_INVALID_PROJECTION QUDA_INVALID_ENUM

#define QudaPCType integer(4)
#define QUDA_4D_PC 4
#define QUDA_5D_PC 5
#define QUDA_PC_INVALID QUDA_INVALID_ENUM

#define QudaTwistFlavorType integer(4)
#define QUDA_TWIST_SINGLET 1
#define QUDA_TWIST_NONDEG_DOUBLET +2
#define QUDA_TWIST_NO  0
#define QUDA_TWIST_INVALID QUDA_INVALID_ENUM

#define QudaTwistDslashType integer(4)
#define QUDA_DEG_TWIST_INV_DSLASH 0
#define QUDA_DEG_DSLASH_TWIST_INV 1
#define QUDA_DEG_DSLASH_TWIST_XPAY 2
#define QUDA_NONDEG_DSLASH 3
#define QUDA_DSLASH_INVALID QUDA_INVALID_ENUM

#define QudaTwistCloverDslashType integer(4)
#define QUDA_DEG_CLOVER_TWIST_INV_DSLASH 0
#define QUDA_DEG_DSLASH_CLOVER_TWIST_INV 1
#define QUDA_DEG_DSLASH_CLOVER_TWIST_XPAY 2
#define QUDA_TC_DSLASH_INVALID QUDA_INVALID_ENUM

#define QudaTwistGamma5Type integer(4)
#define QUDA_TWIST_GAMMA5_DIRECT 0
#define QUDA_TWIST_GAMMA5_INVERSE 1
#define QUDA_TWIST_GAMMA5_INVALID QUDA_INVALID_ENUM

#define QudaUseInitGuess integer(4)
#define QUDA_USE_INIT_GUESS_NO  0 
#define QUDA_USE_INIT_GUESS_YES 1
#define QUDA_USE_INIT_GUESS_INVALID QUDA_INVALID_ENUM

#define QudaComputeNullVector integer(4)
#define QUDA_COMPUTE_NULL_VECTOR_NO  0 
#define QUDA_COMPUTE_NULL_VECTOR_YES 1
#define QUDA_COMPUTE_NULL_VECTOR_INVALID QUDA_INVALID_ENUM

#define QudaSetupType integer(4)
#define QUDA_NULL_VECTOR_SETUP 0
#define QUDA_TEST_VECTOR_SETUP 1
#define QUDA_INVALID_SETUP_TYPE QUDA_INVALID_ENUM

#define QudaTransferType integer(4)
#define QUDA_TRANSFER_AGGREGATE 0
#define QUDA_TRANSFER_COARSE_KD 1
#define QUDA_TRANSFER_OPTIMIZED_KD 2
#define QUDA_TRANSFER_OPTIMIZED_KD_DROP_LONG 3
#define QUDA_TRANSFER_INVALID QUDA_INVALID_ENUM

#define QudaBoolean integer(4)
#define QUDA_BOOLEAN_FALSE 0
#define QUDA_BOOLEAN_TRUE 1
#define QUDA_BOOLEAN_INVALID QUDA_INVALID_ENUM
#define QUDA_BOOLEAN_NO QUDA_BOOLEAN_FALSE // backwards compatibility
#define QUDA_BOOLEAN_YES QUDA_BOOLEAN_TRUE // backwards compatibility

#define QudaBLASOperation integer(4)
#define QUDA_BLAS_OP_N = 0 // No transpose
#define QUDA_BLAS_OP_T = 1 // Transpose only
#define QUDA_BLAS_OP_C = 2 // Conjugate transpose
#define QUDA_BLAS_OP_INVALID QUDA_INVALID_ENUM

#define QudaBLASDataType integer(4)
#define QUDA_BLAS_DATATYPE_S 0 // Single
#define QUDA_BLAS_DATATYPE_D 1 // Double
#define QUDA_BLAS_DATATYPE_C 2 // Complex(single)
#define QUDA_BLAS_DATATYPE_Z 3 // Complex(double)
#define QUDA_BLAS_DATATYPE_INVALID QUDA_INVALID_ENUM

#define QudaBLASDataOrder integer(4)
#define QUDA_BLAS_DATAORDER_ROW 0
#define QUDA_BLAS_DATAORDER_COL 1
#define QUDA_BLAS_DATAORDER_INVALID QUDA_INVALID_ENUM

#define QudaDirection integer(4)
#define QUDA_BACKWARDS -1
#define QUDA_FORWARDS  +1
#define QUDA_BOTH_DIRS 2

#define QudaLinkDirection integer(4)
#define QUDA_LINK_BACKWARDS 0
#define QUDA_LINK_FORWARDS 1
#define QUDA_LINK_BIDIRECTIONAL 2

#define QudaFieldGeometry integer(4)
#define QUDA_SCALAR_GEOMETRY 1
#define QUDA_VECTOR_GEOMETRY 4
#define QUDA_TENSOR_GEOMETRY 6
#define QUDA_COARSE_GEOMETRY 8
#define QUDA_KDINVERSE_GEOMETRY 16 // Decomposition of the Kahler-Dirac block
#define QUDA_INVALID_GEOMETRY QUDA_INVALID_ENUM

#define QudaGhostExchange integer(4)
#define QUDA_GHOST_EXCHANGE_NO       0
#define QUDA_GHOST_EXCHANGE_PAD      1
#define QUDA_GHOST_EXCHANGE_EXTENDED 2
#define QUDA_GHOST_EXCHANGE_INVALID QUDA_INVALID_ENUM

#define QudaStaggeredPhase integer(4)
#define QUDA_STAGGERED_PHASE_NO   0
#define QUDA_STAGGERED_PHASE_MILC 1
#define QUDA_STAGGERED_PHASE_CPS  2
#define QUDA_STAGGERED_PHASE_TIFR 3
#define QUDA_STAGGERED_PHASE_INVALID QUDA_INVALID_ENUM

#define QudaSpinTasteGamma integer(4)
#define QUDA_SPIN_TASTE_G1 0
#define QUDA_SPIN_TASTE_GX 1
#define QUDA_SPIN_TASTE_GY 2
#define QUDA_SPIN_TASTE_GZ 4
#define QUDA_SPIN_TASTE_GT 8
#define QUDA_SPIN_TASTE_G5 15
#define QUDA_SPIN_TASTE_GYGZ 6
#define QUDA_SPIN_TASTE_GZGX 5
#define QUDA_SPIN_TASTE_GXGY 3
#define QUDA_SPIN_TASTE_GXGT 9
#define QUDA_SPIN_TASTE_GYGT 10
#define QUDA_SPIN_TASTE_GZGT 12
#define QUDA_SPIN_TASTE_G5GX 14
#define QUDA_SPIN_TASTE_G5GY 13
#define QUDA_SPIN_TASTE_G5GZ 11
#define QUDA_SPIN_TASTE_G5GT 7
#define QUDA_SPIN_TASTE_INVALID QUDA_INVALID_ENUM

#define QudaContractType integer(4)
#define QUDA_CONTRACT_TYPE_OPEN ,
#define QUDA_CONTRACT_TYPE_DR ,
#define QUDA_CONTRACT_TYPE_INVALID = QUDA_INVALID_ENUM

#define QudaContractGamma integer(4)
#define QUDA_CONTRACT_GAMMA_I 0
#define QUDA_CONTRACT_GAMMA_G1 1
#define QUDA_CONTRACT_GAMMA_G2 2
#define QUDA_CONTRACT_GAMMA_G3 3
#define QUDA_CONTRACT_GAMMA_G4 4
#define QUDA_CONTRACT_GAMMA_G5 5
#define QUDA_CONTRACT_GAMMA_G1G5 6
#define QUDA_CONTRACT_GAMMA_G2G5 7
#define QUDA_CONTRACT_GAMMA_G3G5 8
#define QUDA_CONTRACT_GAMMA_G4G5 9
#define QUDA_CONTRACT_GAMMA_S12 10
#define QUDA_CONTRACT_GAMMA_S13 11
#define QUDA_CONTRACT_GAMMA_S14 12
#define QUDA_CONTRACT_GAMMA_S21 13
#define QUDA_CONTRACT_GAMMA_S23 14
#define QUDA_CONTRACT_GAMMA_S34 15
#define QUDA_CONTRACT_GAMMA_INVALID QUDA_INVALID_ENUM

#define QudaGaugeSmearType integer(4)
#define QUDA_GAUGE_SMEAR_APE 0
#define QUDA_GAUGE_SMEAR_STOUT 1
#define QUDA_GAUGE_SMEAR_OVRIMP_STOUT 2
#define QUDA_GAUGE_SMEAR_WILSON_FLOW 3
#define QUDA_GAUGE_SMEAR_SYMANZIK_FLOW 4
#define QUDA_GAUGE_SMEAR_INVALID QUDA_INVALID_ENUM

#define QudaFermionSmearType integer(4)
#define QUDA_FERMION_SMEAR_TYPE_GAUSSIAN 0
#define QUDA_FERMION_SMEAR_TYPE_WUPPERTAL 1
#define QUDA_FERMION_SMEAR_TYPE_INVALID QUDA_INVALID_ENUM


#define QudaExtLibType integer(4)
#define QUDA_CUSOLVE_EXTLIB 0
#define QUDA_EIGEN_EXTLIB 1
#define QUDA_EXTLIB_INVALID QUDA_INVALID_ENUM
