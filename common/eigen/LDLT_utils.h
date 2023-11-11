//
// Created by leavesnight on 9/11/23.
//

#pragma once

#include "eigen_utils.h"
#include "common/config.h"
#include "common/mlog/log.h"

namespace Eigen {
template <class Tmat>
Tmat GetLTofSemiDefiniteInfo(const Tmat &Omegaij, bool bfix_numerical = false,
#ifdef CHECK_NUMERICAL_SAFETY
                             const bool bassert = true
#else
                             const bool bassert = false
#endif
) {
  assert(Tmat::RowsAtCompileTime == Tmat::ColsAtCompileTime);
  // Tmat Omegaij = (Omegaij_in + Omegaij_in.transpose()).eval() / 2.;
  Eigen::LLT<Tmat> llt(Omegaij);
  Tmat sqrt_Omegaij_;
  if (Eigen::Success == llt.info()) {
    sqrt_Omegaij_ = llt.matrixU();
  } else {
    Eigen::LDLT<Tmat> ldlt(Omegaij);  // A = PT*L*D*LT*P
    Eigen::Matrix<typename Tmat::Scalar, Tmat::RowsAtCompileTime, 1> D = ldlt.vectorD();
    if (bfix_numerical)
      for (int i = 0, iend = D.size(); i < iend; ++i)
        if (D[i] < 0 && D[i] > -1e-5) D[i] = 0;
    Tmat sqrtD = Eigen::DiagonalMatrix<typename Tmat::Scalar, Tmat::RowsAtCompileTime>(D.cwiseSqrt());
    if (bassert) {
      if (Eigen::Success != ldlt.info()) {
        MLOG_ENSURE(0, "ldlt.info={}", (int)ldlt.info());
      }
      if (sqrtD.hasNaN()) {
        MLOG_ENSURE(0, "ldlt.vectorD={},H_in={}", ldlt.vectorD().transpose(), Omegaij);
      }
    } else {
      if (sqrtD.hasNaN()) {
        std::stringstream sstr;
        sstr << "D=" << ldlt.vectorD().transpose();
        PRINT_ERR_MUTEX(yellowSTR << sstr.str().c_str() << whiteSTR << std::endl);
      }
    }
    // so A = L'L'T, L'T = Pinv * sqrt(D) * LT * (P * I)
    Tmat P = ldlt.transpositionsP() * Tmat::Identity(Tmat::RowsAtCompileTime, Tmat::RowsAtCompileTime);
    sqrt_Omegaij_ = ldlt.transpositionsP().transpose() * sqrtD * ldlt.matrixU() * P;
  }

  return sqrt_Omegaij_;
}  // namespace Eigen

template <class MatBlock, class MatrixDiag>
static void EnsureInvertible(MatBlock H_A, MatrixDiag &diag, int8_t mode = 0) {
  const size_t SIZE_LEFT_NAV_LOCAL = MatBlock::RowsAtCompileTime;
  if (2 > mode) {
    for (int row = 0, row_end = MatrixDiag::ColsAtCompileTime; row < row_end; ++row) {
      if (0 && !mode) {
        diag(row) = std::min(std::max(diag(row), 1e-3), 1e32) * 1e-4;  // for sqrt, 1e-6/1e32/1e-8
      } else {
        diag(row) = std::min(std::max(diag(row), 1e-2), 1e32) * 1e-3;  // for sqrt, 1e-6/1e32/1e-8
      }
    }
  }
  // Eigen::Matrix<double, SIZE_LEFT_NAV_LOCAL, SIZE_LEFT_NAV_LOCAL> H_Atmp = H_A;
  H_A.diagonal() += diag;
  if (1 == mode) {
    double diagmax_ref = diag.maxCoeff() * 1e-6;
    for (size_t i = 0; i < SIZE_LEFT_NAV_LOCAL; ++i)
      if (H_A(i, i) < diagmax_ref) {
#ifdef CHECK_NUMERICAL_SAFETY
        PRINT_INFO_FILE_MUTEX("H_A(" << i << ")=" << H_A(i, i) << std::endl, VIEO_SLAM::mlog::vieo_slam_debug_path,
                              "debug.txt");
#endif
        H_A(i, i) = diagmax_ref;
      }
  }
  if (1 != mode) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, SIZE_LEFT_NAV_LOCAL, SIZE_LEFT_NAV_LOCAL>> es(H_A);
    Eigen::Matrix<double, SIZE_LEFT_NAV_LOCAL, 1> eigs = es.eigenvalues();
#ifdef CHECK_NUMERICAL_SAFETY
    bool beig_fail = false;
#endif
    double eig_max = eigs[SIZE_LEFT_NAV_LOCAL - 1];
    for (size_t i = 0; i < SIZE_LEFT_NAV_LOCAL; ++i)
      if (eigs[i] < eig_max * 1e-14) {
#ifdef CHECK_NUMERICAL_SAFETY
        beig_fail = true;
#endif
        eigs[i] = eig_max * 1e-14;
      }
#ifdef CHECK_NUMERICAL_SAFETY
    if (beig_fail) {
      PRINT_INFO_FILE_MUTEX("H_A_eigs=" << es.eigenvalues().transpose() << std::endl,
                            VIEO_SLAM::mlog::vieo_slam_debug_path, "debug.txt");
    }
#endif
    H_A = es.eigenvectors() * eigs.asDiagonal() * es.eigenvectors().transpose();
  }
}

}  // namespace Eigen
