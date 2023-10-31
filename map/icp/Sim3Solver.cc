/**
 * This file is part of VIEO_SLAM
 */

#include "map/icp/Sim3Solver.h"
#include "CameraModels/GeometricCamera.h"
#include "Converter.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

namespace ORB_SLAM3 {

// pKF1 is current kf, pKF2 is loop candidate KFs, vpMatched12[i] matched to
// pKF1->mvpMapPoints[i], bFixScale=true for
//  RGBD/Stereo
Sim3Solver::Sim3Solver(const vector<KeyFrame *> &vpkf12,
                       const vector<MapPoint *> &vpMatched12,
                       const bool bFixScale)
    : mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale) {
  // get cam models
  auto nkfs = vpkf12.size();
  assert(nkfs >= 2);
  usedistort_.reserve(nkfs);
  pcams_.reserve(nkfs);
  vector<size_t> offset_cam0(nkfs + 1);
  offset_cam0[0] = 0;
  for (int i = 0; i < nkfs; ++i) {
    auto &pkf_tmp = vpkf12[i];
    // assert(!pkf_tmp->mpCameras.empty());
    assert(pkf_tmp->mpCamera);
    usedistort_.emplace_back(true); // pkf_tmp->usedistort_);
    size_t num_cams_kf_tmp = 1;
    {
      // pcams_.insert(pcams_.end(), pkf_tmp->mpCameras.begin(),
      // pkf_tmp->mpCameras.end());
      pcams_.emplace_back(pkf_tmp->mpCamera);
      mapidx2Tcr_.emplace_back(Sophus::SE3exf());
      if (pkf_tmp->mpCamera2) {
        pcams_.emplace_back(pkf_tmp->mpCamera2);
        mapidx2Tcr_.emplace_back(pkf_tmp->GetRelativePoseTrl());
        ++num_cams_kf_tmp;
      }
    }
    offset_cam0[i + 1] = offset_cam0[i] + num_cams_kf_tmp;
  }
  assert(offset_cam0[nkfs] == pcams_.size());

  auto pkf1 = vpkf12[0];
  vector<MapPoint *> vpKeyFrameMP1 =
      pkf1->GetMapPointMatches(); // maybe can use vec<MP*>&

  mN1 = vpMatched12.size();

  mvX3Dc1.reserve(mN1);
  mvX3Dc2.reserve(mN1);
  vTc2ic2.reserve(mN1);
  mvnIndices1.reserve(mN1);

  auto Tc1w = pkf1->GetPose().cast<float>(); // pkf1->GetTcw().cast<float>();
  auto Tc2w =
      vpkf12[1]->GetPose().cast<float>(); // vpkf12[1]->GetTcw().cast<float>();
  auto Twc2 = Tc2w.inverse();

  mvAllIndices.reserve(mN1);

  size_t idx = 0;
  for (int i1 = 0; i1 < mN1; i1++) {
    if (vpMatched12[i1]) {
      MapPoint *pMP1 = vpKeyFrameMP1[i1];
      MapPoint *pMP2 = vpMatched12[i1]; // matched MP from pMP1

      // here pMP1=nullptr means pMP2=nullptr too before last SBP() in
      // ComputeSIm3()
      if (!pMP1)
        continue;
      CV_Assert(pMP2);
      if (pMP1->isBad() || pMP2->isBad())
        continue;

      int indexKF1 = i1;
      size_t ikf2 = vpkf12.size() > 1 + i1 ? 1 + i1 : 1;
      auto &pkf2_tmp = vpkf12[ikf2];
      auto Tc2iw = ikf2 > 1
                       ? pkf2_tmp->GetPose().cast<float>() // pkf2_tmp->GetTcw()
                       : Tc2w;
      auto indicesKF2_in = pMP2->GetIndexInKeyFrame(pkf2_tmp);
      vector<int> indicesKF2 = {get<0>(indicesKF2_in), get<1>(indicesKF2_in)};
      int cam2i = 0;
      for (auto indexKF2 : indicesKF2) {
        // it's for safe
        if (indexKF2 < 0)
          continue;

        const cv::KeyPoint &kp1 =
            //! usedistort_[0] ? pkf1->mvKeysUn[indexKF1] :
            //! pkf1->mvKeys[indexKF1];
            pkf1->NLeft == -1        ? pkf1->mvKeysUn[indexKF1]
            : indexKF1 < pkf1->NLeft ? pkf1->mvKeys[indexKF1]
                                     : pkf1->mvKeys[indexKF1 - pkf1->NLeft];
        int cam1i = 0;
        if (pkf1->NLeft == -1 || indexKF1 < pkf1->NLeft)
          cam1i = 0;
        else
          cam1i = 1;
        const cv::KeyPoint &kp2 =
            //! usedistort_[ikf2]? pkf2_tmp->mvKeysUn[indexKF2]:
            //! pkf2_tmp->mvKeys[indexKF2];
            pkf2_tmp->NLeft == -1 ? pkf2_tmp->mvKeysUn[indexKF2]
            : indexKF2 < pkf2_tmp->NLeft
                ? pkf2_tmp->mvKeys[indexKF2]
                : pkf2_tmp->mvKeys[indexKF2 - pkf2_tmp->NLeft];

        const float sigmaSquare1 = pkf1->mvLevelSigma2[kp1.octave];
        // pkf1->scalepyrinfo_.vlevelsigma2_[kp1.octave];
        const float sigmaSquare2 = pkf2_tmp->mvLevelSigma2[kp2.octave];
        // pkf2_tmp->scalepyrinfo_.vlevelsigma2_[kp2.octave];

        // to use chi2 distribution for e^2 with sigma^2, we need expand its
        // standard table like chi2(0.01,2)*sigma2
        mvnMaxError1.push_back(9.210 * sigmaSquare1);
        mvnMaxError2.push_back(9.210 * sigmaSquare2); // chi2(0.01,2)=9.21

        mvnIndices1.push_back(i1);

        auto X3D1w = pMP1->GetWorldPos().cast<float>();
        mvX3Dc1.push_back(Tc1w * X3D1w); // Xc1=(Tc1w*[Xw|1])(0:2)

        auto X3D2w = pMP2->GetWorldPos().cast<float>();
        mvX3Dc2.push_back(Tc2w * X3D2w); // Xc2
        vTc2ic2.emplace_back(Tc2iw * Twc2);

        mvAllIndices.push_back(idx);
        // if (usedistort_[0]) assert(pkf1->mapn2in_.size() > indexKF1);
        mapidx2cami_[0].push_back(usedistort_[0] ? offset_cam0[0] + cam1i
                                                 : offset_cam0[0]);
        // if (usedistort_[ikf2]) assert(pkf2_tmp->mapn2in_.size() > indexKF2);
        mapidx2cami_[1].push_back(usedistort_[ikf2] ? offset_cam0[ikf2] + cam2i
                                                    : offset_cam0[ikf2]);
        idx++;
        ++cam2i;
      }
    }
  }

  // maybe not using real 2d obs means camera obs won't be more accurate than
  // projecting for BE loop closing(after lots of opt.)
  Project(mvX3Dc1, mvP1im1, pcams_, mapidx2cami_[0]);
  Project(mvX3Dc2, mvP2im2, pcams_, mapidx2cami_[1], true);

  SetRansacParameters(); // use default settings for safe
}

void Sim3Solver::SetRansacParameters(double probability, int minInliers,
                                     int maxIterations) {
  mRansacProb = probability;
  mRansacMinInliers = minInliers;
  mRansacMaxIts = maxIterations;

  N = mvnIndices1.size(); // number of correspondences

  mvbInliersi.resize(N);

  // Adjust Parameters according to number of correspondences
  float epsilon = (float)mRansacMinInliers / N;

  // Set RANSAC iterations according to probability, epsilon, and max iterations
  int nIterations;

  if (mRansacMinInliers == N)
    nIterations = 1;
  else
    nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

  mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

  mnIterations = 0;
}

Sim3Solver::Matrix4f Sim3Solver::iterate(int nIterations, bool &bNoMore,
                                         vector<bool> &vbInliers, int &nInliers,
                                         bool &bconverge) {
  bconverge = false;
  bNoMore = false;
  vbInliers = vector<bool>(mN1, false);
  nInliers = 0;

  if (N < mRansacMinInliers) {
    PRINT_INFO_MUTEX("bNoMore by N: " << N << endl);
    bNoMore = true;
    return Matrix4f::Identity();
  }

  vector<size_t> vAvailableIndices;

  Eigen::Matrix3f P3Dc1i, P3Dc2i;

  int nCurrentIterations = 0;
  Eigen::Matrix4f bestSim3;
  while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations) {
    nCurrentIterations++;
    mnIterations++;

    vAvailableIndices = mvAllIndices;

    // Get min set of points
    for (short i = 0; i < 3; ++i) {
      int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

      int idx = vAvailableIndices[randi];

      P3Dc1i.col(i) = mvX3Dc1[idx];
      P3Dc2i.col(i) = mvX3Dc2[idx];

      // don't pick the same point, so log(1-p)/log(1-w^n) is just the upper
      // limit")" of the max iter.
      vAvailableIndices[randi] = vAvailableIndices.back();
      vAvailableIndices.pop_back();
    }

    ComputeSim3(P3Dc1i, P3Dc2i);

    CheckInliers();

    if (mnInliersi >= mnBestInliers) {
      mvbBestInliers = mvbInliersi;
      mnBestInliers = mnInliersi;
      mBestT12 = mT12i;
      mBestRotation = mR12i;
      mBestTranslation = mt12i;
      mBestScale = ms12i;

      if (mnInliersi > mRansacMinInliers) {
        nInliers = mnInliersi;
        for (int i = 0; i < N; i++)
          if (mvbInliersi[i])
            vbInliers[mvnIndices1[i]] = true;
        bconverge = true;
        return mBestT12;
      } else
        bestSim3 = mBestT12;
    }
  }

  if (mnIterations >= mRansacMaxIts)
    bNoMore = true;

  return bestSim3;
}

void Sim3Solver::ComputeCentroid(Matrix3f &P, Matrix3f &Pr, Vector3f &C) {
  // cv::reduce(P, C, 1, cv::REDUCE_SUM);
  C = P.rowwise().sum();
  C = C / P.cols();
  for (int i = 0; i < P.cols(); ++i)
    Pr.col(i) = P.col(i) - C;
}

void Sim3Solver::ComputeSim3(Matrix3f &P1, Matrix3f &P2) {
  // Custom implementation of:
  // Horn 1987, Closed-form solution of absolute orientataion using unit
  // quaternions

  // Step 1: Centroid and relative coordinates
  Eigen::Matrix3f Pr1; // Relative coordinates to centroid (set 1)
  Eigen::Matrix3f Pr2; // Relative coordinates to centroid (set 2)
  Eigen::Vector3f O1;  // Centroid of P1
  Eigen::Vector3f O2;  // Centroid of P2

  ComputeCentroid(P1, Pr1, O1);
  ComputeCentroid(P2, Pr2, O2);

  // Step 2: Compute M matrix
  Matrix3f M = Pr2 * Pr1.transpose();

  // Step 3: Compute N matrix
  double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;
  Matrix4f N;
  N11 = M(0, 0) + M(1, 1) + M(2, 2);
  N12 = M(1, 2) - M(2, 1);
  N13 = M(2, 0) - M(0, 2);
  N14 = M(0, 1) - M(1, 0);
  N22 = M(0, 0) - M(1, 1) - M(2, 2);
  N23 = M(0, 1) + M(1, 0);
  N24 = M(2, 0) + M(0, 2);
  N33 = -M(0, 0) + M(1, 1) - M(2, 2);
  N34 = M(1, 2) + M(2, 1);
  N44 = -M(0, 0) - M(1, 1) + M(2, 2);
  N << N11, N12, N13, N14, N12, N22, N23, N24, N13, N23, N33, N34, N14, N24,
      N34, N44;

  // Step 4: Eigenvector of the highest eigenvalue
  // evec[0] is the quaternion of the desired rotation
  // cv::eigen(N, eval, evec);
  Eigen::EigenSolver<Matrix4f> eigSolver;
  eigSolver.compute(N);
  Vector4f eval = eigSolver.eigenvalues().real();
  Matrix4f evec = eigSolver.eigenvectors().real();

  // extract imaginary part of the quaternion (sin*axis)
  // (evec.row(0).colRange(1, 4)).copyTo(vec);
  int maxIndex; // should be zero
  eval.maxCoeff(&maxIndex);
  Vector3f vec = evec.block<3, 1>(1, maxIndex);

  // Rotation angle. sin is the norm of the imaginary part, cos is the real part
  double ang = atan2(vec.norm(), evec(0, maxIndex));

  // Angle-axis representation. quaternion angle is the half
  vec = 2 * ang * vec / vec.norm();
  // computes the rotation matrix from angle-axis
  mR12i = Sophus::SO3exf::exp(vec).matrix();

  // Step 5: Rotate set 2
  Matrix3f P3 = mR12i * Pr2;

  // Step 6: Scale
  if (!mbFixScale) {
    double nom = (Pr1.array() * P3.array()).sum();
    Eigen::Array<float, 3, 3> aux_P3;
    aux_P3 = P3.array() * P3.array();
    double den = aux_P3.sum();

    ms12i = nom / den;
  } else
    ms12i = 1.0f;

  // Step 7: Translation
  mt12i = O1 - ms12i * mR12i * O2;

  // Step 8: Transformation
  // Step 8.1 T12
  mT12i.setIdentity();

  Matrix3f sR = ms12i * mR12i;
  mT12i.block<3, 3>(0, 0) = sR;
  mT12i.block<3, 1>(0, 3) = mt12i;

  // Step 8.2 T21
  mT21i.setIdentity();
  Matrix3f sRinv = (1.0 / ms12i) * mR12i.transpose();

  // sRinv.copyTo(mT21i.rowRange(0,3).colRange(0,3));
  mT21i.block<3, 3>(0, 0) = sRinv;

  Vector3f tinv = -sRinv * mt12i;
  mT21i.block<3, 1>(0, 3) = tinv;
}

void Sim3Solver::CheckInliers() {
  vector<Vector2f> vP1im2, vP2im1;
  Project(mvX3Dc2, vP2im1, pcams_, mapidx2cami_[0], false, &mT12i);
  Project(mvX3Dc1, vP1im2, pcams_, mapidx2cami_[1], true, &mT21i);

  mnInliersi = 0;

  for (size_t i = 0; i < mvP1im1.size(); i++) {
    Vector2f dist1 = mvP1im1[i] - vP2im1[i], dist2 = vP1im2[i] - mvP2im2[i];

    const float err1 = dist1.dot(dist1);
    const float err2 = dist2.dot(dist2);

    if (err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i]) {
      mvbInliersi[i] = true;
      mnInliersi++;
    } else
      mvbInliersi[i] = false;
  }
}

Sim3Solver::Matrix3f Sim3Solver::GetEstimatedRotation() {
  return mBestRotation;
}

Sim3Solver::Vector3f Sim3Solver::GetEstimatedTranslation() {
  return mBestTranslation;
}

float Sim3Solver::GetEstimatedScale() { return mBestScale; }

void Sim3Solver::Project(const vector<Vector3f> &vP3Dw, vector<Vector2f> &vP2D,
                         const vector<GeometricCamera *> &pcams,
                         vector<size_t> &mapidx2cami, bool bTcricr,
                         Matrix4f *pScrw) {
  Matrix4f Scrw;
  if (pScrw)
    Scrw = *pScrw;

  vP2D.clear();
  vP2D.reserve(vP3Dw.size());

  for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++) {
    Eigen::Vector3f crP3D = vP3Dw[i];
    if (pScrw)
      crP3D = Scrw.block<3, 3>(0, 0) * crP3D + Scrw.block<3, 1>(0, 3);
    if (bTcricr)
      crP3D = vTc2ic2[i] * crP3D;
    auto pcam = pcams[mapidx2cami[i]];
    Eigen::Vector3d cP3d = (mapidx2Tcr_[mapidx2cami[i]] * crP3D).cast<double>();

    auto p2dnorm = pcam->project(cP3d);
    vP2D.push_back(p2dnorm.cast<float>());
  }
}

} // namespace ORB_SLAM3
