/**
 * This file is part of VIEO_SLAM
 */

#ifndef SIM3SOLVER_H
#define SIM3SOLVER_H

#include "CameraModels/GeometricCamera.h"
#include <vector>

namespace ORB_SLAM3 {
class KeyFrame;
class MapPoint;

// 3D-2D solver, similar to PnPSolver
class Sim3Solver {
public:
  template <typename _Tp> using vector = std::vector<_Tp>;
  template <typename _Tp> using shared_ptr = std::shared_ptr<_Tp>;
  using Vector3d = Eigen::Vector3d;
  using Vector2f = Eigen::Vector2f;
  using Vector3f = Eigen::Vector3f;
  using Vector4f = Eigen::Vector4f;
  using Matrix3f = Eigen::Matrix3f;
  using Matrix4f = Eigen::Matrix4f;

  Sim3Solver(const vector<KeyFrame *> &vpkf12,
             const vector<MapPoint *> &vpMatched12,
             const bool bFixScale = true);

  void SetRansacParameters(double probability = 0.99, int minInliers = 6,
                           int maxIterations = 300);

  Matrix4f find(vector<bool> &vbInliers12, int &nInliers);

  Matrix4f iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers,
                   int &nInliers, bool &bconverge);

  Matrix3f GetEstimatedRotation();
  Vector3f GetEstimatedTranslation();
  float GetEstimatedScale();

protected:
  void ComputeCentroid(Matrix3f &P, Matrix3f &Pr, Vector3f &C);

  void ComputeSim3(Matrix3f &P1, Matrix3f &P2);

  void CheckInliers();

  void Project(const vector<Vector3f> &vP3Dw, vector<Vector2f> &vP2D,
               const vector<GeometricCamera *> &pcams,
               vector<size_t> &mapidx2cami, bool bTcricr = false,
               Matrix4f *pScrw = nullptr);

protected:
  // all the following vec has the same size<=mN1
  vector<Vector3f> mvX3Dc1; // matched MPs' Xc1
  vector<Vector3f> mvX3Dc2; // matched MPs' Xc2
  vector<Sophus::SE3exf> vTc2ic2;

  // Calibration
  vector<bool> usedistort_;
  vector<GeometricCamera *> pcams_;
  vector<Sophus::SE3exf> mapidx2Tcr_;
  vector<size_t> mapidx2cami_[2];
  vector<size_t> mvnIndices1; // matched MPs' index in pKF1

  // element is chi2(0.01,2)*sigma2 for pKF1->mvpMapPoints[i](matched ones have
  // MaxError1)
  vector<size_t> mvnMaxError1;
  vector<size_t>
      mvnMaxError2; // for pKF2->mvpMapPoints[j](matched ones have MaxError2)

  int N;
  int mN1; // number/size of the pkf1->mvpMapPoints

  // Current Estimation
  Matrix3f mR12i;
  Vector3f mt12i;
  float ms12i;
  Matrix4f mT12i;
  Matrix4f mT21i;
  vector<bool> mvbInliersi;
  int mnInliersi;

  // Current Ransac State
  int mnIterations;
  vector<bool> mvbBestInliers;
  int mnBestInliers;
  Matrix4f mBestT12;
  Matrix3f mBestRotation;
  Vector3f mBestTranslation;
  float mBestScale;

  // Scale is fixed to 1 in the stereo/RGBD case
  bool mbFixScale;

  // Indices for random selection
  // size is the number of matched 2 MPs <=mN1, recording the index of entering
  // vector(mvX3Dc1...)
  vector<size_t> mvAllIndices;

  // Projections, size is the same as mvX3Dc1.../mvAllIndices
  vector<Vector2f> mvP1im1; // image coordinate of matched MPs in pKF1
  vector<Vector2f> mvP2im2; // vec<matched MPs' image coordinate in pKF2>

  // RANSAC probability
  double mRansacProb;

  // RANSAC min inliers
  int mRansacMinInliers;

  // RANSAC max iterations
  int mRansacMaxIts;

  // Threshold inlier/outlier. e = dist(Pi,T_ij*Pj)^2 < 5.991*mSigma2
  float mTh;
  float mSigma2;
};

} // namespace ORB_SLAM3

#endif // SIM3SOLVER_H
