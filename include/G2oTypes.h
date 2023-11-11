/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef G2OTYPES_H
#define G2OTYPES_H

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_sba.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"
#include "Thirdparty/g2o/g2o/core/base_unary_edge.h"

#include<opencv2/core/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <Frame.h>
#include <KeyFrame.h>

#include"Converter.h"
#include <math.h>

#define TIMER_FLOW
#ifdef TIMER_FLOW
#include "common/mlog/timer.h"
using namespace VIEO_SLAM::mlog;
#endif
#include "common/camera_models/camera_base.h"
#include "common/navstate/nav_state.h"

namespace ORB_SLAM3
{

class KeyFrame;
class Frame;
class GeometricCamera;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 9, 1> Vector9d;
typedef Eigen::Matrix<double, 12, 1> Vector12d;
typedef Eigen::Matrix<double, 15, 1> Vector15d;
typedef Eigen::Matrix<double, 12, 12> Matrix12d;
typedef Eigen::Matrix<double, 15, 15> Matrix15d;
typedef Eigen::Matrix<double, 9, 9> Matrix9d;

Eigen::Matrix3d ExpSO3(const double x, const double y, const double z);
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w);

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R);

Eigen::Matrix3d InverseRightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &v);
Eigen::Matrix3d RightJacobianSO3(const double x, const double y, const double z);

Eigen::Matrix3d Skew(const Eigen::Vector3d &w);
Eigen::Matrix3d InverseRightJacobianSO3(const double x, const double y, const double z);

template<typename T = double>
Eigen::Matrix<T,3,3> NormalizeRotation(const Eigen::Matrix<T,3,3> &R) {
    Eigen::JacobiSVD<Eigen::Matrix<T,3,3>> svd(R,Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
}


class ImuCamPose
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuCamPose(){}
    ImuCamPose(KeyFrame* pKF);
    ImuCamPose(Frame* pF);
    ImuCamPose(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, KeyFrame* pKF);

    void SetParam(const std::vector<Eigen::Matrix3d> &_Rcw, const std::vector<Eigen::Vector3d> &_tcw, const std::vector<Eigen::Matrix3d> &_Rbc,
                  const std::vector<Eigen::Vector3d> &_tbc, const double &_bf);

    void Update(const double *pu); // update in the imu reference
    void UpdateW(const double *pu); // update in the world reference
    Eigen::Vector2d Project(const Eigen::Vector3d &Xw, int cam_idx=0) const; // Mono
    Eigen::Vector3d ProjectStereo(const Eigen::Vector3d &Xw, int cam_idx=0) const; // Stereo
    bool isDepthPositive(const Eigen::Vector3d &Xw, int cam_idx=0) const;

public:
    // For IMU
    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;

    // For set of cameras
    std::vector<Eigen::Matrix3d> Rcw;
    std::vector<Eigen::Vector3d> tcw;
    std::vector<Eigen::Matrix3d> Rcb, Rbc;
    std::vector<Eigen::Vector3d> tcb, tbc;
    double bf;
    std::vector<GeometricCamera*> pCamera;

    // For posegraph 4DoF
    Eigen::Matrix3d Rwb0;
    Eigen::Matrix3d DR;

    int its;
};

class InvDepthPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    InvDepthPoint(){}
    InvDepthPoint(double _rho, double _u, double _v, KeyFrame* pHostKF);

    void Update(const double *pu);

    double rho;
    double u, v; // they are not variables, observation in the host frame

    double fx, fy, cx, cy, bf; // from host frame

    int its;
};

// Optimizable parameters are IMU pose
class VertexPose : public g2o::BaseVertex<6,ImuCamPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose(){}
    VertexPose(KeyFrame* pKF){
        setEstimate(ImuCamPose(pKF));
    }
    VertexPose(Frame* pF){
        setEstimate(ImuCamPose(pF));
    }


    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        _estimate.Update(update_);
        updateCache();
    }
};

class VertexPose4DoF : public g2o::BaseVertex<4,ImuCamPose>
{
    // Translation and yaw are the only optimizable variables
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPose4DoF(){}
    VertexPose4DoF(KeyFrame* pKF){
        setEstimate(ImuCamPose(pKF));
    }
    VertexPose4DoF(Frame* pF){
        setEstimate(ImuCamPose(pF));
    }
    VertexPose4DoF(Eigen::Matrix3d &_Rwc, Eigen::Vector3d &_twc, KeyFrame* pKF){

        setEstimate(ImuCamPose(_Rwc, _twc, pKF));
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        double update6DoF[6];
        update6DoF[0] = 0;
        update6DoF[1] = 0;
        update6DoF[2] = update_[0];
        update6DoF[3] = update_[1];
        update6DoF[4] = update_[2];
        update6DoF[5] = update_[3];
        _estimate.UpdateW(update6DoF);
        updateCache();
    }
};

class VertexVelocity : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexVelocity(){}
    VertexVelocity(KeyFrame* pKF);
    VertexVelocity(Frame* pF);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d uv;
        uv << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uv);
    }
};

class VertexGyroBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGyroBias(){}
    VertexGyroBias(KeyFrame* pKF);
    VertexGyroBias(Frame* pF);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d ubg;
        ubg << update_[0], update_[1], update_[2];
        setEstimate(estimate()+ubg);
    }
};


class VertexAccBias : public g2o::BaseVertex<3,Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexAccBias(){}
    VertexAccBias(KeyFrame* pKF);
    VertexAccBias(Frame* pF);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        Eigen::Vector3d uba;
        uba << update_[0], update_[1], update_[2];
        setEstimate(estimate()+uba);
    }
};


// Gravity direction vertex
class GDirection
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GDirection(){}
    GDirection(Eigen::Matrix3d pRwg): Rwg(pRwg){}

    void Update(const double *pu)
    {
        Rwg=Rwg*ExpSO3(pu[0],pu[1],0.0);
    }

    Eigen::Matrix3d Rwg, Rgw;

    int its;
};

class VertexGDir : public g2o::BaseVertex<2,GDirection>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexGDir(){}
    VertexGDir(Eigen::Matrix3d pRwg){
        setEstimate(GDirection(pRwg));
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        _estimate.Update(update_);
        updateCache();
    }
};

// scale vertex
class VertexScale : public g2o::BaseVertex<1,double>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexScale(){
        setEstimate(1.0);
    }
    VertexScale(double ps){
        setEstimate(ps);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl(){
        setEstimate(1.0);
    }

    virtual void oplusImpl(const double *update_){
        setEstimate(estimate()*exp(*update_));
    }
};


// Inverse depth point (just one parameter, inverse depth at the host frame)
class VertexInvDepth : public g2o::BaseVertex<1,InvDepthPoint>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexInvDepth(){}
    VertexInvDepth(double invDepth, double u, double v, KeyFrame* pHostKF){
        setEstimate(InvDepthPoint(invDepth, u, v, pHostKF));
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    virtual void setToOriginImpl() {
        }

    virtual void oplusImpl(const double* update_){
        _estimate.Update(update_);
        updateCache();
    }
};

class EdgeMono : public g2o::BaseBinaryEdge<2,Eigen::Vector2d,g2o::VertexSBAPointXYZ,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef TIMER_FLOW
  double sum_dt_[2] = {0};
  int num_dt_[2] = {0};
  Timer timer_[2];
#endif

    EdgeMono(int cam_idx_=0): cam_idx(cam_idx_){
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
#ifdef TIMER_FLOW
      ++num_dt_[0];
      timer_[0].Start();
#endif
        const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().Project(VPoint->estimate(),cam_idx);
#ifdef TIMER_FLOW
      sum_dt_[0] += timer_[0].GetDTms(true);
#endif
    }


    virtual void linearizeOplus();

    bool isDepthPositive()
    {
        const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        return VPose->estimate().isDepthPositive(VPoint->estimate(),cam_idx);
    }

    Eigen::Matrix<double,2,9> GetJacobian(){
        linearizeOplus();
        Eigen::Matrix<double,2,9> J;
        J.block<2,3>(0,0) = _jacobianOplusXi;
        J.block<2,6>(0,3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double,9,9> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,2,9> J;
        J.block<2,3>(0,0) = _jacobianOplusXi;
        J.block<2,6>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }

public:
    const int cam_idx;
};

class EdgeMonoOnlyPose : public g2o::BaseUnaryEdge<2,Eigen::Vector2d,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeMonoOnlyPose(const Eigen::Vector3f &Xw_, int cam_idx_=0):Xw(Xw_.cast<double>()),
        cam_idx(cam_idx_){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        const Eigen::Vector2d obs(_measurement);
        _error = obs - VPose->estimate().Project(Xw,cam_idx);
    }

    virtual void linearizeOplus();

    bool isDepthPositive()
    {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        return VPose->estimate().isDepthPositive(Xw,cam_idx);
    }

    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        Eigen::Vector3d rho;
        bool robust = true;
        if (robust) {
            const g2o::RobustKernel* robustkernel = robustKernel();
            if (robustkernel)
                robustkernel->robustify(chi2(), rho);
            else
                robust = false;
        }
        const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
        return _jacobianOplusXi.transpose()*rinfo*_jacobianOplusXi;
        //return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }

public:
    const Eigen::Vector3d Xw;
    const int cam_idx;
};

class EdgeStereo : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,g2o::VertexSBAPointXYZ,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef TIMER_FLOW
  double sum_dt_[2] = {0};
  int num_dt_[2] = {0};
  Timer timer_[2];
#endif

    EdgeStereo(int cam_idx_=0): cam_idx(cam_idx_){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
#ifdef TIMER_FLOW
      ++num_dt_[0];
      timer_[0].Start();
#endif
        const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        const Eigen::Vector3d obs(_measurement);
        _error = obs - VPose->estimate().ProjectStereo(VPoint->estimate(),cam_idx);
#ifdef TIMER_FLOW
      sum_dt_[0] += timer_[0].GetDTms(true);
#endif
    }

    bool isDepthPositive()
    {
        const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
        return VPose->estimate().isDepthPositive(VPoint->estimate(),cam_idx);
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double,3,9> GetJacobian(){
        linearizeOplus();
        Eigen::Matrix<double,3,9> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,6>(0,3) = _jacobianOplusXj;
        return J;
    }

    Eigen::Matrix<double,9,9> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,3,9> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,6>(0,3) = _jacobianOplusXj;
        return J.transpose()*information()*J;
    }

public:
    const int cam_idx;
};

// extend edges to get H
typedef enum HessianExactMode { kExactNoRobust, kExactRobust, kNotExact } eHessianExactMode;
using namespace Eigen;
using namespace g2o;
template <int D, typename E, typename VertexXi, typename VertexXj>
class BaseBinaryEdgeEx : public BaseBinaryEdge<D, E, VertexXi, VertexXj> {
public:
#ifdef TIMER_FLOW
  double sum_dt_[2] = {0};
  int num_dt_[2] = {0};
  Timer timer_[2];
#endif

protected:
  using Base = BaseBinaryEdge<D, E, VertexXi, VertexXj>;
  using Base::_jacobianOplusXi;
  using Base::_jacobianOplusXj;
  using MatrixXid = Matrix<double, VertexXi::Dimension, VertexXi::Dimension>;
  using MatrixXjd = Matrix<double, VertexXj::Dimension, VertexXj::Dimension>;
  using MatrixXijd = Matrix<double, VertexXi::Dimension, VertexXj::Dimension>;
  using MatrixXjid = Matrix<double, VertexXj::Dimension, VertexXi::Dimension>;
#ifdef USE_G2O_NEWEST
  using Base::_hessianTuple;
  using Base::_hessianTupleTransposed;
#else
  using Base::_hessian;
  using Base::_hessianTransposed;
#endif
  using Base::_hessianRowMajor;
  using Base::robustInformation;

public:
  using Base::chi2;
  using Base::information;
  using Base::robustKernel;
#ifdef USE_G2O_NEWEST
  using typename Base::HessianTuple;
  using typename Base::HessianTupleTransposed;
  using JacobianXiOplusType =
      typename BaseFixedSizedEdge<D, E, VertexXi, VertexXj>::template JacobianType<D, VertexXi::Dimension>;
  using JacobianXjOplusType =
      typename BaseFixedSizedEdge<D, E, VertexXi, VertexXj>::template JacobianType<D, VertexXj::Dimension>;
#else
  using typename Base::HessianBlockTransposedType;
  using typename Base::HessianBlockType;
  using typename Base::JacobianXiOplusType;
  using typename Base::JacobianXjOplusType;
#endif
  using typename Base::InformationType;
  virtual void getRho(bool& robust, Vector3d& rho) const {
    if (robust) {
      const RobustKernel* robustkernel = robustKernel();
      if (robustkernel)
        robustkernel->robustify(chi2(), rho);
      else
        robust = false;
    }
  }

  virtual MatrixXid getHessianXi(bool robust = true) const {
    const JacobianXiOplusType& jac = _jacobianOplusXi;
    Vector3d rho;
    getRho(robust, rho);
    const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
    return jac.transpose() * rinfo * jac;
  }
  virtual MatrixXjd getHessianXj(bool robust = true) const {
    const JacobianXjOplusType& jac = _jacobianOplusXj;
    Vector3d rho;
    getRho(robust, rho);
    const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
    return jac.transpose() * rinfo * jac;
  }
  virtual MatrixXijd getHessianXij(int8_t exact_mode = (int8_t)kExactRobust) const {
    if ((int8_t)kNotExact == exact_mode) {
#ifdef USE_G2O_NEWEST
      if (_hessianRowMajor[0]) {
        return MatrixXjid(std::get<0>(_hessianTupleTransposed)).transpose();
#else
      if (_hessianRowMajor) {
        return MatrixXijd(_hessianTransposed.transpose());
#endif
      } else {
#ifdef USE_G2O_NEWEST
        return MatrixXijd(std::get<0>(_hessianTuple));
#else
        return MatrixXijd(_hessian);
#endif
      }
    } else {
      const JacobianXiOplusType& jaci = _jacobianOplusXi;
      const JacobianXjOplusType& jacj = _jacobianOplusXj;
      Vector3d rho;
      bool robust = (int8_t)kExactRobust == exact_mode;
      getRho(robust, rho);
      const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
      return jaci.transpose() * rinfo * jacj;
    }
  }
  virtual MatrixXjid getHessianXji(int8_t exact_mode = (int8_t)kExactRobust) const {
    if ((int8_t)kNotExact == exact_mode) {
#ifdef USE_G2O_NEWEST
      if (_hessianRowMajor[0]) {
        return MatrixXjid(std::get<0>(_hessianTupleTransposed));
#else
      if (_hessianRowMajor) {
        return MatrixXjid(_hessianTransposed);
#endif
      } else {
#ifdef USE_G2O_NEWEST
        return MatrixXijd(std::get<0>(_hessianTuple)).transpose();
#else
        return MatrixXjid(_hessian.transpose());
#endif
      }
    } else {
      const JacobianXiOplusType& jaci = _jacobianOplusXi;
      const JacobianXjOplusType& jacj = _jacobianOplusXj;
      Vector3d rho;
      bool robust = (int8_t)kExactRobust == exact_mode;
      getRho(robust, rho);
      const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
      return jacj.transpose() * rinfo * jaci;
    }
  }
};
class ImuCamInfo {
public:
  ImuCamInfo() {}
  ImuCamInfo(KeyFrame *pKF);

  // For IMU
  Eigen::Matrix3d Rwb_;
  Eigen::Vector3d pwb_;

  // to speedup
  // Eigen::aligned_vector<Sophus::SE3exd> vTcw_;
  std::vector<Matrix3d> vRcw_;
  std::vector<Vector3d> vtcw_;

  // Camera-IMU in/extrinsics
  // Eigen::aligned_vector<Sophus::SE3exd> vTcb_;
  std::vector<Matrix3d> vRcb_;
  std::vector<Vector3d> vtcb_;
  float bf_;
  typedef VIEO_SLAM::camm::Camera Camera;
  std::vector<Camera::Ptr> vintr_;

  // nswb then nsbw
  int sz_nswb_;

  void SetParams(const std::vector<Camera::Ptr>& vintr, const Sophus::SE3exd& Tcrb = Sophus::SE3exd(),
                 const float* bf = nullptr, int sz_nswb = -1) {
    vintr_ = vintr;
    int iend = (int)vintr.size();
    // vTcb_.reserve(iend);
    vRcb_.reserve(iend);
    vtcb_.reserve(iend);
    for (int i = 0; i < iend; ++i) {
      auto Tccr = vintr[i]->GetTcr().cast<double>();
      // vTcb_.emplace_back(Tccr * Tcrb);
      // vRcb_.emplace_back(vTcb_.back().rotationMatrix());
      auto Tcb = Tccr * Tcrb;
      vRcb_.emplace_back(Tcb.rotationMatrix());
      vtcb_.emplace_back(Tcb.translation());
    }
    // vTcw_ = vTcb_;
    vRcw_ = vRcb_;
    vtcw_ = vtcb_;
    if (bf) bf_ = *bf;
    if (sz_nswb < 0) sz_nswb = iend;
    sz_nswb_ = sz_nswb;
  }

  // Xw can also be Xh
  template <int MODE_OPT_VAR = 0>
  inline void GetTcw_wX(int cam_idx, Vector3d& Xw, double& scale, const ImuCamInfo* pposeh, Matrix3d& Rcw,
                        Vector3d& tcw, int camh_idx = 0, Vector3d* phX_unscale = nullptr, Matrix3d* pRwh = nullptr,
                        Vector3d* ptwh = nullptr) const {
    // twbh and twb has the same scale due to imu restrict=>only mp has scale problem, so no 2 scales opt. needed
    //  or we only consider visual factors: we suppose twbh and twb has the same scale or their diff. can be optimized
    //  =>sRcw when no Rwh else sRwh or 2MODE: 1/s*Rcw when no Rwh else 1/s*Rwh
    // s * Rcw(R12) * wX + tcw or 2MODE: 1/s * Rcw(R21) * wX + tcw(-1/s*R21*t12)
    {
      if (2 == MODE_OPT_VAR) {
        scale = 1. / scale;
        assert(sz_nswb_ <= cam_idx && "Please use Faster Vertex params / sz_nswb<=cam_idx!");
      } else {
        if (sz_nswb_ <= cam_idx) {
          PRINT_INFO_MUTEX("check sz=" << sz_nswb_ << ",cam_idx=" << cam_idx << std::endl);
        }
        assert(sz_nswb_ > cam_idx);
      }
      Rcw = vRcw_[cam_idx];  // vTcw_[cam_idx].rotationMatrix();
      tcw = vtcw_[cam_idx];  // vTcw_[cam_idx].translation();
      // notice that tcb&twb could be scale near 1 and Tcb is Tcicr used in MODE2
      if (2 == MODE_OPT_VAR && !pposeh) tcw *= scale;
    }
    if (phX_unscale) *phX_unscale = Xw;
    Xw *= scale;
    if (pposeh) {
      Matrix3d Rwh;
      Vector3d twh;
      {
        if (2 == MODE_OPT_VAR)
          assert(pposeh->sz_nswb_ <= camh_idx);
        else
          assert(pposeh->sz_nswb_ > camh_idx);
        auto Twh = Sophus::SE3exd(pposeh->vRcw_[camh_idx], pposeh->vtcw_[camh_idx])
            .inverse();  // pposeh->vTcw_[camh_idx].inverse();
        Rwh = Twh.rotationMatrix();
        twh = Twh.translation();

        if (pRwh) *pRwh = Rwh.matrix();
        if (2 == MODE_OPT_VAR) twh *= scale;
      }
      // wX=Twh*hX=Rwh*hX+twh=Rwb*Rbh*hX + (Rwb*tbh+twb)=Rwb(Rbh*hX + tbh) + twb
      Xw = Rwh * Xw + twh;

      if (ptwh) *ptwh = std::move(twh);
    }
  }

  using Vector2img = Eigen::Matrix<VIEO_SLAM::FLT_CAMM, 2, 1>;
  template <int DE>
  void cam_project(int cam_idx, const Vector3d& x_C, Eigen::Matrix<double, DE, 1>* pp2d,
                   Eigen::Matrix<double, DE, 3>* pJproj = nullptr) const {
    if (pp2d) {
      auto& res = *pp2d;
      Vector2img x_img;
      vintr_[cam_idx]->Project(x_C, &x_img);
      res.template segment<2>(0) = x_img.cast<double>();
      if (DE > 2) res[2] = res[0] - bf_ / x_C[2];
    }
    if (pJproj) {
      auto& Jproj = *pJproj;
      Matrix<double, 2, 3> Jproj_tmp;
      vintr_[cam_idx]->Project(x_C, nullptr, &Jproj_tmp);
      Jproj.template block<2, 3>(0, 0) = -Jproj_tmp;
      // ur=ul-b*fx/dl,dl=z => J_e_P'=J_e_Pc=-[fx/z 0 -fx*x/z^2; 0 fy/z -fy*y/z^2; fx/z 0 -fx*x/z^2+bf/z^2]
      if (DE > 2) {
        double invz = 1 / x_C[2], invz_2 = invz * invz;
        Jproj.template block<1, 3>(2, 0) << Jproj(0, 0), Jproj(0, 1), Jproj(0, 2) - bf_ * invz_2;
      }
    }
  }
  Eigen::Vector3d ProjectStereo(const Eigen::Vector3d &Xw, int cam_idx) const;

  void Update() {
    // Update camera poses
    auto Tbw = Sophus::SE3exd(Rwb_, pwb_);
    int iend = (int)vRcb_.size();  //(int)vTcb_.size();
    int iwb_end = std::min(iend, sz_nswb_), i = iwb_end;
    for (; i < iend; ++i) {
      // vTcw_[i] = vTcb_[i] * Tbw;
      // vRcw_[i] = vTcw_[i].rotationMatrix();
      vRcw_[i] = vRcb_[i] * Tbw.rotationMatrix();
      vtcw_[i] = vRcb_[i] * Tbw.translation() + vtcb_[i];
    }
    if (0 < iwb_end) {
      // Tbw = Tbw.inverse();
      for (i = 0; i < iwb_end; ++i) {
        // vTcw_[i] = vTcb_[i] * Tbw;
        // vRcw_[i] = vTcw_[i].rotationMatrix();
        Matrix3d Rbw = Tbw.rotationMatrix().transpose();
        vRcw_[i] = vRcb_[i] * Rbw;
        vtcw_[i] = vRcb_[i] * Rbw * -Tbw.translation() + vtcb_[i];
      }
    }
  }
};
class VertexNavStatePR : public BaseVertex<6, ImuCamInfo> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Base = BaseVertex<6, ImuCamInfo>;

  VertexNavStatePR() : Base() {}
  VertexNavStatePR(KeyFrame* pKF){
    setEstimate(ImuCamInfo(pKF));
  }

  // default constructor is enough
  virtual bool read(std::istream& is) { return true; }

  virtual bool write(std::ostream& os) const { return true; }

  // virtual
  void setToOriginImpl() {}
  void oplusImpl(const double* update_) {
    Eigen::Map<const Matrix<double, 6, 1>> update(update_);
    VIEO_SLAM::NavStateBased ns;
    ns.pwb_ = this->_estimate.pwb_;
    ns.setRwb(this->_estimate.Rwb_);
    ns.IncSmall(update);
    this->_estimate.pwb_ = ns.pwb_;
    this->_estimate.Rwb_ = ns.getRwb();
    this->_estimate.Update();
    updateCache();
  }

  typedef VIEO_SLAM::camm::Camera Camera;
  void SetParams(const std::vector<Camera::Ptr>& vintr, const Sophus::SE3exd& Tcrb = Sophus::SE3exd(),
                 const float* bf = nullptr, int sz_nswb = -1) {
    this->_estimate.SetParams(vintr, Tcrb, bf, sz_nswb);
  }
  void SetNavState(const VIEO_SLAM::NavStateBased& et) {
    auto& imucam_info = this->_estimate;
    imucam_info.pwb_ = et.pwb_;
    imucam_info.Rwb_ = et.Rwb_.matrix();
    assert(!imucam_info.vintr_.empty());
    imucam_info.Update();
  }
};
class EdgeReprojectPRStereo : public BaseBinaryEdgeEx<3, Matrix<double, 3, 1>, VertexSBAPointXYZ, VertexNavStatePR> {
  using VectorDEd = Matrix<double, 3, 1>;

  typedef BaseBinaryEdgeEx<3, VectorDEd, VertexSBAPointXYZ, VertexNavStatePR>
      Base;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeReprojectPRStereo(int cam_idx = 0) : Base(), cam_idx_(cam_idx) {}

  bool read(std::istream &is) { return true; }
  bool write(std::ostream &os) const { return true; }

  void computeError() override {
#ifdef TIMER_FLOW
    ++num_dt_[0];
    timer_[0].Start();
#endif
    const auto *VPoint =
        static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
    const auto *VPose = static_cast<const VertexNavStatePR *>(_vertices[1]);
    // auto &pose = VPose->estimate();
    const Eigen::Vector3d obs(_measurement);
    _error =
        obs - VPose->estimate().ProjectStereo(VPoint->estimate(), cam_idx_);
#ifdef TIMER_FLOW
    sum_dt_[0] += timer_[0].GetDTms(true);
#endif
  }
  void linearizeOplus() override;

  bool isDepthPositive()
      const { // unused in IMU motion-only BA, but used in localBA&GBA
    const auto *pXh = static_cast<const VertexSBAPointXYZ *>(_vertices[0]);
    Vector3d wX = pXh->estimate();
    const auto *vns = static_cast<const VertexNavStatePR *>(_vertices[1]);
    auto &pose = vns->estimate();
    double scale = 1.;
    Matrix3d Rcw = pose.vRcw_[cam_idx_];
    Vector3d tcw = pose.vtcw_[cam_idx_];
    //pose.GetTcw_wX(cam_idx_, wX, scale, nullptr, Rcw, tcw);
    return (Rcw * wX + tcw)(2) > 0.0; // Xc.z>0
  }

protected:
  int cam_idx_;

  using Base::_jacobianOplusXi;
  using Base::_jacobianOplusXj;
  using Base::_vertices;
};
class EdgeReprojectPR : public BaseBinaryEdgeEx<2, Matrix<double, 2, 1>, VertexSBAPointXYZ, VertexNavStatePR> {
  using VectorDEd = Matrix<double, 2, 1>;

  typedef BaseBinaryEdgeEx<2, VectorDEd, VertexSBAPointXYZ, VertexNavStatePR> Base;

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeReprojectPR(int cam_idx = 0) : Base(), cam_idx_(cam_idx) {}

  bool read(std::istream& is) { return true; }
  bool write(std::ostream& os) const { return true; }

  void computeError() override {
#ifdef TIMER_FLOW
    ++num_dt_[0];
    timer_[0].Start();
#endif
    const g2o::VertexSBAPointXYZ* VPoint = static_cast<const g2o::VertexSBAPointXYZ*>(_vertices[0]);
    const VertexNavStatePR* VPose = static_cast<const VertexNavStatePR*>(_vertices[1]);
    const Eigen::Vector2d obs(_measurement);
    Eigen::Vector2f p2d;
    VPose->estimate().vintr_[cam_idx_]->Project(
        VPose->estimate().vRcw_[cam_idx_] * VPoint->estimate() + VPose->estimate().vtcw_[cam_idx_], &p2d);
    _error = obs - p2d.cast<double>();  // VPose->estimate().Project(VPoint->estimate(),cam_idx);
    /*
    const auto* pXh = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);  // Xh/Ph
    Vector3d wX = pXh->estimate();
    // Tbs_w, bs is b for slam
    const auto* vns = static_cast<const VertexNavStatePR*>(_vertices[1]);
    double scale = 1.;
    Matrix3d Rcw;
    Vector3d tcw;
    assert(vns->pimucam_info_);
    vns->pimucam_info_->GetTcw_wX(cam_idx_, wX, scale, nullptr, Rcw, tcw);
    // Pc=Tcb*Tbw*Pw=Rcb*Rbw*Pw+Rcb*tbw(-Rcb*Rbw*twb)+tcb(-Rcb*tbc)=Rcb*Rbw*(Pw-twb)+tcb;
    VectorDEd p2d;
    vns->pimucam_info_->cam_project(cam_idx_, Rcw * wX + tcw, &p2d);
    this->_error = this->_measurement - p2d;*/
#ifdef TIMER_FLOW
    sum_dt_[0] += timer_[0].GetDTms(true);
#endif
  }
  void linearizeOplus() override;

  bool isDepthPositive() const {  // unused in IMU motion-only BA, but used in localBA&GBA
    const auto* pXh = static_cast<const VertexSBAPointXYZ*>(_vertices[0]);
    Vector3d wX = pXh->estimate();
    const auto* vns = static_cast<const VertexNavStatePR*>(_vertices[1]);
    auto& pose = vns->estimate();
    double scale = 1.;
    Matrix3d Rcw;
    Vector3d tcw;
    pose.GetTcw_wX(cam_idx_, wX, scale, nullptr, Rcw, tcw);
    return (Rcw * wX + tcw)(2) > 0.0;  // Xc.z>0
  }

protected:
  int cam_idx_;

  using Base::_jacobianOplusXi;
  using Base::_jacobianOplusXj;
  using Base::_vertices;
};

class EdgeStereoOnlyPose : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeStereoOnlyPose(const Eigen::Vector3f &Xw_, int cam_idx_=0):
        Xw(Xw_.cast<double>()), cam_idx(cam_idx_){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        const Eigen::Vector3d obs(_measurement);
        _error = obs - VPose->estimate().ProjectStereo(Xw, cam_idx);
    }

    bool isDepthPositive()
    {
        const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
        return VPose->estimate().isDepthPositive(Xw,cam_idx);
    }

    virtual void linearizeOplus();

    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }

public:
    const Eigen::Vector3d Xw; // 3D point coordinates
    const int cam_idx;
};

class EdgeInertial : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef TIMER_FLOW
  double sum_dt_[2] = {0};
  int num_dt_[2] = {0};
  Timer timer_[2];
#endif

    EdgeInertial(IMU::Preintegrated* pInt);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    Eigen::Matrix<double,24,24> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,9,24> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        Eigen::Vector3d rho;
        bool robust = true;
        if (robust) {
            const g2o::RobustKernel* robustkernel = robustKernel();
            if (robustkernel)
                robustkernel->robustify(chi2(), rho);
            else
                robust = false;
        }
        const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
        return J.transpose()*rinfo*J;
    }

    Eigen::Matrix<double,18,18> GetHessianNoPose1(){
        linearizeOplus();
        Eigen::Matrix<double,9,18> J;
        J.block<9,3>(0,0) = _jacobianOplus[1];
        J.block<9,3>(0,3) = _jacobianOplus[2];
        J.block<9,3>(0,6) = _jacobianOplus[3];
        J.block<9,6>(0,9) = _jacobianOplus[4];
        J.block<9,3>(0,15) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,9,9> GetHessian2(){
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,6>(0,0) = _jacobianOplus[4];
        J.block<9,3>(0,6) = _jacobianOplus[5];
        return J.transpose()*information()*J;
    }

    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    IMU::Preintegrated* mpInt;
    const double dt;
    Eigen::Vector3d g;
};


class EdgeInertial2 : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef TIMER_FLOW
  double sum_dt_[2] = {0};
  int num_dt_[2] = {0};
  Timer timer_[2];
#endif

  EdgeInertial2(IMU::Preintegrated* pInt);

  virtual bool read(std::istream& is){return false;}
  virtual bool write(std::ostream& os) const{return false;}

  void computeError();
  virtual void linearizeOplus();

  Eigen::Matrix<double,24,24> GetHessian(){
    linearizeOplus();
    Eigen::Matrix<double,9,24> J;
    J.block<9,6>(0,0) = _jacobianOplus[0];
    J.block<9,3>(0,6) = _jacobianOplus[1];
    J.block<9,3>(0,9) = _jacobianOplus[2];
    J.block<9,3>(0,12) = _jacobianOplus[3];
    J.block<9,6>(0,15) = _jacobianOplus[4];
    J.block<9,3>(0,21) = _jacobianOplus[5];
    Eigen::Vector3d rho;
    bool robust = true;
    if (robust) {
      const g2o::RobustKernel* robustkernel = robustKernel();
      if (robustkernel)
        robustkernel->robustify(chi2(), rho);
      else
        robust = false;
    }
    const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
    return J.transpose()*rinfo*J;
  }

  Eigen::Matrix<double,18,18> GetHessianNoPose1(){
    linearizeOplus();
    Eigen::Matrix<double,9,18> J;
    J.block<9,3>(0,0) = _jacobianOplus[1];
    J.block<9,3>(0,3) = _jacobianOplus[2];
    J.block<9,3>(0,6) = _jacobianOplus[3];
    J.block<9,6>(0,9) = _jacobianOplus[4];
    J.block<9,3>(0,15) = _jacobianOplus[5];
    return J.transpose()*information()*J;
  }

  Eigen::Matrix<double,9,9> GetHessian2(){
    linearizeOplus();
    Eigen::Matrix<double,9,9> J;
    J.block<9,6>(0,0) = _jacobianOplus[4];
    J.block<9,3>(0,6) = _jacobianOplus[5];
    return J.transpose()*information()*J;
  }

  const Eigen::Matrix3d JRg, JVg, JPg;
  const Eigen::Matrix3d JVa, JPa;
  IMU::Preintegrated* mpInt;
  const double dt;
  Eigen::Vector3d g;
};


// Edge inertial whre gravity is included as optimizable variable and it is not supposed to be pointing in -z axis, as well as scale
class EdgeInertialGS : public g2o::BaseMultiEdge<9,Vector9d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // EdgeInertialGS(IMU::Preintegrated* pInt);
    EdgeInertialGS(IMU::Preintegrated* pInt);

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError();
    virtual void linearizeOplus();

    const Eigen::Matrix3d JRg, JVg, JPg;
    const Eigen::Matrix3d JVa, JPa;
    IMU::Preintegrated* mpInt;
    const double dt;
    Eigen::Vector3d g, gI;

    Eigen::Matrix<double,27,27> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,9,27> J;
        J.block<9,6>(0,0) = _jacobianOplus[0];
        J.block<9,3>(0,6) = _jacobianOplus[1];
        J.block<9,3>(0,9) = _jacobianOplus[2];
        J.block<9,3>(0,12) = _jacobianOplus[3];
        J.block<9,6>(0,15) = _jacobianOplus[4];
        J.block<9,3>(0,21) = _jacobianOplus[5];
        J.block<9,2>(0,24) = _jacobianOplus[6];
        J.block<9,1>(0,26) = _jacobianOplus[7];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,27,27> GetHessian2(){
        linearizeOplus();
        Eigen::Matrix<double,9,27> J;
        J.block<9,3>(0,0) = _jacobianOplus[2];
        J.block<9,3>(0,3) = _jacobianOplus[3];
        J.block<9,2>(0,6) = _jacobianOplus[6];
        J.block<9,1>(0,8) = _jacobianOplus[7];
        J.block<9,3>(0,9) = _jacobianOplus[1];
        J.block<9,3>(0,12) = _jacobianOplus[5];
        J.block<9,6>(0,15) = _jacobianOplus[0];
        J.block<9,6>(0,21) = _jacobianOplus[4];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,9,9> GetHessian3(){
        linearizeOplus();
        Eigen::Matrix<double,9,9> J;
        J.block<9,3>(0,0) = _jacobianOplus[2];
        J.block<9,3>(0,3) = _jacobianOplus[3];
        J.block<9,2>(0,6) = _jacobianOplus[6];
        J.block<9,1>(0,8) = _jacobianOplus[7];
        return J.transpose()*information()*J;
    }



    Eigen::Matrix<double,1,1> GetHessianScale(){
        linearizeOplus();
        Eigen::Matrix<double,9,1> J = _jacobianOplus[7];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,3,3> GetHessianBiasGyro(){
        linearizeOplus();
        Eigen::Matrix<double,9,3> J = _jacobianOplus[2];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,3,3> GetHessianBiasAcc(){
        linearizeOplus();
        Eigen::Matrix<double,9,3> J = _jacobianOplus[3];
        return J.transpose()*information()*J;
    }

    Eigen::Matrix<double,2,2> GetHessianGDir(){
        linearizeOplus();
        Eigen::Matrix<double,9,2> J = _jacobianOplus[6];
        return J.transpose()*information()*J;
    }
};



class EdgeGyroRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexGyroBias,VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef TIMER_FLOW
  double sum_dt_[2] = {0};
  int num_dt_[2] = {0};
  Timer timer_[2];
#endif

    EdgeGyroRW(){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
#ifdef TIMER_FLOW
      ++num_dt_[0];
      timer_[0].Start();
#endif
        const VertexGyroBias* VG1= static_cast<const VertexGyroBias*>(_vertices[0]);
        const VertexGyroBias* VG2= static_cast<const VertexGyroBias*>(_vertices[1]);
        _error = VG2->estimate()-VG1->estimate();
#ifdef TIMER_FLOW
      sum_dt_[0] += timer_[0].GetDTms(true);
#endif
    }

    virtual void linearizeOplus(){
#ifdef TIMER_FLOW
      ++num_dt_[1];
      timer_[1].Start();
#endif
        _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        _jacobianOplusXj.setIdentity();
#ifdef TIMER_FLOW
      sum_dt_[1] += timer_[1].GetDTms(true);
#endif
    }

    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,3,6> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,3>(0,3) = _jacobianOplusXj;
        Eigen::Vector3d rho;
        bool robust = true;
        if (robust) {
            const g2o::RobustKernel* robustkernel = robustKernel();
            if (robustkernel)
                robustkernel->robustify(chi2(), rho);
            else
                robust = false;
        }
        const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
        return J.transpose()*rinfo*J;
    }

    Eigen::Matrix3d GetHessian2(){
        linearizeOplus();
        return _jacobianOplusXj.transpose()*information()*_jacobianOplusXj;
    }
};


class EdgeAccRW : public g2o::BaseBinaryEdge<3,Eigen::Vector3d,VertexAccBias,VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef TIMER_FLOW
  double sum_dt_[2] = {0};
  int num_dt_[2] = {0};
  Timer timer_[2];
#endif

    EdgeAccRW(){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
#ifdef TIMER_FLOW
      ++num_dt_[0];
      timer_[0].Start();
#endif
        const VertexAccBias* VA1= static_cast<const VertexAccBias*>(_vertices[0]);
        const VertexAccBias* VA2= static_cast<const VertexAccBias*>(_vertices[1]);
        _error = VA2->estimate()-VA1->estimate();
#ifdef TIMER_FLOW
      sum_dt_[0] += timer_[0].GetDTms(true);
#endif
    }

    virtual void linearizeOplus(){
#ifdef TIMER_FLOW
      ++num_dt_[1];
      timer_[1].Start();
#endif
        _jacobianOplusXi = -Eigen::Matrix3d::Identity();
        _jacobianOplusXj.setIdentity();
#ifdef TIMER_FLOW
      sum_dt_[1] += timer_[1].GetDTms(true);
#endif
    }

    Eigen::Matrix<double,6,6> GetHessian(){
        linearizeOplus();
        Eigen::Matrix<double,3,6> J;
        J.block<3,3>(0,0) = _jacobianOplusXi;
        J.block<3,3>(0,3) = _jacobianOplusXj;
        Eigen::Vector3d rho;
        bool robust = true;
        if (robust) {
            const g2o::RobustKernel* robustkernel = robustKernel();
            if (robustkernel)
                robustkernel->robustify(chi2(), rho);
            else
                robust = false;
        }
        const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
        return J.transpose()*rinfo*J;
    }

    Eigen::Matrix3d GetHessian2(){
        linearizeOplus();
        return _jacobianOplusXj.transpose()*information()*_jacobianOplusXj;
    }
};

class ConstraintPoseImu
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ConstraintPoseImu(const Eigen::Matrix3d &Rwb_, const Eigen::Vector3d &twb_, const Eigen::Vector3d &vwb_,
                       const Eigen::Vector3d &bg_, const Eigen::Vector3d &ba_, const Matrix15d &H_):
                       Rwb(Rwb_), twb(twb_), vwb(vwb_), bg(bg_), ba(ba_), H(H_)
    {
        H = (H+H)/2;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,15,15> > es(H);
        Eigen::Matrix<double,15,1> eigs = es.eigenvalues();
        for(int i=0;i<15;i++)
            if(eigs[i]<1e-12)
                eigs[i]=0;
        H = es.eigenvectors()*eigs.asDiagonal()*es.eigenvectors().transpose();
    }

    Eigen::Matrix3d Rwb;
    Eigen::Vector3d twb;
    Eigen::Vector3d vwb;
    Eigen::Vector3d bg;
    Eigen::Vector3d ba;
    Matrix15d H;
};

class EdgePriorPoseImu : public g2o::BaseMultiEdge<15,Vector15d>
{
public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgePriorPoseImu(ConstraintPoseImu* c);

        virtual bool read(std::istream& is){return false;}
        virtual bool write(std::ostream& os) const{return false;}

        void computeError();
        virtual void linearizeOplus();

        Eigen::Matrix<double,15,15> GetHessian(){
            linearizeOplus();
            Eigen::Matrix<double,15,15> J;
            J.block<15,6>(0,0) = _jacobianOplus[0];
            J.block<15,3>(0,6) = _jacobianOplus[1];
            J.block<15,3>(0,9) = _jacobianOplus[2];
            J.block<15,3>(0,12) = _jacobianOplus[3];
            Eigen::Vector3d rho;
            bool robust = true;
            if (robust) {
                const g2o::RobustKernel* robustkernel = robustKernel();
                if (robustkernel)
                    robustkernel->robustify(chi2(), rho);
                else
                    robust = false;
            }
            const InformationType& rinfo = robust ? InformationType(rho[1] * information()) : information();
            return J.transpose()*rinfo*J;
        }

        Eigen::Matrix<double,9,9> GetHessianNoPose(){
            linearizeOplus();
            Eigen::Matrix<double,15,9> J;
            J.block<15,3>(0,0) = _jacobianOplus[1];
            J.block<15,3>(0,3) = _jacobianOplus[2];
            J.block<15,3>(0,6) = _jacobianOplus[3];
            return J.transpose()*information()*J;
        }
        Eigen::Matrix3d Rwb;
        Eigen::Vector3d twb, vwb;
        Eigen::Vector3d bg, ba;
};

// Priors for biases
class EdgePriorAcc : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexAccBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePriorAcc(const Eigen::Vector3f &bprior_):bprior(bprior_.cast<double>()){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexAccBias* VA = static_cast<const VertexAccBias*>(_vertices[0]);
        _error = bprior - VA->estimate();
    }
    virtual void linearizeOplus();

    Eigen::Matrix<double,3,3> GetHessian(){
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }

    const Eigen::Vector3d bprior;
};

class EdgePriorGyro : public g2o::BaseUnaryEdge<3,Eigen::Vector3d,VertexGyroBias>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgePriorGyro(const Eigen::Vector3f &bprior_):bprior(bprior_.cast<double>()){}

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexGyroBias* VG = static_cast<const VertexGyroBias*>(_vertices[0]);
        _error = bprior - VG->estimate();
    }
    virtual void linearizeOplus();

    Eigen::Matrix<double,3,3> GetHessian(){
        linearizeOplus();
        return _jacobianOplusXi.transpose()*information()*_jacobianOplusXi;
    }

    const Eigen::Vector3d bprior;
};


class Edge4DoF : public g2o::BaseBinaryEdge<6,Vector6d,VertexPose4DoF,VertexPose4DoF>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Edge4DoF(const Eigen::Matrix4d &deltaT){
        dTij = deltaT;
        dRij = deltaT.block<3,3>(0,0);
        dtij = deltaT.block<3,1>(0,3);
    }

    virtual bool read(std::istream& is){return false;}
    virtual bool write(std::ostream& os) const{return false;}

    void computeError(){
        const VertexPose4DoF* VPi = static_cast<const VertexPose4DoF*>(_vertices[0]);
        const VertexPose4DoF* VPj = static_cast<const VertexPose4DoF*>(_vertices[1]);
        _error << LogSO3(VPi->estimate().Rcw[0]*VPj->estimate().Rcw[0].transpose()*dRij.transpose()),
                 VPi->estimate().Rcw[0]*(-VPj->estimate().Rcw[0].transpose()*VPj->estimate().tcw[0])+VPi->estimate().tcw[0] - dtij;
    }

    // virtual void linearizeOplus(); // numerical implementation

    Eigen::Matrix4d dTij;
    Eigen::Matrix3d dRij;
    Eigen::Vector3d dtij;
};

} //namespace ORB_SLAM2

#endif // G2OTYPES_H
