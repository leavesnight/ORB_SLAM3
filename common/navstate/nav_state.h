//
// Created by leavesnight on 10/1/17. Inspired by Jing Wang
//
#pragma once

#include "common/sophus/so3_extra.h"

//#define USE_PR_SEPARATE
#ifndef USE_PR_SEPARATE
// USE_SE3_LD_UPDATE for prior more accurate:
// 1.T*(X _+ dx)=Texp(dx)X=exp(adjT*dx)TX=TX _+ adjT*Tdx=TX _+ FT(Tdx), adjT=[R t^R;0 R]
// 2.only Left Disturbance model/Right Invariant Model could let null space N of
// observability M to be unrelated to state vector
// (feature pos could be unrelated by the same LD retraction form of p/v, but O(l)->O(l^2),
// so prior try to not include feature pos, then f can use norml f+df retraction with O(l))
//#define USE_SE3_LD_UPDATE
#endif
#ifndef USE_SE3_LD_UPDATE
/*
 * if LD:R<-dR*R from RD:R<-R*dR's Jacobians: for LD:R<-Exp(dphi_ld)R=RExp(RT*dphi_ld)
 * =>dphi_rd=f_rd(dphi_ld)=RT*dphi_ld, so by Chain Rules: de_dphi_ld=de_dphi_rd*RT+...
 * where ...=de_dp*dp_dphi_ld+de_dv*dv_dphi_ld+dbias_dphi_ld(usually 0), for e(p,phi,v,bias)
 * for now used FLD:p<-dR*p+dp, v<-dv*v+dv; we've derived the RD(not Full) in the handlers paper:
 * RD:p<-p+dp, v<-v+dv, where FRD:p<-p+Rdp, v<-v+dv, so the Chan Rules become:
 * de_dphi_ld = de_dphi_rd * RT + de_dp_rd * (-p_ld^) + de_dv_rd * (-v_ld^)
 * de_dp_fld = de_dp_rd, de_dv_fld = de_dv_rd(FLD Jto_p,v,bias is the same as RD)
 */
//#define USE_LD_DRR
#if not defined(USE_LD_DRR) and not defined(USE_PR_SEPARATE)
#define USE_P_PLUS_RDP
#endif
#endif

namespace VIEO_SLAM {

// navigation state xj for IMU, and my encoder state is included
// suggested PR&V optimization order but saved order is RPV like SE3(save r-rho but Log rho-r),
// w usually means world frame, but can also mean master SLAM body frame(b then from body frame->slave body frame)
template <class Tdata, class Tcalc = double>
class NavStateBase {
 public:
  // for quaterniond in SO3ex
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /*template <typename _Tp>
  using shared_ptr = std::shared_ptr<_Tp>;
  using Ptr = shared_ptr<NavStateBase>;*/
  template <typename _Scalar, int _Rows, int _Cols>
  using Matrix = Eigen::Matrix<_Scalar, _Rows, _Cols>;
  using Vec3data = Matrix<Tdata, 3, 1>;
  using Vec6data = Matrix<Tdata, 6, 1>;
  using Mat3data = Matrix<Tdata, 3, 3>;
  using SO3data = Sophus::SO3ex<Tdata>;
  using Vec3calc = Matrix<Tcalc, 3, 1>;
  using Vec6calc = Matrix<Tcalc, 6, 1>;
  using Mat3calc = Matrix<Tcalc, 3, 3>;
  using SO3calc = Sophus::SO3ex<Tcalc>;

  // notice w(inertial) can be b_slam(non-inertial), b can mean bh for handler
  // rotation Rwbj=Rwb(tj)=qwbj=wqwb(tj)=Phiwbj=wPhiwb(tj), public for g2otypes, use so3ex to store is compact
  SO3data Rwb_;
  // position pwbj=wpwb(tj) or twbj, not use se3ex to store for rho transformed to p is slower
  Vec3data pwb_ = Vec3data::Zero();
  // velocity vwbj=wvwb(tj)
  Vec3data vwb_ = Vec3data::Zero();
  // update below term during optimization(or exactly the dbg=bg_-bg_i_bar_/dba=ba_-ba_i_bar_ part), d means small delta
  // unchanged bg/ba is bg/ba_i_bar in preint class during to avoid multithread problem when update bg in SetBadFlag
  Vec3data bg_ = Vec3data::Zero();  // bias of gyroscope bgj_bar=bg(tj)_bar
  Vec3data ba_ = Vec3data::Zero();  // bias of accelerometer baj_bar=ba(tj)_bar

#ifdef USE_LD_DRR
  const double *dxr_ = nullptr;
#endif

  NavStateBase() {}
  // for Eigen has deep copy, we don't define default copy constructor and operator= and we don't need ~NavState(),
  // so move copy constructor and move operator= will also be default

  Mat3data getRwb() const { return Rwb_.matrix(); }  // get rotation matrix of Rwbj, const is very important!
  // if SO3ex could be used, it will be safer here
  void setRwb(const Mat3data &Rwb) { Rwb_ = Sophus::SO3ex<Tdata>(Rwb); }
  Vec6data getBias() const {
    Vec6data bias;
    bias.template segment<3>(0) = bg_, bias.template segment<3>(3) = ba_;
    return bias;
  }
  void setBias(const Vec6data &bias) { bg_ = bias.template segment<3>(0), ba_ = bias.template segment<3>(3); }

  // incremental addition, dx = [dP, dV, dPhi, dBa, dBg] for oplusImpl(), see Manifold paper (70)
  template <int PRV = 3>
  void IncSmall(const Eigen::Map<const Matrix<Tcalc, 6, 1>> &dpr) {  // also inline
    Vec3calc pwb = pwb_.template cast<Tcalc>();
    SO3calc Rwb = Rwb_.template cast<Tcalc>();
    const auto &dphi = dpr.template segment<3>(3);
    const auto &dR = SO3calc::exp(dphi);
    const auto &dp = dpr.template segment<3>(0);
#ifdef USE_LD_DRR
    // enter PR means drho,dphi, where Jl(dphi)drho=dp
#ifdef USE_SE3_LD_UPDATE
    Vec3calc dpprime = SO3calc::JacobianL(dphi) * dp;
    pwb = dR * pwb + dpprime;
#else
    pwb = dR * pwb + dp;
#endif
    Rwb = dR * Rwb;  // left distrubance model
#else
#ifdef USE_P_PLUS_RDP
    pwb += Rwb * dp;  // here dp<-p+R*dp(in paper On-Manifold Preintegration)
#else
    pwb += dp;  // here p<-p+dp, dp=R*dp(in paper On-Manifold Preintegration)
#endif
    Rwb *= dR;  // right distrubance model
#endif
    pwb_ = pwb.template cast<Tdata>();
    Rwb_ = Rwb.template cast<Tdata>();
  }
  // use overload to implement part specialization; dr for LD
  template <int PRV = 2>
  void IncSmall(const Eigen::Map<const Vec3calc> &dv) {
    switch (PRV) {
      case 0: {
        const Eigen::Map<const Vec3calc> &dp = dv;
        Vec3calc pwb = pwb_.template cast<Tcalc>();
#ifdef USE_LD_DRR
        Vec3calc dr = Vec3calc::Zero();
        if (dxr_) dr = Vec3calc(dxr_);
        const auto &dR = SO3calc::exp(dr);
#ifdef USE_SE3_LD_UPDATE
        Vec3calc dpprime = SO3calc::JacobianL(dr) * dp;  // vr/fr also this op.
        pwb = dR * pwb + dpprime;
#else
        pwb = dR * pwb + dp;
#endif
#else
#ifdef USE_P_PLUS_RDP
        // it may be better to IncSmall<1> after IncSmall<0>
        pwb += Rwb_ * dp;  // here dp<-p+R*dp(in paper On-Manifold Preintegration)
#else
        pwb += dp;  // here p<-p+dp, dp=R*dp(in paper On-Manifold Preintegration)
#endif
#endif
        pwb_ = pwb.template cast<Tdata>();
      } break;
      case 1: {
        const Eigen::Map<const Vec3calc> &dr = dv;
        SO3calc Rwb = Rwb_.template cast<Tcalc>();
#ifdef USE_LD_DRR
        Rwb = SO3calc::exp(dr) * Rwb;  // left distrubance model
#else
        Rwb *= SO3calc::exp(dr);  // right distrubance model
#endif
        Rwb_ = Rwb.template cast<Tdata>();
      } break;
      default: {
        Vec3calc vwb = vwb_.template cast<Tcalc>();
#ifdef USE_LD_DRR
        Vec3calc dr = Vec3calc::Zero();
        if (dxr_) dr = Vec3calc(dxr_);
        const auto &dR = SO3calc::exp(dr);
#ifdef USE_SE3_LD_UPDATE
        Vec3calc dvprime = SO3calc::JacobianL(dr) * dv;  // pr/fr also this op.
        vwb = dR * vwb + dvprime;
#else
        vwb = dR * vwb + dv;
#endif
#else
        vwb += dv;
#endif
        vwb_ = vwb.template cast<Tdata>();
      }
    }
  }

  // for we hope to include as few headers as possible, we not ifdef USE_TB_BH_OPT here
  inline void TransToNsBrB(const NavStateBase &ns_fixed, SO3calc &R, Vec3calc &p, Vec3calc *pv = nullptr) const {
    // get Tbbh/Tbrb+vbbh/vbrb
    SO3calc Rbrw = ns_fixed.Rwb_.inverse();
    R = Rbrw * Rwb_;
    p = Rbrw * (pwb_ - ns_fixed.pwb_);
    if (pv) *pv = Rbrw * (vwb_ - ns_fixed.vwb_);  // just part of dp1/dt
  }
  inline void TransToNsBrB(const NavStateBase &ns_fixed, Mat3calc &R, Vec3calc &p, Vec3calc *pv = nullptr) const {
    SO3calc R_so3;
    TransToNsBrB(ns_fixed, R_so3, p, pv);
    R = R_so3.matrix();
  }
  inline void TransFromNsBrB(const SO3calc &Rwbr, const Vec3calc &pwbr, SO3calc &R, Vec3calc &p,
                             Vec3calc *pvwbr = nullptr, Vec3calc *pv = nullptr) const {
    // get Twbh=Twb*Tbbh/Twb=Twbr*Tbrb
    R = Rwbr * Rwb_;
    p = Rwbr * pwb_ + pwbr;
    if (pv && pvwbr) *pv = Rwbr * vwb_ + *pvwbr;  // just part of dp1/dt
  }

  virtual std::string FormatString(bool bfull = true) const {
    if (bfull)
      return mlog::FormatString("p={};r={};v={};bg={};ba={}", pwb_.transpose(), Rwb_.log().transpose(),
                                vwb_.transpose(), bg_.transpose(), ba_.transpose());
    else
      return mlog::FormatString("p={};r={};v={};bg={};ba={}", pwb_.transpose(), Rwb_.log().transpose(),
                                vwb_.transpose(), bg_.norm(), ba_.norm());
  }
};

template <class Tdata, class Tcalc = double>
class NavState : public NavStateBase<Tdata, Tcalc> {
  using Base = NavStateBase<Tdata, Tcalc>;

 public:
  // using Ptr = typename Base::template shared_ptr<NavState>;
  using typename Base::SO3data;
  using typename Base::Vec3data;

  // b_omega_wb
  Vec3data omega_bb_ = Vec3data::Zero();

  // for Eigen has deep copy, we don't define default copy constructor and operator= and we don't need ~NavState(),
  // so move copy constructor and move operator= will also be default

  Vec3data TransToVwc(const Vec3data &tbc) {
    // vwc = dot(pwb + Rwb*tcb) = vwb + Rwb*wbb^tbc
    return this->vwb_ + this->Rwb_ * (SO3data::hat(omega_bb_) * tbc);
  }
  Vec3data TransToOmegacc(const SO3data &Rcb) {
    // for Rwc'=Rwc*wcc^=Rwb*wbb^*Rbc=Rwb*Rbc*(Rcb*wbb)^
    return Rcb * omega_bb_;
  }

  std::string FormatString(bool bfull = true) const override {
    return Base::FormatString(bfull) + mlog::FormatString(";w={}", omega_bb_.transpose());
  }

 protected:
  // Extrinsics related params
  // Tbc is from IMU frame to camera frame;Tce is from camera frame to encoder frame
  //  (the centre of 2 driving wheels, +x pointing to forward,+z pointing up)
  // cv::Mat mTbc, mTce;
};

using NavStateBased = NavStateBase<double>;
using NavStated = NavState<double>;

template <class Tdata, class Tcalc>
std::ostream &operator<<(std::ostream &out, const NavStateBase<Tdata, Tcalc> &ns) {
  out << ns.FormatString(false);
  return out;
}

}  // namespace VIEO_SLAM
