//
// Created by leavesnight on 2021/3/29.
//

#pragma once

#include <Eigen/StdVector>
#include <Eigen/Geometry>
#ifdef USE_SOPHUS_NEWEST
#include "sophus/so3.hpp"
#else
#include "so3ex_base.h"
#endif

namespace Sophus {
#ifdef USE_SOPHUS_NEWEST
template <class Derived>
using SO3exBase = SO3Base<Derived>;
#endif

template <class Scalar_, int Options = 0>
class SO3ex : public SO3<Scalar_, Options> {
 protected:
  using Base = SO3<Scalar_, Options>;
  using Base::unit_quaternion_;
  using Base::unit_quaternion_nonconst;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = Scalar_;
  using typename Base::QuaternionType;  // Eigen::Quaternion<Scalar, Options>
  using typename Base::Tangent;         // Eigen::Matrix<Scalar, 3, 1>
  using typename Base::Transformation;  // Eigen::Matrix<Scalar, 3, 3>
  // here Eigen::ScalarBinaryOpTraits<> in 3.3.7 and sophus (Wed Apr 21 18:12:08 2021 -0700) should only accept same
  // scalar or one is complex
  template <typename OtherDerived>
  using ReturnScalar = typename Base::template ReturnScalar<OtherDerived>;
  template <typename OtherDerived>
  using SO3exProduct = SO3ex<ReturnScalar<OtherDerived>>;
  template <typename PointDerived>
  using PointProduct = typename Base::template PointProduct<PointDerived>;
  using Base::hat;
  using Base::unit_quaternion;

  // SOPHUS_FUNC now is EIGEN_DEVICE_FUNC for CUDA usage: __host__ __device__ means cpu&&gpu both make this func
  SOPHUS_FUNC SO3ex() : Base() {}
  template <class OtherDerived>
  SOPHUS_FUNC SO3ex(SO3exBase<OtherDerived> const& other) : Base(other) {}
  SOPHUS_FUNC SO3ex(SO3ex const& other) : Base(other) {}
  // so3 will assert identity of R, but for convenience, we normalize all R here
  SOPHUS_FUNC SO3ex(Transformation const& R) {
    // Eigen::JacobiSVD<Transformation> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // this->unit_quaternion_ = QuaternionType(Transformation(svd.matrixU() * svd.matrixV().transpose()));
    this->unit_quaternion_ = QuaternionType(R);
    this->unit_quaternion_.normalize();
  }
  template <class D>
  SOPHUS_FUNC explicit SO3ex(Eigen::QuaternionBase<D> const& quat) : Base(quat) {
    // assert(quat.squaredNorm() > SMALL_EPS);
  }

  // safer vee
  SOPHUS_FUNC static Tangent vee(Transformation const& Omega) {
    assert(fabs(Omega(2, 1) + Omega(1, 2)) < SMALL_EPS);
    assert(fabs(Omega(0, 2) + Omega(2, 0)) < SMALL_EPS);
    assert(fabs(Omega(1, 0) + Omega(0, 1)) < SMALL_EPS);
    return Base::vee(Omega);
  }

  // use our creative auto eps based Talyor expansion of special sin/1-cos(theta) to solve raw sin/cos numerical problem
  // theta_2 can be assigned by user while default NAN is theta^2
  SOPHUS_FUNC static Scalar GetTaylorSinTheta(const Scalar& theta, int n_order_start = 1, int n_theta = 1,
                                              Scalar theta_2 = Scalar(NAN),
                                              const Scalar& eps = std::numeric_limits<Scalar>::epsilon());
  // no numerical problem, eps only for speed up
  SOPHUS_FUNC static SO3ex exp(Tangent const& omega, const Scalar& eps = SMALL_EPS) {
    Scalar theta = omega.norm();

    Scalar imag_factor;
    Scalar real_factor;
    if (theta < eps) {
      Scalar theta2 = theta * theta;
      // Scalar theta_po4 = theta2 * theta2;
      // Taylor expansion of sin(x/2)/x, the one more order term commented, where 0.5 + it is useless
      // imag_factor = Scalar(0.5) - theta2 / Scalar(48.);  // + theta_po4 / Scalar(3840.);
      // real_factor = Scalar(1.0) - theta2 / Scalar(8.);   // + theta_po4 / Scalar(384.);
      imag_factor = Scalar(0.5) * GetTaylorSinTheta(theta, 1, 0, Scalar(0.25) * theta2);
      real_factor = GetTaylorSinTheta(Scalar(0.5) * theta, 0, 0);
    } else {
      Scalar half_theta = Scalar(0.5) * theta;
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta / theta;
      real_factor = cos(half_theta);
    }

    return SO3ex(
        QuaternionType(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z()));
  }
  SOPHUS_FUNC Tangent log(const Scalar& eps = SMALL_EPS) const { return log(*this, eps); }
  // range[-pi,pi) is mainly designed for residual error(abs or pow 2), (-pi,pi) could ensure uniqueness, but when
  // theta = -pi this log() cannot ensure uniqueness, except we define sth. like
  // if abs(theta)=pi {suppose q=(w,xyz) then theta=-pi, ensure z>=0{(if z<0:xyz=-xyz), if (z==0) ensure y>=0{if (y ==
  // 0) ensure x>=0}}}, g2o's se3quat also has such problem, which could limit the interpolation op. of system's state
  // one way is to use Eigen's slerp(<=3.3.7 confirmed), which will limit the angle between 2 qs <= pi/2, suitable for
  // system's state's interpolation and won't require the uniqueness of q but if you use linear interpolation here, you
  // should ensure the input q's uniqueness problem)
  // Also we should notice interpolation op. on angular velocity(which should have no limit on its range, so usually
  // it's interpolated through simple linear one instead of angular linear one like slerp)
  SOPHUS_FUNC static Tangent log(const SO3ex& other, const Scalar& eps = SMALL_EPS) {
    Scalar n = other.unit_quaternion_.vec().norm();  // sin(theta/2)
    Scalar w = other.unit_quaternion_.w();           // cos(theta/2)
    Scalar squared_w = w * w;

    Scalar two_atan_nbyw_by_n;
    // Atan-based log thanks to
    //
    // C. Hertzberg et al.:
    // "Integrating Generic Sensor Fusion Algorithms with Sound State
    // Representation through Encapsulation of Manifolds"
    // Information Fusion, 2011

    // small variable approximation is used for speed but keep the max or reasonable accuracy
    // (so3.cpp here choose the max or double(1) + double(2^(-52)))
    if (n < eps) {
      // If quaternion is normalized and n=1, then w should be 1;
      // w=0 should never happen here!
      assert(fabs(w) > eps);

      two_atan_nbyw_by_n = Scalar(2.) / w - Scalar(2. / 3) * (n * n) / (w * squared_w);  // right Taylor 2./3
    } else {
      if (fabs(w) < eps) {    // notice atan(x) = pi/2 - atan(1/x)
        if (w > Scalar(0)) {  // notice for range[-pi,pi), atan(x) = pi/2 - atan(1/x) for x>0
          two_atan_nbyw_by_n = M_PI / n;
        } else                             // w=0 corresponds to theta = Pi or -Pi, here choose -Pi
        {                                  // notice for range[-pi,pi), atan(x) = -pi/2 - atan(1/x) for x<=0
          two_atan_nbyw_by_n = -M_PI / n;  // theta belongs to [-Pi,Pi)=>theta/2 in [-Pi/2,Pi/2)
        }
        Scalar n_pow2 = n * n;
        Scalar n_pow4 = n_pow2 * n_pow2;
        two_atan_nbyw_by_n -= Scalar(2.) * w / n_pow2 - Scalar(2. / 3) * (w * squared_w) / n_pow4;
      } else
        two_atan_nbyw_by_n = 2 * atan(n / w) / n;  // theta/sin(theta/2)
      /*
      // when w < 0 use atan2(-n, -w) just for abs(output theta) <= pi/2
      Scalar atan_nbyw = (w < Scalar(0)) ? Scalar(atan2(-n, -w)) : Scalar(atan2(n, w));
      two_atan_nbyw_by_n = Scalar(2) * atan_nbyw / n;*/
    }

    return two_atan_nbyw_by_n * other.unit_quaternion_.vec();
  }
  static SOPHUS_FUNC Tangent Log(const Transformation& R, const Scalar& eps = SMALL_EPS);
  // exponential map from vec3 to mat3x3 (Rodrigues formula)
  SOPHUS_FUNC static Transformation Exp(const Tangent& v, const Scalar& eps = SMALL_EPS);
  // here is inline, but defined in .cpp is ok for efficiency due to copy elision(default gcc -O2 uses it) when
  // return a temporary variable(NRVO/URVO)
  SOPHUS_FUNC static Transformation ExpQ(const Tangent& v, const Scalar& eps = SMALL_EPS) {
    return exp(v, eps).matrix();  // here is URVO
  }

  // Jl, left jacobian of SO(3), result EPS~=8.e-12
  SOPHUS_FUNC static Transformation JacobianL(const Tangent& w, const Scalar& eps = SMALL_EPS);
  // Jr, right jacobian of SO(3), Jr(x) = Jl(-x)
  SOPHUS_FUNC static Transformation JacobianR(const Tangent& w, const Scalar& eps = SMALL_EPS) {
    return JacobianL(-w, eps);
  }
  // Jl^(-1), result EPS~=dEPS * 2~=4.e-16
  SOPHUS_FUNC static Transformation JacobianLInv(const Tangent& w, const Scalar& eps = SMALL_EPS);
  // Jr^(-1)
  SOPHUS_FUNC static Transformation JacobianRInv(const Tangent& w, const Scalar& eps = SMALL_EPS) {
    return JacobianLInv(-w, eps);
  }
  // we ensure result EPS~=2.e-10 from this func. & small result same on double eps (~theta*2^(-52))
  // Jls, Jls(wt) = 2int_0to1_Jl(swt)sds
  SOPHUS_FUNC static Transformation JacobianLS(const Tangent& wt, const Scalar& eps = SMALL_EPS_LS);
  // Jls2, Jls2(wt) = 3int_0to1_Jl(swt)s^2ds, notice 3Jls(wt)-2Jls2(wt)=Jls_s2(wt), we just use this func. to check
  SOPHUS_FUNC static Transformation JacobianLS2(const Tangent& wt, const Scalar& eps = SMALL_EPS_LS_S2);
  // Jls_s2, Jls_s2(wt) = 3int_0to1_Jls(swt)s^2ds
  SOPHUS_FUNC static Transformation JacobianLS_S2(const Tangent& wt, const Scalar& eps = SMALL_EPS_LS_S2);
  // Jls_s2_s3, Jls_s2_s3(wt) = 4int_0to1_Jls_s2(swt)s^3ds
  SOPHUS_FUNC static Transformation JacobianLS_S2_S3(const Tangent& wt, const Scalar& eps = SMALL_EPS_LS_S2_S3);
  // Jl(phi+dphi)x ~= Jl(phi)x + Jl_A(phi,x)*dphi, result EPS~=8.e-11
  template <bool bx_is_dphi = false>
  SOPHUS_FUNC static Transformation JacobianL_A(const Tangent& wt, const Tangent& x, const Scalar& eps = SMALL_EPS_LS);
  // Jls(phi+dphi)x ~= Jls(phi)x + Jls_A(phi,x)*dphi, result EPS~=2.e-11
  template <bool bx_is_dphi = false>
  SOPHUS_FUNC static Transformation JacobianLS_A(const Tangent& wt, const Tangent& x,
                                                 const Scalar& eps = SMALL_EPS_LS_S2);
  // Jls2(phi+dphi)x ~= Jls2(phi)x + Jls2_A(phi,x)*dphi
  //  for 3Jls_A(phi)-2Jls2_A(phi)=Jls_s2_A(phi), we just use this func. to check, result EPS~=2.e-11
  template <bool bx_is_dphi = false>
  SOPHUS_FUNC static Transformation JacobianLS2_A(const Tangent& wt, const Tangent& x,
                                                  const Scalar& eps = SMALL_EPS_LS_S2_S3);
  // Jls_s2(phi+dphi)x ~= Jls_s2(phi)x + Jls_s2_A(phi,x)*dphi, result EPS~=4.e-11
  template <bool bx_is_dphi = false>
  SOPHUS_FUNC static Transformation JacobianLS_S2_A(const Tangent& wt, const Tangent& x,
                                                    const Scalar& eps = SMALL_EPS_LS_S2_S3);
  // Jls_s2_s3(phi+dphi)x ~= Jls_s2_s3(phi)x + Jls_s2_s3_A(phi,x)*dphi
  template <bool bx_is_dphi = false>
  SOPHUS_FUNC static Transformation JacobianLS_S2_S3_A(const Tangent& wt, const Tangent& x,
                                                       const Scalar& eps = SMALL_EPS_LS_S2_S3_A);

  // for no usage of quaternion, and please ensure det(R) > 0 by user, much slower than NormalizeRotationM,
  //  this func. has numerical problem with result EPS~=4.e-15
  SOPHUS_FUNC static inline Transformation NormalizeRotation(const Transformation& R) {
    Eigen::JacobiSVD<Transformation> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
  }
  // normalize to avoid numerical error accumulation
  SOPHUS_FUNC static inline QuaternionType NormalizeRotationQ(const QuaternionType& r) {
    QuaternionType _r(r);
    if (_r.w() < 0)  // is this necessary?
    {
      _r.coeffs() *= -1;
    }
    return _r.normalized();
  }
  SOPHUS_FUNC static inline Transformation NormalizeRotationM(const Transformation& R) {
    QuaternionType qr(R);
    return NormalizeRotationQ(qr).toRotationMatrix();
  }

  // keep result EPS~=1.e-10
  const static double SMALL_EPS;
  const static double SMALL_EPS_LS;
  const static double SMALL_EPS_LS_S2;
  const static double SMALL_EPS_LS_S2_S3;
  const static double SMALL_EPS_LS_S2_S3_A;
};

template <class Scalar, int Options>
const double SO3ex<Scalar, Options>::SMALL_EPS = 1.e-5;
template <class Scalar, int Options>
const double SO3ex<Scalar, Options>::SMALL_EPS_LS = 1.e-3;
template <class Scalar, int Options>
const double SO3ex<Scalar, Options>::SMALL_EPS_LS_S2 = 2.e-2;
template <class Scalar, int Options>
const double SO3ex<Scalar, Options>::SMALL_EPS_LS_S2_S3 = 6.e-2;
template <class Scalar, int Options>
const double SO3ex<Scalar, Options>::SMALL_EPS_LS_S2_S3_A = 1.e-1;

template <class Scalar, int Options>
Scalar SO3ex<Scalar, Options>::GetTaylorSinTheta(const Scalar& theta, int n, int n_theta, Scalar theta_2,
                                                 const Scalar& eps) {
  Scalar ret = Scalar(0.);
  // notice (std::)pow(0.,0) return 1
  Scalar theta_n = pow(theta, n_theta);
  if (std::isnan(theta_2)) theta_2 = theta * theta;
  // assert(n < std::numeric_limits<int>::max());
  Scalar n_order = Scalar(std::tgamma(n + 1));  // n!
  auto next_term = theta_n / n_order;           // start from 3!
  while (abs(next_term) > eps) {
    ret += next_term;
    // assert(n < std::numeric_limits<int>::max() - 1);
    ++n;
    next_term = -next_term * theta_2 / Scalar(n * (n + 1));
    ++n;
  }
  /*
  static int n_max_test = 0;
  if (n_max_test < n) {
    n_max_test = n;
    PRINT_INFO_MUTEX("check GetTaylorSinTheta n_max=" << n_max_test << std::endl);
  }*/
  return ret;
}
// when USE_EXPLOG_NOQ_MATH: here static Log/Exp will directly operate R2r/r2R, without using quaternion,
//  but Exp can be much slower than ExpQ if final SVD called!
//#define USE_EXPLOG_NOQ_MATH
template <class Scalar, int Options>
typename SO3ex<Scalar, Options>::Tangent SO3ex<Scalar, Options>::Log(const Transformation& R, const Scalar& eps) {
#ifndef USE_EXPLOG_NOQ_MATH
  return SO3ex<Scalar, Options>(R).log(eps);
#else
  const Scalar tr = R(0, 0) + R(1, 1) + R(2, 2);
  Tangent w;
  w << (R(2, 1) - R(1, 2)) / 2, (R(0, 2) - R(2, 0)) / 2, (R(1, 0) - R(0, 1)) / 2;
  const Scalar costheta = (tr - 1.0) * 0.5f;
  if (costheta > 1 || costheta < -1) return w;
  const Scalar theta = acos(costheta);
  const Scalar s = sin(theta);
  if (fabs(s) < eps) {
    w *= 1 + theta * theta / Scalar(6.);
    /*
    auto SinTheta3_2 = GetTaylorSinTheta(theta, 3, 2);
    w *= 1 + SinTheta3_2;*/
  } else {
    w *= theta / s;
  }
  return w;
#endif
}
template <class Scalar, int Options>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::Exp(const Tangent& w, const Scalar& eps) {
#ifndef USE_EXPLOG_NOQ_MATH
  return ExpQ(w, eps);
#else
  Transformation res = Transformation::Identity();
  Scalar theta = w.norm();
  Tangent k = w.normalized();  // k - unit direction vector of w
  Transformation K = hat(k);
  if (theta < eps) {
    // res += W + 0.5 * W * W;
    res += GetTaylorSinTheta(theta, 1, 0) * theta * K + GetTaylorSinTheta(theta, 2, 1) * theta * K * K;
  } else {
    res += sin(theta) * K + (1.0 - cos(theta)) * K * K;
  }
  // return NormalizeRotation(res);
  return NormalizeRotationM(res);
#endif
}

template <class Scalar, int Options>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianL(const Tangent& w, const Scalar& eps) {
  Transformation Jl = Transformation::Identity();
  // sqrt(sqauredNorm()) is slower than norm(); Omega / theta doesn't faster/more exact than K
  Scalar theta = w.norm();
  Tangent k = w.normalized();  // k - unit direction vector of w
  Transformation K = hat(k);
  if (theta < eps) {
    // ref Geometric integration on Euclidean group with application to articulated multibody systems(Jonghoon Park)
    //  the coeff is just 1/(j+1)!(w^)^j
    // omit 4th order eps & more for 1e-5 (accuracy:e-15), similar to omit >=1st order (Jl=I/R) for 1e-10
    // the one more order term is - theta2 / Scalar(120.)) * Omega2 < theta * 2^(-52), where theta + it is useless
    // Jl += (Scalar(0.5) - theta2 / Scalar(24.)) * Omega + (Scalar(1. / 6) - theta2 / Scalar(120.)) * Omega2;
    Jl += GetTaylorSinTheta(theta, 2, 0) * theta * K + GetTaylorSinTheta(theta, 3, 1) * theta * K * K;
  } else {
    Jl += (1 - cos(theta)) / theta * K + (1 - sin(theta) / theta) * K * K;
  }
  return Jl;
}
template <class Scalar, int Options>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLInv(const Tangent& w,
                                                                                     const Scalar& eps) {
  Transformation Jlinv = Transformation::Identity();
  Scalar theta = w.norm();
  Tangent k = w.normalized();  // k - unit direction vector of w
  Transformation K = hat(k);

  // very small angle
  if (theta < eps) {
    // limit(theta->0)((1-theta/(2*tan(theta/2)))/theta^2)~=(omit theta^5&&less)=1/12
    Scalar theta2 = theta * theta;
    // notice the coeff is just related to Bernoulli number(z/(e^z-1)=Sigma(n=0,inf){Bn*z^n/n!}):
    //  meaning it's (Bj/j!)*ad(w)^j, notice ad(w)=w^ and w^3=-w^,w^4=-w^2,w^5=w^,w^6=w^2... and
    //  Bj={0:1,1:-0.5,1/6,0,-1/30,0,1/42,0,-1/30,0...}; ref from (DOI:10.1109/TRO.2005.852253) or
    //  Geometric integration on Euclidean group with application to articulated multibody systems(Jonghoon Park)
    // + theta2 * theta2 * theta * Scalar(1. / 30240) (omit theta^5(*theta)&&less)
    Jlinv += -Scalar(0.5) * theta * K + (Scalar(1. / 12) * theta + theta2 * theta * Scalar(1. / 720)) * theta * K * K;
    /*
    auto SinTheta2_1 = GetTaylorSinTheta(theta, 2, 1);
    auto SinTheta3_1 = GetTaylorSinTheta(theta, 3, 1), SinTheta3_2 = SinTheta3_1 * theta;
    // auto SinTheta3_2_sq = SinTheta3_2 * SinTheta3_2; + SinTheta3_2_sq) * theta * K * K
    Jlinv += -Scalar(0.5) * theta * K +
             (Scalar(0.5) * SinTheta2_1 - SinTheta3_1) * (Scalar(1.) + SinTheta3_2) * theta * K * K;*/
  } else {
    // notice theta(1+cos(theta))/(2sin(theta))=theta * cos(half_theta) / (Scalar(2) * sin(half_theta)),
    //  and K * K=Omega * Omega / (theta * theta)
    Jlinv +=
        -Scalar(0.5) * theta * K + (Scalar(1.) - (Scalar(1.) + cos(theta)) * theta / (Scalar(2.) * sin(theta))) * K * K;
  }

  return Jlinv;
}
template <class Scalar, int Options>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLS(const Tangent& wt,
                                                                                   const Scalar& eps) {
  Transformation Jls = Transformation::Identity();
  Scalar theta = wt.norm();
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  if (theta < eps) {
    // the one more order term is theta4 / Scalar(20160.)) * Omega2 < theta * 2^(-52), where theta + it is useless under
    //  result EPS~=2.e-10 * 1.e-1
    // Jls += (Scalar(1. / 3) - theta2 / Scalar(60.) + theta4 / Scalar(2520.)) * Omega +
    //       (Scalar(1. / 12) - theta2 / Scalar(360.)) * Omega2;
    Jls += 2 * (GetTaylorSinTheta(theta, 3, 0) * theta * K + GetTaylorSinTheta(theta, 4, 1) * theta * K * K);
  } else {
    Scalar theta2 = theta * theta;
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    Jls += 2 * (1 - sinwtdivwt) / theta * K + (1 - 2 * one_coswt_divwt2) * K * K;
  }
  return Jls;
}
template <class Scalar, int Options>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLS2(const Tangent& wt,
                                                                                    const Scalar& eps) {
  // 3Jls(wt)-2Jls2(wt)=Jls_s2(wt)
  // return (3 * JacobianLS(wt, eps) - JacobianLS_S2(wt, eps)) / 2; // this is 1.5x slower than the following
  // just check here, only consider eps=1.e-5 approx
  Transformation Jls2 = Transformation::Identity();
  Scalar theta = wt.norm();
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  if (theta < eps) {
    // Jls2 = Jls2 + Scalar(3. / 8) * Omega + Scalar(1. / 10) * Omega2;
    auto SinTheta4_0 = GetTaylorSinTheta(theta, 4, 0);
    Jls2 += 3 * (GetTaylorSinTheta(theta, 3, 0) - SinTheta4_0) * theta * K +
            3 * (SinTheta4_0 * theta - GetTaylorSinTheta(theta, 5, 1)) * theta * K * K;
  } else {
    Scalar theta2 = theta * theta;
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    Jls2 += 3 * (Scalar(1. / 2) - sinwtdivwt + one_coswt_divwt2) / theta * K +
            (1 + 3 * (cos(theta) - sinwtdivwt) / theta2) * K * K;
  }
  return Jls2;
}
template <class Scalar, int Options>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLS_S2(const Tangent& wt,
                                                                                      const Scalar& eps) {
  Transformation Jls_s2 = Transformation::Identity();
  Scalar theta = wt.norm();
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  if (theta < eps) {
    // K with eps_scalar faster than Omega faster than Omega with eps_scalar/theta,10.1<11.1<11.6
    // if eps=1.e-5: Scalar(1. / 4) * Omega + Scalar(1. / 20) * Omega2
    Jls_s2 += 6 * (GetTaylorSinTheta(theta, 4, 0) * theta * K + GetTaylorSinTheta(theta, 5, 1) * theta * K * K);
  } else {
    Scalar theta2 = theta * theta;
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    Jls_s2 += 3 * (1 - 2 * one_coswt_divwt2) / theta * K + (1 - Scalar(6.) * (1 - sinwtdivwt) / theta2) * K * K;
  }
  return Jls_s2;
}
template <class Scalar, int Options>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLS_S2_S3(const Tangent& wt,
                                                                                         const Scalar& eps) {
  Transformation Jls_s2_s3 = Transformation::Identity();
  Scalar theta = wt.norm();
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  if (theta < eps) {
    // Jls_s2_s3 += Scalar(1. / 5) * Omega + Scalar(1. / 30) * Omega2;
    Jls_s2_s3 += 24 * ((GetTaylorSinTheta(theta, 5, 0)) * theta * K + (GetTaylorSinTheta(theta, 6, 1)) * theta * K * K);
  } else {
    Scalar theta2 = theta * theta;
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    Jls_s2_s3 += 4 * (1 - Scalar(6.) * (1 - sinwtdivwt) / theta2) / theta * K +
                 (1 - 12 * (1 - 2 * one_coswt_divwt2) / theta2) * K * K;
  }
  return Jls_s2_s3;
}
template <class Scalar, int Options>
template <bool bx_is_dphi>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianL_A(const Tangent& wt, const Tangent& x,
                                                                                    const Scalar& eps) {
  Transformation JlA = Transformation::Identity();
  Scalar theta = wt.norm();
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  Transformation I = Transformation::Identity();
  Transformation xkT = !bx_is_dphi ? Transformation(x * k.transpose()) : k.dot(x) * I;
  Transformation hat_x = !bx_is_dphi ? hat(-x) : hat(x);
  Transformation kxT_2xkTpkTxI = !bx_is_dphi ? Transformation(xkT.transpose() - 2 * xkT + k.dot(x) * I)
                                             : Transformation(k * x.transpose() - 2 * xkT + x * k.transpose());
  if (theta < eps) {
    // -x^ term's no theta part should be the same as Exp(wt/2)'s Exp(wt/2)*(-x)^*Jr(wt/2)/2's no theta part
    // 1stly choose 1.e-13 instead of 2^(-52) to avoid too many terms, but now we use auto Taylor and still use 2^(-52)
    // term < 1.e-13: ((-theta6 / Scalar(6720.)) * K + (theta5 / Scalar(1260.)) * K * K) * xkT
    // + (-theta6 / Scalar(40320.)) * hat_x + (theta5 / Scalar(5040.)) * kxT_2xkTpkTxI
    // JlA = ((-theta2 / Scalar(12.) + theta4 / Scalar(180.)) * K + (-theta3 / Scalar(60.)) * K * K) * xkT +
    //      (Scalar(0.5) - theta2 / Scalar(24.) + theta4 / Scalar(720.)) * hat_x +
    //      (theta / Scalar(6.) - theta3 / Scalar(120.)) * kxT_2xkTpkTxI;
    auto SinTheta3_0 = GetTaylorSinTheta(theta, 3, 0);
    auto SinTheta4_1 = GetTaylorSinTheta(theta, 4, 1);
    JlA = ((-SinTheta3_0 * theta + 2 * SinTheta4_1) * theta * K +
           (3 * GetTaylorSinTheta(theta, 5, 2) - SinTheta4_1 * theta) * theta * K * K) *
              xkT +
          GetTaylorSinTheta(theta, 2, 0) * hat_x + SinTheta3_0 * theta * kxT_2xkTpkTxI;
  } else {
    Scalar theta2 = theta * theta;
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    JlA = ((sinwtdivwt - 2 * one_coswt_divwt2) * K + (3 * sinwtdivwt - (2 + cos(theta))) / theta * K * K) * xkT +
          one_coswt_divwt2 * hat_x + (1 - sinwtdivwt) / theta * kxT_2xkTpkTxI;
  }
  return JlA;
}
template <class Scalar, int Options>
template <bool bx_is_dphi>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLS_A(const Tangent& wt,
                                                                                     const Tangent& x,
                                                                                     const Scalar& eps) {
  Transformation JlsA = Transformation::Identity();
  Scalar theta = wt.norm();
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  Transformation I = Transformation::Identity();
  Transformation xkT = !bx_is_dphi ? Transformation(x * k.transpose()) : k.dot(x) * I;
  Transformation hat_x = !bx_is_dphi ? hat(-x) : hat(x);
  Transformation kxT_2xkTpkTxI = !bx_is_dphi ? Transformation(xkT.transpose() - 2 * xkT + k.dot(x) * I)
                                             : Transformation(k * x.transpose() - 2 * xkT + x * k.transpose());
  if (theta < eps) {
    auto SinTheta3_0 = GetTaylorSinTheta(theta, 3, 0);
    auto SinTheta4_0 = GetTaylorSinTheta(theta, 4, 0);
    auto SinTheta5_1 = GetTaylorSinTheta(theta, 5, 1);
    // -x^ term's no theta part should be the same as Exp(wt/3)'s Exp(wt/3)*(-x)^*Jr(wt/3)/3's no theta part
    // JlsA = -theta2 / Scalar(30.) * K * xkT + (Scalar(1. / 3) - theta2 / Scalar(60.)) * hat_x +
    //       theta / Scalar(12.) * kxT_2xkTpkTxI;
    JlsA = ((3 * SinTheta5_1 - SinTheta4_0 * theta) * theta * K +
            (4 * GetTaylorSinTheta(theta, 6, 2) - SinTheta5_1 * theta) * theta * K * K) *
               xkT +
           SinTheta3_0 * hat_x + SinTheta4_0 * theta * kxT_2xkTpkTxI;
    JlsA *= 2;
  } else {
    Scalar theta2 = theta * theta;
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    JlsA = (2 / theta2 * (3 * sinwtdivwt - (2 + cos(theta))) * K +
            2 / theta * (-1 - sinwtdivwt + 4 * one_coswt_divwt2) * K * K) *
               xkT +
           2 * (1 - sinwtdivwt) / theta2 * hat_x + (1 - 2 * one_coswt_divwt2) / theta * kxT_2xkTpkTxI;
  }
  return JlsA;
}
template <class Scalar, int Options>
template <bool bx_is_dphi>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLS2_A(const Tangent& wt,
                                                                                      const Tangent& x,
                                                                                      const Scalar& eps) {
  // 3Jls_A(phi)-2Jls2_A(phi)=Jls_s2_A(phi)
  // return (3 * JacobianLS_A(wt, x, eps) - JacobianLS_S2_A(wt, x, eps)) / 2; // this is 1.5x slower than the following
  // just check here
  Transformation Jls2A = Transformation::Identity();
  Scalar theta = wt.norm(), theta2 = theta * theta;
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  Transformation I = Transformation::Identity();
  Transformation xkT = !bx_is_dphi ? Transformation(x * k.transpose()) : k.dot(x) * I;
  Transformation hat_x = !bx_is_dphi ? hat(-x) : hat(x);
  Transformation kxT_2xkTpkTxI = !bx_is_dphi ? Transformation(xkT.transpose() - 2 * xkT + k.dot(x) * I)
                                             : Transformation(k * x.transpose() - 2 * xkT + x * k.transpose());
  if (theta < eps) {
    auto SinTheta3_0 = GetTaylorSinTheta(theta, 3, 0);
    auto SinTheta4_0 = GetTaylorSinTheta(theta, 4, 0);
    auto SinTheta5_0 = GetTaylorSinTheta(theta, 5, 0);
    auto SinTheta6_1 = GetTaylorSinTheta(theta, 6, 1);
    // -x^ term's no theta part should be the same as Exp(wt*3/8)'s Exp(wt*3/8)*(-x)^*Jr(wt*3/8)*3/8's no theta part
    // Jls2A = -theta2 / Scalar(24.) * K * xkT + (Scalar(3. / 8) - theta2 / Scalar(48.)) * hat_x +
    //        theta / Scalar(10.) * kxT_2xkTpkTxI;
    Jls2A = ((4 * (SinTheta5_0 * theta - SinTheta6_1) - SinTheta4_0 * theta) * theta * K +
             (5 * (SinTheta6_1 * theta - GetTaylorSinTheta(theta, 7, 2)) - SinTheta5_0 * theta2) * theta * K * K) *
                xkT +
            (SinTheta3_0 - SinTheta4_0) * hat_x + (SinTheta4_0 - SinTheta5_0) * theta * kxT_2xkTpkTxI;
    Jls2A *= 3;
  } else {
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    Jls2A = (3 * (-2 / theta2 + one_coswt_divwt2 + 4 / theta2 * (sinwtdivwt - one_coswt_divwt2)) * K +
             (-2 - 3 * sinwtdivwt + 15 / theta2 * (sinwtdivwt - cos(theta))) / theta * K * K) *
                xkT +
            3 * (Scalar(1. / 2) - sinwtdivwt + one_coswt_divwt2) / theta2 * hat_x +
            (1 + 3 * (cos(theta) - sinwtdivwt) / theta2) / theta * kxT_2xkTpkTxI;
  }
  return Jls2A;
}
template <class Scalar, int Options>
template <bool bx_is_dphi>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLS_S2_A(const Tangent& wt,
                                                                                        const Tangent& x,
                                                                                        const Scalar& eps) {
  Transformation Jls_s2A = Transformation::Identity();
  Scalar theta = wt.norm();
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  Transformation I = Transformation::Identity();
  Transformation xkT = !bx_is_dphi ? Transformation(x * k.transpose()) : k.dot(x) * I;
  Transformation hat_x = !bx_is_dphi ? hat(-x) : hat(x);
  Transformation kxT_2xkTpkTxI = !bx_is_dphi ? Transformation(xkT.transpose() - 2 * xkT + k.dot(x) * I)
                                             : Transformation(k * x.transpose() - 2 * xkT + x * k.transpose());
  if (theta < eps) {
    auto SinTheta3_0 = GetTaylorSinTheta(theta, 3, 0);
    auto SinTheta4_0 = GetTaylorSinTheta(theta, 4, 0);
    auto SinTheta5_0 = GetTaylorSinTheta(theta, 5, 0);
    auto SinTheta6_1 = GetTaylorSinTheta(theta, 6, 1);
    // -x^ term's no theta part should be the same as Exp(wt/4)'s Exp(wt/4)*(-x)^*Jr(wt/4)/4's no theta part
    // Jls_s2A = -theta2 / Scalar(60.) * K * xkT + (Scalar(1. / 4) - theta2 / Scalar(120.)) * hat_x +
    //          theta / Scalar(20.) * kxT_2xkTpkTxI;
    Jls_s2A = ((4 * SinTheta6_1 - SinTheta5_0 * theta) * theta * K +
               (5 * GetTaylorSinTheta(theta, 7, 2) - SinTheta6_1 * theta) * theta * K * K) *
                  xkT +
              SinTheta4_0 * hat_x + SinTheta5_0 * theta * kxT_2xkTpkTxI;
    Jls_s2A *= 6;
  } else {
    Scalar theta2 = theta * theta;
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    Jls_s2A = (6 / theta2 * (-1 - sinwtdivwt + 4 * one_coswt_divwt2) * K +
               2 / theta * (-1 + (12 + 3 * cos(theta)) / theta2 - 15 / theta2 * sinwtdivwt) * K * K) *
                  xkT +
              3 * (1 - 2 * one_coswt_divwt2) / theta2 * hat_x +
              (1 - Scalar(6.) * (1 - sinwtdivwt) / theta2) / theta * kxT_2xkTpkTxI;
  }
  return Jls_s2A;
}
template <class Scalar, int Options>
template <bool bx_is_dphi>
typename SO3ex<Scalar, Options>::Transformation SO3ex<Scalar, Options>::JacobianLS_S2_S3_A(const Tangent& wt,
                                                                                           const Tangent& x,
                                                                                           const Scalar& eps) {
  Transformation Jls_s2_s3A = Transformation::Identity();
  Scalar theta = wt.norm();
  Tangent k = wt.normalized();  // k - unit direction vector of wt
  Transformation K = hat(k);
  Transformation I = Transformation::Identity();
  Transformation xkT = !bx_is_dphi ? Transformation(x * k.transpose()) : k.dot(x) * I;
  Transformation hat_x = !bx_is_dphi ? hat(-x) : hat(x);
  Transformation kxT_2xkTpkTxI = !bx_is_dphi ? Transformation(xkT.transpose() - 2 * xkT + k.dot(x) * I)
                                             : Transformation(k * x.transpose() - 2 * xkT + x * k.transpose());
  if (theta < eps) {
    auto SinTheta3_0 = GetTaylorSinTheta(theta, 3, 0);
    auto SinTheta4_0 = GetTaylorSinTheta(theta, 4, 0);
    auto SinTheta5_0 = GetTaylorSinTheta(theta, 5, 0);
    auto SinTheta6_0 = GetTaylorSinTheta(theta, 6, 0);
    auto SinTheta7_1 = GetTaylorSinTheta(theta, 7, 1);
    // notice 210.=7!/4!=7x6x5, 30.=6x5
    // Jls_s2_s3A = -theta2 / Scalar(105.) * K * xkT + (Scalar(1. / 5) - theta2 / Scalar(210.)) * hat_x +
    //             theta / Scalar(30.) * kxT_2xkTpkTxI;
    Jls_s2_s3A = ((5 * SinTheta7_1 - SinTheta6_0 * theta) * theta * K +
                  (6 * GetTaylorSinTheta(theta, 8, 2) - SinTheta7_1 * theta) * theta * K * K) *
                     xkT +
                 SinTheta5_0 * hat_x + SinTheta6_0 * theta * kxT_2xkTpkTxI;
    Jls_s2_s3A *= 24;
  } else {
    Scalar theta2 = theta * theta;
    Scalar sinwtdivwt = sin(theta) / theta;
    Scalar one_coswt_divwt2 = (1 - cos(theta)) / theta2;
    Jls_s2_s3A = (8 / theta2 * (-1 + (12 + 3 * cos(theta)) / theta2 - 15 / theta2 * sinwtdivwt) * K +
                  2 / theta * (-1 + 12 / theta2 * (2 + sinwtdivwt - 6 * one_coswt_divwt2)) * K * K) *
                     xkT +
                 4 * (1 - Scalar(6.) * (1 - sinwtdivwt) / theta2) / theta2 * hat_x +
                 (1 - 12 * (1 - 2 * one_coswt_divwt2) / theta2) / theta * kxT_2xkTpkTxI;
  }
  return Jls_s2_s3A;
}

typedef SO3ex<double> SO3exd;
typedef SO3ex<float> SO3exf;

}  // namespace Sophus
