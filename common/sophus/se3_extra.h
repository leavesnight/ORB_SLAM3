//
// Created by leavesnight on 2021/3/29.
//

#pragma once

#include <Eigen/StdVector>
#include <Eigen/Geometry>
#ifdef USE_SOPHUS_NEWEST
#include "sophus/se3.hpp"
#include "so3_extra.h"
#else
#include "se3ex_base.h"
#endif

namespace Sophus {
#ifdef USE_SOPHUS_NEWEST
template <class Scalar_, int Options = 0>
using SE3ex = SE3<Scalar_, Options>;
#else
template <class Scalar_, int Options>
class SE3ex : public SE3exBase<SE3ex<Scalar_, Options> > {
  const static double SMALL_EPS;

 protected:
  using Base = SE3exBase<SE3ex<Scalar_, Options> >;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = Scalar_;
  using Base::DoF;
  using Base::num_parameters;
  using typename Base::SO3exType;
  using typename Base::Tangent;  // first rho then phi, different from save order!
  using typename Base::Transformation;
  using typename Base::TranslationType;

 protected:
  SO3exType so3_ex_;
  TranslationType t_ex_;

 public:
  // SOPHUS_FUNC now is EIGEN_DEVICE_FUNC for CUDA usage: __host__ __device__ means cpu&&gpu both make this func
  SOPHUS_FUNC SE3ex() : t_ex_(TranslationType::Zero()) {
    static_assert(std::is_standard_layout<SE3ex>::value, "Assume standard layout for the use of offsetof check below.");
    static_assert(offsetof(SE3ex, so3_ex_) + sizeof(Scalar) * SO3exType::num_parameters == offsetof(SE3ex, t_ex_),
                  "This class assumes packed storage and hence will only work "
                  "correctly depending on the compiler (options) - in "
                  "particular when using [this->data(), this-data() + "
                  "num_parameters] to access the raw data in a contiguous fashion.");
  }
  template <class OtherDerived>
  SOPHUS_FUNC SE3ex(SE3exBase<OtherDerived> const& other) : so3_ex_(other.so3()), t_ex_(other.translation()) {
    static_assert(std::is_same<typename OtherDerived::Scalar, Scalar>::value, "must be same Scalar type");
  }
  // if we ensure all SE3ex changing func will ensure the unit property on its so3, normalize can be omitted
  SOPHUS_FUNC SE3ex(SE3ex const& other) = default;
  template <class OtherDerived, class D>
  SOPHUS_FUNC SE3ex(SO3exBase<OtherDerived> const& so3, Eigen::MatrixBase<D> const& translation)
      : so3_ex_(so3), t_ex_(translation) {
    static_assert(std::is_same<typename OtherDerived::Scalar, Scalar>::value, "must be same Scalar type");
    static_assert(std::is_same<typename D::Scalar, Scalar>::value, "must be same Scalar type");
  }
  // so3 will assert identity of R, but for convenience, we nomralize all R in so3ex
  SOPHUS_FUNC SE3ex(typename SO3exType::Transformation const& R, TranslationType const& t) : so3_ex_(R), t_ex_(t) {}
  template <class D>
  SOPHUS_FUNC SE3ex(Eigen::Quaternion<D> const& q, TranslationType const& t) : so3_ex_(q), t_ex_(t) {}
  SOPHUS_FUNC explicit SE3ex(Matrix4<Scalar> const& T)
      : so3_ex_(T.template topLeftCorner<3, 3>()), t_ex_(T.template block<3, 1>(0, 3)) {
    MLOG_ENSURE((T.row(3) - Matrix<Scalar, 1, 4>(Scalar(0), Scalar(0), Scalar(0), Scalar(1))).squaredNorm() <
                    Constants<Scalar>::epsilon(),
                "Last row is not (0,0,0,1), but ({}).", T.row(3));
  }

  SOPHUS_FUNC SO3exType& so3() { return so3_ex_; }
  SOPHUS_FUNC SO3exType const& so3() const { return so3_ex_; }
  SOPHUS_FUNC TranslationType& translation() { return t_ex_; }
  SOPHUS_FUNC TranslationType const& translation() const { return t_ex_; }

  SOPHUS_FUNC static SE3ex exp(Tangent const& a) {
    using std::cos;
    using std::sin;
    typename SO3exType::Tangent const omega = a.template tail<3>();
    SO3exType const so3 = SO3exType::exp(omega);
    // V=JL(phi)=JR(-phi)
    typename SO3exType::Transformation const V = SO3exType::JacobianL(omega);
    return SE3ex<Scalar>(so3, V * a.template head<3>());
  }

 private:
  // private some Base func. for safety and prepared to use
  SOPHUS_FUNC Scalar* data() { return this->so3_ex_.data(); }
  SOPHUS_FUNC Scalar const* data() const { return this->so3_ex_.data(); }
  SOPHUS_FUNC static Matrix<Scalar, num_parameters, DoF> Dx_exp_x(Tangent const& upsilon_omega) {
    assert(0);
    Matrix<Scalar, num_parameters, DoF> J;
    return J;
  }
  /// Returns derivative of exp(x).matrix() wrt. ``x_i at x=0``.
  SOPHUS_FUNC static Transformation Dxi_exp_x_matrix_at_0(int i) { return generator(i); }
  template <class S = Scalar>
  SOPHUS_FUNC static enable_if_t<std::is_floating_point<S>::value, SE3ex> fitToSE3(Matrix4<Scalar> const& T) {
    return SE3ex(SO3ex<Scalar>::fitToSO3(T.template block<3, 3>(0, 0)), T.template block<3, 1>(0, 3));
  }
  SOPHUS_FUNC static Transformation generator(int i) {
    MLOG_ENSURE(i >= 0 && i <= 5, "i should be in range [0,5].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }
  SOPHUS_FUNC static Transformation hat(Tangent const& a) {
    Transformation Omega;
    Omega.setZero();
    Omega.template topLeftCorner<3, 3>() = SO3ex<Scalar>::hat(a.template tail<3>());
    Omega.col(3).template head<3>() = a.template head<3>();
    return Omega;
  }
  SOPHUS_FUNC static Tangent lieBracket(Tangent const& a, Tangent const& b) {
    Vector3<Scalar> const upsilon1 = a.template head<3>();
    Vector3<Scalar> const upsilon2 = b.template head<3>();
    Vector3<Scalar> const omega1 = a.template tail<3>();
    Vector3<Scalar> const omega2 = b.template tail<3>();
    Tangent res;
    // for so3:(a^b)^=a^b^-b^a^=[a,b]
    res.template head<3>() = omega1.cross(upsilon2) + upsilon1.cross(omega2);
    res.template tail<3>() = omega1.cross(omega2);
    return res;
  }
  static SOPHUS_FUNC SE3ex rotX(Scalar const& x) {
    return SE3ex(SO3ex<Scalar>::rotX(x), Sophus::Vector3<Scalar>::Zero());
  }
  static SOPHUS_FUNC SE3ex rotY(Scalar const& y) {
    return SE3ex(SO3ex<Scalar>::rotY(y), Sophus::Vector3<Scalar>::Zero());
  }
  static SOPHUS_FUNC SE3ex rotZ(Scalar const& z) {
    return SE3ex(SO3ex<Scalar>::rotZ(z), Sophus::Vector3<Scalar>::Zero());
  }
  template <class UniformRandomBitGenerator>
  static SE3ex sampleUniform(UniformRandomBitGenerator& generator) {
    std::uniform_real_distribution<Scalar> uniform(Scalar(-1), Scalar(1));
    return SE3ex(SO3ex<Scalar>::sampleUniform(generator),
                 Vector3<Scalar>(uniform(generator), uniform(generator), uniform(generator)));
  }
  template <class T0, class T1, class T2>
  static SOPHUS_FUNC SE3ex trans(T0 const& x, T1 const& y, T2 const& z) {
    return SE3ex(SO3ex<Scalar>(), Vector3<Scalar>(x, y, z));
  }
  static SOPHUS_FUNC SE3ex trans(Vector3<Scalar> const& xyz) { return SE3ex(SO3ex<Scalar>(), xyz); }
  static SOPHUS_FUNC SE3ex transX(Scalar const& x) { return SE3ex::trans(x, Scalar(0), Scalar(0)); }
  static SOPHUS_FUNC SE3ex transY(Scalar const& y) { return SE3ex::trans(Scalar(0), y, Scalar(0)); }
  static SOPHUS_FUNC SE3ex transZ(Scalar const& z) { return SE3ex::trans(Scalar(0), Scalar(0), z); }
  SOPHUS_FUNC static Tangent vee(Transformation const& Omega) {
    Tangent upsilon_omega;
    upsilon_omega.template head<3>() = Omega.col(3).template head<3>();
    upsilon_omega.template tail<3>() = SO3ex<Scalar>::vee(Omega.template topLeftCorner<3, 3>());
    return upsilon_omega;
  }
};

template <class Scalar, int Options>
const double SE3ex<Scalar, Options>::SMALL_EPS = 1e-5;
#endif

typedef SE3ex<double> SE3exd;
typedef SE3ex<float> SE3exf;
template <class Scalar_, int Options = 0>
using SE3 = SE3ex<Scalar_, Options>;
using SE3d = SE3exd;
using SE3f = SE3exf;

}  // namespace Sophus
