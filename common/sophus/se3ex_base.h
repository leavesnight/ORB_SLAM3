//
// Created by leavesnight on 7/6/23.
//

#pragma once

#include "so3_extra.h"
#include "common/mlog/log.h"

namespace Sophus {
template <class Scalar_, int Options = 0>
class SE3ex;
}

namespace Eigen {
namespace internal {
template <class Scalar_, int Options>
struct traits<Sophus::SE3ex<Scalar_, Options>> {
  using Scalar = Scalar_;
  using TranslationType = Sophus::Vector3<Scalar, Options>;
  using SO3exType = Sophus::SO3ex<Scalar, Options>;
};
template <class Scalar_, int Options>
struct traits<Map<Sophus::SE3ex<Scalar_>, Options>> : traits<Sophus::SE3ex<Scalar_, Options>> {
  using Scalar = Scalar_;
  using TranslationType = Map<Sophus::Vector3<Scalar>, Options>;
  using SO3exType = Map<Sophus::SO3ex<Scalar>, Options>;
};
template <class Scalar_, int Options>
struct traits<Map<Sophus::SE3ex<Scalar_> const, Options>> : traits<Sophus::SE3ex<Scalar_, Options> const> {
  using Scalar = Scalar_;
  using TranslationType = Map<Sophus::Vector3<Scalar> const, Options>;
  using SO3exType = Map<Sophus::SO3ex<Scalar> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {
template <class Derived>
class SE3exBase {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using SO3exType = typename Eigen::internal::traits<Derived>::SO3exType;
  using QuaternionType = typename SO3exType::QuaternionType;
  using TranslationType = typename Eigen::internal::traits<Derived>::TranslationType;

  static int constexpr DoF = 6;
  static int constexpr num_parameters = 7;
  static int constexpr N = 4;
  using Transformation = Matrix<Scalar, N, N>;
  using Tangent = Vector<Scalar, DoF>;
  template <typename OtherDerived>
  using ReturnScalar = typename Eigen::ScalarBinaryOpTraits<Scalar, typename OtherDerived::Scalar>::ReturnType;
  template <typename OtherDerived>
  using SE3exProduct = SE3ex<ReturnScalar<OtherDerived>>;
  template <typename PointDerived>
  using PointProduct = Vector3<ReturnScalar<PointDerived>>;
  template <typename HPointDerived>
  using HomogeneousPointProduct = Vector4<ReturnScalar<HPointDerived>>;

  SOPHUS_FUNC SO3exType& so3() { return static_cast<Derived*>(this)->so3(); }
  SOPHUS_FUNC SO3exType const& so3() const { return static_cast<const Derived*>(this)->so3(); }
  SOPHUS_FUNC QuaternionType const& unit_quaternion() const { return this->so3().unit_quaternion(); }
  SOPHUS_FUNC TranslationType& translation() { return static_cast<Derived*>(this)->translation(); }
  SOPHUS_FUNC TranslationType const& translation() const { return static_cast<Derived const*>(this)->translation(); }
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quat) { so3().setQuaternion(quat); }
  SOPHUS_FUNC void setRotationMatrix(Matrix3<Scalar> const& R) {
    MLOG_ENSURE(isOrthogonal(R), "R is not orthogonal:\n {}", R);
    MLOG_ENSURE(R.determinant() > Scalar(0), "det(R) is not positive: {}", R.determinant());
    so3().setQuaternion(Eigen::Quaternion<Scalar>(R));
  }

  template <class NewScalarType>
  SOPHUS_FUNC SE3ex<NewScalarType> cast() const {
    return SE3ex<NewScalarType>(so3().template cast<NewScalarType>(), translation().template cast<NewScalarType>());
  }
  SOPHUS_FUNC SE3ex<Scalar> inverse() const {
    SO3exType invR = so3().inverse();
    return SE3ex<Scalar>(invR, invR * (translation() * Scalar(-1)));
  }
  // first rho then phi, different from save order!
  SOPHUS_FUNC Tangent log() const {
    // For the derivation of the logarithm of SE(3), see
    // J. Gallier, D. Xu, "Computing exponentials of skew symmetric matrices
    // and logarithms of orthogonal matrices", IJRA 2002.
    // https:///pdfs.semanticscholar.org/cfe3/e4b39de63c8cabd89bf3feff7f5449fc981d.pdf
    // (Sec. 6., pp. 8)
    using std::abs;
    using std::cos;
    using std::sin;
    Tangent upsilon_omega;
    typename SO3exType::Tangent omega_and_theta = so3().log();
    upsilon_omega.template tail<3>() = omega_and_theta;
    // V_inv=Jlinv(phi)=Jrinv(-phi)
    typename SO3exType::Transformation V_inv = SO3exType::JacobianLInv(omega_and_theta);
    upsilon_omega.template head<3>() = V_inv * translation();
    return upsilon_omega;
  }
  SOPHUS_FUNC void normalize() { so3().normalize(); }
  SOPHUS_FUNC Transformation matrix() const {
    Transformation homogenious_matrix;
    homogenious_matrix.template topLeftCorner<3, 4>() = matrix3x4();
    homogenious_matrix.row(3) = Matrix<Scalar, 1, 4>(Scalar(0), Scalar(0), Scalar(0), Scalar(1));
    return homogenious_matrix;
  }
  SOPHUS_FUNC Matrix<Scalar, 3, 4> matrix3x4() const {
    Matrix<Scalar, 3, 4> matrix;
    matrix.template topLeftCorner<3, 3>() = rotationMatrix();
    matrix.col(3) = translation();
    return matrix;
  }
  SOPHUS_FUNC Matrix3<Scalar> rotationMatrix() const { return so3().matrix(); }
  template <class OtherDerived>
  SOPHUS_FUNC SE3exBase<Derived>& operator=(SE3exBase<OtherDerived> const& other) {
    so3() = other.so3();
    translation() = other.translation();
    return *this;
  }
  template <typename OtherDerived>
  SOPHUS_FUNC SE3exProduct<OtherDerived> operator*(SE3exBase<OtherDerived> const& other) const {
    return SE3exProduct<OtherDerived>(so3() * other.so3(), translation() + so3() * other.translation());
  }
  template <typename PointDerived, typename = typename std::enable_if<IsFixedSizeVector<PointDerived, 3>::value>::type>
  SOPHUS_FUNC PointProduct<PointDerived> operator*(Eigen::MatrixBase<PointDerived> const& p) const {
    return so3() * p + translation();
  }
  template <typename HPointDerived,
            typename = typename std::enable_if<IsFixedSizeVector<HPointDerived, 4>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<HPointDerived> operator*(Eigen::MatrixBase<HPointDerived> const& p) const {
    const PointProduct<HPointDerived> tp = so3() * p.template head<3>() + p(3) * translation();
    return HomogeneousPointProduct<HPointDerived>(tp(0), tp(1), tp(2), p(3));
  }
  template <typename OtherDerived,
            typename = typename std::enable_if<std::is_same<Scalar, ReturnScalar<OtherDerived>>::value>::type>
  SOPHUS_FUNC SE3exBase<Derived>& operator*=(SE3exBase<OtherDerived> const& other) {
    *static_cast<Derived*>(this) = *this * other;
    return *this;
  }

 private:
  using Adjoint = Matrix<Scalar, DoF, DoF>;
  SOPHUS_FUNC Adjoint Adj() const {
    Matrix3<Scalar> const R = so3().matrix();
    Adjoint res;
    res.block(0, 0, 3, 3) = R;
    res.block(3, 3, 3, 3) = R;
    res.block(0, 3, 3, 3) = SO3exType::hat(translation()) * R;
    res.block(3, 0, 3, 3) = Matrix3<Scalar>::Zero(3, 3);
    return res;
  }
  Scalar angleX() const { return so3().angleX(); }
  Scalar angleY() const { return so3().angleY(); }
  Scalar angleZ() const { return so3().angleZ(); }
  /// Returns derivative of  this * exp(x)  wrt x at x=0.
  SOPHUS_FUNC Matrix<Scalar, num_parameters, DoF> Dx_this_mul_exp_x_at_0() const {
    assert(0);
    Matrix<Scalar, num_parameters, DoF> J;
    return J;
  }
  // save order: phi then rho
  SOPHUS_FUNC Vector<Scalar, num_parameters> params() const {
    Vector<Scalar, num_parameters> p;
    p << so3().params(), translation();
    return p;
  }
  using Line = ParametrizedLine3<Scalar>;
  /// Group action on lines. This function rotates and translates a parametrized line
  /// ``l(t) = o + t * d`` by the SE(3) element:
  /// Origin is transformed using SE(3) action Direction is transformed using rotation part
  SOPHUS_FUNC Line operator*(Line const& l) const { return Line((*this) * l.origin(), so3() * l.direction()); }
};
}  // namespace Sophus

namespace Eigen {
/// Specialization of Eigen::Map for ``SE3``; derived from SE3Base.
/// Allows us to wrap SE3 objects around POD array.
template <class Scalar_, int Options>
class Map<Sophus::SE3ex<Scalar_>, Options> : public Sophus::SE3exBase<Map<Sophus::SE3ex<Scalar_>, Options>> {
 public:
  using Base = Sophus::SE3exBase<Map<Sophus::SE3ex<Scalar_>, Options>>;
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using typename Base::SO3exType;
  using typename Base::TranslationType;
  using SO3exMember = Sophus::SO3ex<Scalar>;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs) : so3_(coeffs), translation_(coeffs + SO3exMember::num_parameters) {}

  /// Mutator of SO3ex
  SOPHUS_FUNC SO3exType& so3() { return so3_; }
  /// Accessor of SO3ex
  SOPHUS_FUNC SO3exType const& so3() const { return so3_; }
  /// Mutator of translation vector
  SOPHUS_FUNC TranslationType& translation() { return translation_; }
  /// Accessor of translation vector
  SOPHUS_FUNC TranslationType const& translation() const { return translation_; }

 protected:
  SO3exType so3_;
  TranslationType translation_;
};

/// Specialization of Eigen::Map for ``SE3 const``; derived from SE3Base.
/// Allows us to wrap SE3 objects around POD array.
template <class Scalar_, int Options>
class Map<Sophus::SE3ex<Scalar_> const, Options>
    : public Sophus::SE3exBase<Map<Sophus::SE3ex<Scalar_> const, Options>> {
 public:
  using Base = Sophus::SE3exBase<Map<Sophus::SE3ex<Scalar_> const, Options>>;
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using typename Base::SO3exType;
  using typename Base::TranslationType;
  using SO3exMember = Sophus::SO3ex<Scalar>;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar const* coeffs) : so3_(coeffs), translation_(coeffs + SO3exMember::num_parameters) {}

  /// Accessor of SO3ex
  SOPHUS_FUNC SO3exType const& so3() const { return so3_; }
  /// Accessor of translation vector
  SOPHUS_FUNC TranslationType const& translation() const { return translation_; }

 protected:
  SO3exType const so3_;
  TranslationType const translation_;
};
}  // namespace Eigen
