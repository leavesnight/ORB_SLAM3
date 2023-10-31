//
// Created by leavesnight on 7/6/23.
//

#pragma once

#include "types_sophus.h"
#include "common/mlog/log.h"

namespace Sophus {
template <class Scalar_, int Options = 0>
class SO3;
using SO3d = SO3<double>;
using SO3f = SO3<float>;

/// Takes in arbitrary square matrix (2x2 or larger) and returns closest
/// orthogonal matrix with positive determinant.
template <class D>
SOPHUS_FUNC enable_if_t<std::is_floating_point<typename D::Scalar>::value,
                        Matrix<typename D::Scalar, D::RowsAtCompileTime, D::RowsAtCompileTime>>
makeRotationMatrix(Eigen::MatrixBase<D> const& R) {
  using Scalar = typename D::Scalar;
  static int const N = D::RowsAtCompileTime;
  static int const M = D::ColsAtCompileTime;
  static_assert(N == M, "must be a square matrix");
  static_assert(N >= 2, "must have compile time dimension >= 2");

  Eigen::JacobiSVD<Matrix<Scalar, N, N>> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  // Determine determinant of orthogonal matrix U*V'.
  Scalar d = (svd.matrixU() * svd.matrixV().transpose()).determinant();
  // Starting from the identity matrix D, set the last entry to d (d=+1/-1),  so that det(U*D*V') = 1.
  Matrix<Scalar, N, N> Diag = Matrix<Scalar, N, N>::Identity();
  Diag(N - 1, N - 1) = d;
  return svd.matrixU() * Diag * svd.matrixV().transpose();
}
}  // namespace Sophus

namespace Eigen {
namespace internal {
// specialization of traits struct to get type even it's c++ internal type(int/double/...)
template <class Scalar_, int Options>
struct traits<Sophus::SO3<Scalar_, Options>> {
  using Scalar = Scalar_;
  using QuaternionType = Eigen::Quaternion<Scalar, Options>;
};
template <class Scalar_, int Options>
struct traits<Map<Sophus::SO3<Scalar_>, Options>> : traits<Sophus::SO3<Scalar_, Options>> {
  using Scalar = Scalar_;
  using QuaternionType = Map<Eigen::Quaternion<Scalar>, Options>;
};
template <class Scalar_, int Options>
struct traits<Map<Sophus::SO3<Scalar_> const, Options>> : traits<Sophus::SO3<Scalar_, Options> const> {
  using Scalar = Scalar_;
  using QuaternionType = Map<Eigen::Quaternion<Scalar> const, Options>;
};
}  // namespace internal
}  // namespace Eigen

namespace Sophus {
template <class Derived>
class SO3exBase {
 public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using QuaternionType = typename Eigen::internal::traits<Derived>::QuaternionType;

  static int constexpr DoF = 3;
  static int constexpr num_parameters = 4;
  // SO3 group is 3x3 matrices
  static int constexpr N = 3;
  using Transformation = Matrix<Scalar, N, N>;
  using Tangent = Vector<Scalar, DoF>;
  template <typename OtherDerived>
  using ReturnScalar = typename Eigen::ScalarBinaryOpTraits<Scalar, typename OtherDerived::Scalar>::ReturnType;
  template <typename OtherDerived>
  using SO3Product = SO3<ReturnScalar<OtherDerived>>;
  template <typename PointDerived>
  using PointProduct = Vector3<ReturnScalar<PointDerived>>;
  template <typename HPointDerived>
  using HomogeneousPointProduct = Vector4<ReturnScalar<HPointDerived>>;

  SOPHUS_FUNC QuaternionType const& unit_quaternion() const {
    return static_cast<Derived const*>(this)->unit_quaternion();
  }
  SOPHUS_FUNC void setQuaternion(Eigen::Quaternion<Scalar> const& quaternion) {
    unit_quaternion_nonconst() = quaternion;
    normalize();
  }

  template <class NewScalarType>
  SOPHUS_FUNC SO3<NewScalarType> cast() const {
    return SO3<NewScalarType>(unit_quaternion().template cast<NewScalarType>());
  }
  SOPHUS_FUNC SO3<Scalar> inverse() const { return SO3<Scalar>(unit_quaternion().conjugate()); }
  SOPHUS_FUNC Tangent log() const {
    using std::abs;
    using std::atan2;
    using std::sqrt;
    Scalar squared_n = unit_quaternion().vec().squaredNorm();
    Scalar w = unit_quaternion().w();

    Scalar two_atan_nbyw_by_n;

    /// Atan-based log thanks to
    ///
    /// C. Hertzberg et al.:
    /// "Integrating Generic Sensor Fusion Algorithms with Sound State
    /// Representation through Encapsulation of Manifolds"
    /// Information Fusion, 2011

    if (squared_n < Constants<Scalar>::epsilon() * Constants<Scalar>::epsilon()) {
      // If quaternion is normalized and n=0, then w should be 1;
      // w=0 should never happen here!
      MLOG_ENSURE(abs(w) >= Constants<Scalar>::epsilon(), "Quaternion ({}) should be normalized!",
                  unit_quaternion().coeffs().transpose());
      Scalar squared_w = w * w;
      two_atan_nbyw_by_n = Scalar(2) / w - Scalar(2.0 / 3.0) * (squared_n) / (w * squared_w);
    } else {
      Scalar n = sqrt(squared_n);

      // when w < 0 use atan2(-n, -w) just for abs(output theta) <= pi/2
      Scalar atan_nbyw = (w < Scalar(0)) ? Scalar(atan2(-n, -w)) : Scalar(atan2(n, w));
      two_atan_nbyw_by_n = Scalar(2) * atan_nbyw / n;
    }

    return two_atan_nbyw_by_n * unit_quaternion().vec();
  }
  SOPHUS_FUNC void normalize() {
    Scalar length = unit_quaternion_nonconst().norm();
    MLOG_ENSURE(length >= Constants<Scalar>::epsilon(), "Quaternion ({}) should not be close to zero!",
                unit_quaternion_nonconst().coeffs().transpose());
    unit_quaternion_nonconst().coeffs() /= length;
  }
  SOPHUS_FUNC Transformation matrix() const { return unit_quaternion().toRotationMatrix(); }
  template <class OtherDerived>
  SOPHUS_FUNC SO3exBase<Derived>& operator=(SO3exBase<OtherDerived> const& other) {
    unit_quaternion_nonconst() = other.unit_quaternion();
    return *this;
  }
  template <typename OtherDerived>
  SOPHUS_FUNC SO3Product<OtherDerived> operator*(SO3exBase<OtherDerived> const& other) const {
    using QuaternionProductType = typename SO3Product<OtherDerived>::QuaternionType;
    const QuaternionType& a = unit_quaternion();
    const typename OtherDerived::QuaternionType& b = other.unit_quaternion();
    /// NOTE: We cannot use Eigen's Quaternion multiplication because it always
    /// returns a Quaternion of the same Scalar as this object, so it is not
    /// able to multiple Jets and doubles correctly.
    /// Notice SO3 here will normalize input quaternion, but u can inherit another SO3 to not normalize but be careful!
    return SO3Product<OtherDerived>(
        QuaternionProductType(a.w() * b.w() - a.x() * b.x() - a.y() * b.y() - a.z() * b.z(),
                              a.w() * b.x() + a.x() * b.w() + a.y() * b.z() - a.z() * b.y(),
                              a.w() * b.y() + a.y() * b.w() + a.z() * b.x() - a.x() * b.z(),
                              a.w() * b.z() + a.z() * b.w() + a.x() * b.y() - a.y() * b.x()));
  }
  template <typename PointDerived, typename = typename std::enable_if<IsFixedSizeVector<PointDerived, 3>::value>::type>
  SOPHUS_FUNC PointProduct<PointDerived> operator*(Eigen::MatrixBase<PointDerived> const& p) const {
    /// NOTE: We cannot use Eigen's Quaternion transformVector because it always
    /// returns a Vector3 of the same Scalar as this quaternion, so it is not
    /// able to be applied to Jets and doubles correctly.
    /// like "unit_quaternion_._transformVector(p);" only suitable for SO3<Scalar> * Vector3<Scalar>
    const QuaternionType& q = unit_quaternion();
    // uv type is the key merit
    PointProduct<PointDerived> uv = q.vec().cross(p);
    uv += uv;
    return p + q.w() * uv + q.vec().cross(uv);
  }
  template <typename HPointDerived,
            typename = typename std::enable_if<IsFixedSizeVector<HPointDerived, 4>::value>::type>
  SOPHUS_FUNC HomogeneousPointProduct<HPointDerived> operator*(Eigen::MatrixBase<HPointDerived> const& p) const {
    const PointProduct<HPointDerived> rp = *this * p.template head<3>();
    return HomogeneousPointProduct<HPointDerived>(rp(0), rp(1), rp(2), p(3));
  }
  template <typename OtherDerived,
            typename = typename std::enable_if<std::is_same<Scalar, ReturnScalar<OtherDerived>>::value>::type>
  SOPHUS_FUNC SO3exBase<Derived>& operator*=(SO3exBase<OtherDerived> const& other) {
    *static_cast<Derived*>(this) = (*this) * other;
    return *this;
  }

 private:
  SOPHUS_FUNC QuaternionType& unit_quaternion_nonconst() {
    return static_cast<Derived*>(this)->unit_quaternion_nonconst();
  }

  using Adjoint = Matrix<Scalar, DoF, DoF>;
  SOPHUS_FUNC Adjoint Adj() const { return matrix(); }
  template <class S = Scalar>
  SOPHUS_FUNC enable_if_t<std::is_floating_point<S>::value, S> angleX() const {
    assert(0);
    return S();
    // Sophus::Matrix3<Scalar> R = matrix();
    // Sophus::Matrix2<Scalar> Rx = R.template block<2, 2>(1, 1);
    // return SO2<Scalar>(makeRotationMatrix(Rx)).log();
  }
  template <class S = Scalar>
  SOPHUS_FUNC enable_if_t<std::is_floating_point<S>::value, S> angleY() const {
    assert(0);
    return S();
    /*Sophus::Matrix3<Scalar> R = matrix();
    Sophus::Matrix2<Scalar> Ry;
    // clang-format off
    Ry << R(0, 0), R(2, 0),
          R(0, 2), R(2, 2);
    // clang-format on
    return SO2<Scalar>(makeRotationMatrix(Ry)).log();*/
  }
  template <class S = Scalar>
  SOPHUS_FUNC enable_if_t<std::is_floating_point<S>::value, S> angleZ() const {
    assert(0);
    return S();
    // Sophus::Matrix3<Scalar> R = matrix();
    // Sophus::Matrix2<Scalar> Rz = R.template block<2, 2>(0, 0);
    // return SO2<Scalar>(makeRotationMatrix(Rz)).log();
  }
  SOPHUS_FUNC Scalar* data() { return unit_quaternion_nonconst().coeffs().data(); }
  SOPHUS_FUNC Scalar const* data() const { return unit_quaternion().coeffs().data(); }
  /// Returns derivative of  this * exp(x)  wrt x at x=0.
  SOPHUS_FUNC Matrix<Scalar, num_parameters, DoF> Dx_this_mul_exp_x_at_0() const {
    assert(0);
    Matrix<Scalar, num_parameters, DoF> J;
    return J;
  }
  // save order: phi then rho
  SOPHUS_FUNC Vector<Scalar, num_parameters> params() const { return unit_quaternion().coeffs(); }
  using Line = ParametrizedLine3<Scalar>;
  /// Group action on lines. This function rotates a parametrized line
  /// ``l(t) = o + t * d`` by the SO(3) element:
  /// Both direction ``d`` and origin ``o`` are rotated as a 3 dimensional point
  SOPHUS_FUNC Line operator*(Line const& l) const { return Line((*this) * l.origin(), (*this) * l.direction()); }
};

template <class Scalar_, int Options>
class SO3 : public SO3exBase<SO3<Scalar_, Options>> {
 protected:
  using Base = SO3exBase<SO3<Scalar_, Options>>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = Scalar_;
  using Base::DoF;
  using Base::num_parameters;
  // though it may be Map<QuaternionMember>, but in this class, it's QuaternionMember
  using typename Base::QuaternionType;
  using typename Base::Tangent;
  using typename Base::Transformation;

  // for private unit_quaternion_nonconst access
  friend class SO3exBase<SO3<Scalar, Options>>;

 protected:
  QuaternionType unit_quaternion_;

  SOPHUS_FUNC QuaternionType& unit_quaternion_nonconst() { return unit_quaternion_; }

 public:
  // SOPHUS_FUNC now is EIGEN_DEVICE_FUNC for CUDA usage: __host__ __device__ means cpu&&gpu both make this func
  SOPHUS_FUNC SO3() : unit_quaternion_(Scalar(1), Scalar(0), Scalar(0), Scalar(0)) {}
  // default Copy constructor
  /*SOPHUS_FUNC SO3(const SO3& other) {
    unit_quaternion_ = other.unit_quaternion_;
    // if we ensure all SO3 changing func will ensure the unit property, this normalize can be omitted
    unit_quaternion_.normalize();
  }*/
  SOPHUS_FUNC SO3(SO3 const& other) = default;
  template <class OtherDerived>
  SOPHUS_FUNC SO3(SO3exBase<OtherDerived> const& other) : unit_quaternion_(other.unit_quaternion()) {}
  // so3 will assert identity of R, but for convenience, we nomralize all R in so3ex
  SOPHUS_FUNC SO3(Transformation const& R) : unit_quaternion_(R) {
    MLOG_ENSURE(isOrthogonal(R), "R is not orthogonal:\n {}", R * R.transpose());
    MLOG_ENSURE(R.determinant() > Scalar(0), "det(R) is not positive: {}", R.determinant());
  }
  template <class D>
  SOPHUS_FUNC SO3(Eigen::QuaternionBase<D> const& q) : unit_quaternion_(q) {
    static_assert(std::is_same<typename Eigen::QuaternionBase<D>::Scalar, Scalar>::value,
                  "Input must be of same scalar type");
    Base::normalize();
  }

  SOPHUS_FUNC QuaternionType const& unit_quaternion() const { return unit_quaternion_; }

  SOPHUS_FUNC static SO3 exp(Tangent const& omega) {
    using std::abs;
    using std::cos;
    using std::sin;
    using std::sqrt;
    Scalar theta_impl = 0;
    Scalar* theta = &theta_impl;
    Scalar theta_sq = omega.squaredNorm();

    Scalar imag_factor;
    Scalar real_factor;
    if (theta_sq < Constants<Scalar>::epsilon() * Constants<Scalar>::epsilon()) {
      *theta = Scalar(0);
      Scalar theta_po4 = theta_sq * theta_sq;
      imag_factor = Scalar(0.5) - Scalar(1.0 / 48.0) * theta_sq + Scalar(1.0 / 3840.0) * theta_po4;
      real_factor = Scalar(1) - Scalar(1.0 / 8.0) * theta_sq + Scalar(1.0 / 384.0) * theta_po4;
    } else {
      *theta = sqrt(theta_sq);
      Scalar half_theta = Scalar(0.5) * (*theta);
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta / (*theta);
      real_factor = cos(half_theta);
    }

    SO3 q;
    q.unit_quaternion_nonconst() =
        QuaternionType(real_factor, imag_factor * omega.x(), imag_factor * omega.y(), imag_factor * omega.z());
    MLOG_ENSURE(abs(q.unit_quaternion().squaredNorm() - Scalar(1)) < Sophus::Constants<Scalar>::epsilon(),
                "SO3::exp failed! omega: {}, real: {}, img: {}", omega.transpose(), real_factor, imag_factor);
    return q;
  }
  SOPHUS_FUNC static Transformation hat(Tangent const& omega) {
    Transformation Omega;
    // clang-format off
    Omega << Scalar(0), -omega(2),  omega(1),
        omega(2), Scalar(0), -omega(0),
        -omega(1),  omega(0), Scalar(0);
    // clang-format on
    return Omega;
  }
  SOPHUS_FUNC static Tangent vee(Transformation const& Omega) { return Tangent(Omega(2, 1), Omega(0, 2), Omega(1, 0)); }

 private:
  // private some Base func. for safety and prepared to use
  SOPHUS_FUNC Scalar* data() { return this->so3_ex_.data(); }
  SOPHUS_FUNC Scalar const* data() const { return this->so3_ex_.data(); }
  /// Returns derivative of exp(x)(xyzw) wrt. x(xyz).
  SOPHUS_FUNC static Sophus::Matrix<Scalar, num_parameters, DoF> Dx_exp_x(Tangent const& omega) {
    assert(0);
    Sophus::Matrix<Scalar, num_parameters, DoF> J;
    return J;
  }
  /// Returns derivative of exp(x)(xyzw) wrt. x(xyz) at x=0.
  SOPHUS_FUNC static Sophus::Matrix<Scalar, num_parameters, DoF> Dx_exp_x_at_0() {
    Sophus::Matrix<Scalar, num_parameters, DoF> J;
    // clang-format off
    J << Scalar(0.5),   Scalar(0),   Scalar(0),
           Scalar(0), Scalar(0.5),   Scalar(0),
           Scalar(0),   Scalar(0), Scalar(0.5),
           Scalar(0),   Scalar(0),   Scalar(0);
    // clang-format on
    return J;
  }
  /// Returns derivative of exp(x).matrix() wrt. ``x_i at x=0``.
  SOPHUS_FUNC static Transformation Dxi_exp_x_matrix_at_0(int i) { return generator(i); }
  template <class S = Scalar>
  static SOPHUS_FUNC enable_if_t<std::is_floating_point<S>::value, SO3> fitToSO3(Transformation const& R) {
    return SO3(::Sophus::makeRotationMatrix(R));
  }
  SOPHUS_FUNC static Transformation generator(int i) {
    MLOG_ENSURE(i >= 0 && i <= 2, "i should be in range [0,2].");
    Tangent e;
    e.setZero();
    e[i] = Scalar(1);
    return hat(e);
  }
  // for so3:(a^b)^=a^b^-b^a^=[a,b]
  SOPHUS_FUNC static Tangent lieBracket(Tangent const& omega1, Tangent const& omega2) { return omega1.cross(omega2); }
  static SOPHUS_FUNC SO3 rotX(Scalar const& x) { return SO3::exp(Tangent(x, Scalar(0), Scalar(0))); }
  static SOPHUS_FUNC SO3 rotY(Scalar const& y) { return SO3::exp(Tangent(Scalar(0), y, Scalar(0))); }
  static SOPHUS_FUNC SO3 rotZ(Scalar const& z) { return SO3::exp(Tangent(Scalar(0), Scalar(0), z)); }
  template <class UniformRandomBitGenerator>
  static SO3 sampleUniform(UniformRandomBitGenerator& generator) {
    static_assert(IsUniformRandomBitGenerator<UniformRandomBitGenerator>::value,
                  "generator must meet the UniformRandomBitGenerator concept");
    std::uniform_real_distribution<Scalar> uniform(Scalar(0), Scalar(1));
    std::uniform_real_distribution<Scalar> uniform_twopi(Scalar(0), 2 * Constants<Scalar>::pi());
    const Scalar u1 = uniform(generator);
    const Scalar u2 = uniform_twopi(generator);
    const Scalar u3 = uniform_twopi(generator);
    const Scalar a = sqrt(1 - u1);
    const Scalar b = sqrt(u1);
    return SO3(QuaternionType(a * sin(u2), a * cos(u2), b * sin(u3), b * cos(u3)));
  }
};

}  // namespace Sophus

namespace Eigen {
/// Specialization of Eigen::Map for ``SO3``; derived from SO3Base.
/// Allows us to wrap SO3 objects around POD array (e.g. external c style quaternion).
template <class Scalar_, int Options>
class Map<Sophus::SO3<Scalar_>, Options> : public Sophus::SO3exBase<Map<Sophus::SO3<Scalar_>, Options>> {
 public:
  using Base = Sophus::SO3exBase<Map<Sophus::SO3<Scalar_>, Options>>;
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using typename Base::QuaternionType;

  /// ``Base`` is friend so unit_quaternion_nonconst can be accessed from ``Base``.
  friend class Sophus::SO3exBase<Map<Sophus::SO3<Scalar_>, Options>>;

  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar* coeffs) : unit_quaternion_(coeffs) {}

  /// Accessor of unit quaternion.
  SOPHUS_FUNC QuaternionType const& unit_quaternion() const { return unit_quaternion_; }

 protected:
  QuaternionType unit_quaternion_;

  /// Mutator of unit_quaternion is protected to ensure class invariant.
  SOPHUS_FUNC QuaternionType& unit_quaternion_nonconst() { return unit_quaternion_; }
};

/// Specialization of Eigen::Map for ``SO3 const``; derived from SO3Base.
///
/// Allows us to wrap SO3 objects around POD array (e.g. external c style
/// quaternion).
template <class Scalar_, int Options>
class Map<Sophus::SO3<Scalar_> const, Options> : public Sophus::SO3exBase<Map<Sophus::SO3<Scalar_> const, Options>> {
 public:
  using Base = Sophus::SO3exBase<Map<Sophus::SO3<Scalar_> const, Options>>;
  using Scalar = Scalar_;
  using Transformation = typename Base::Transformation;
  using Point = typename Base::Point;
  using HomogeneousPoint = typename Base::HomogeneousPoint;
  using Tangent = typename Base::Tangent;
  using Adjoint = typename Base::Adjoint;
  using typename Base::QuaternionType;

  using Base::operator*=;
  using Base::operator*;

  SOPHUS_FUNC Map(Scalar const* coeffs) : unit_quaternion_(coeffs) {}

  /// Accessor of unit quaternion.
  SOPHUS_FUNC QuaternionType const& unit_quaternion() const { return unit_quaternion_; }

 protected:
  /// Mutator of unit_quaternion is protected to ensure class invariant.
  QuaternionType const unit_quaternion_;
};
}  // namespace Eigen
