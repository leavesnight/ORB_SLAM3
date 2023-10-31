//
// Created by leavesnight on 7/6/23.
//

#pragma once

#include <random>
#include <type_traits>
#include <Eigen/Core>

#define SOPHUS_FUNC EIGEN_DEVICE_FUNC

namespace Sophus {
template <class Scalar, int M, int Options = 0>
using Vector = Eigen::Matrix<Scalar, M, 1, Options>;

template <class Scalar, int Options = 0>
using Vector2 = Vector<Scalar, 2, Options>;
using Vector2f = Vector2<float>;
using Vector2d = Vector2<double>;

template <class Scalar, int Options = 0>
using Vector3 = Vector<Scalar, 3, Options>;
using Vector3f = Vector3<float>;
using Vector3d = Vector3<double>;

template <class Scalar>
using Vector4 = Vector<Scalar, 4>;
using Vector4f = Vector4<float>;
using Vector4d = Vector4<double>;

template <class Scalar>
using Vector6 = Vector<Scalar, 6>;
using Vector6f = Vector6<float>;
using Vector6d = Vector6<double>;

template <class Scalar>
using Vector7 = Vector<Scalar, 7>;
using Vector7f = Vector7<float>;
using Vector7d = Vector7<double>;

template <class Scalar, int M, int N>
using Matrix = Eigen::Matrix<Scalar, M, N>;

template <class Scalar>
using Matrix2 = Matrix<Scalar, 2, 2>;
using Matrix2f = Matrix2<float>;
using Matrix2d = Matrix2<double>;

template <class Scalar>
using Matrix3 = Matrix<Scalar, 3, 3>;
using Matrix3f = Matrix3<float>;
using Matrix3d = Matrix3<double>;

template <class Scalar>
using Matrix4 = Matrix<Scalar, 4, 4>;
using Matrix4f = Matrix4<float>;
using Matrix4d = Matrix4<double>;

template <class Scalar>
using Matrix6 = Matrix<Scalar, 6, 6>;
using Matrix6f = Matrix6<float>;
using Matrix6d = Matrix6<double>;

template <class Scalar>
using Matrix7 = Matrix<Scalar, 7, 7>;
using Matrix7f = Matrix7<float>;
using Matrix7d = Matrix7<double>;

template <class Scalar, int N, int Options = 0>
using ParametrizedLine = Eigen::ParametrizedLine<Scalar, N, Options>;

template <class Scalar, int Options = 0>
using ParametrizedLine3 = ParametrizedLine<Scalar, 3, Options>;
using ParametrizedLine3f = ParametrizedLine3<float>;
using ParametrizedLine3d = ParametrizedLine3<double>;

template <class Scalar, int Options = 0>
using ParametrizedLine2 = ParametrizedLine<Scalar, 2, Options>;
using ParametrizedLine2f = ParametrizedLine2<float>;
using ParametrizedLine2d = ParametrizedLine2<double>;

template <class Scalar>
struct Constants {
  SOPHUS_FUNC static Scalar epsilon() { return Scalar(1e-10); }
  SOPHUS_FUNC static Scalar pi() { return Scalar(3.141592653589793238462643383279502884); }
};
template <>
struct Constants<float> {
  SOPHUS_FUNC static float constexpr epsilon() { return static_cast<float>(1e-5); }
  SOPHUS_FUNC static float constexpr pi() { return 3.141592653589793238462643383279502884f; }
};

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
/// If the Vector type is of fixed size, then IsFixedSizeVector::value will be true.
template <typename Vector, int NumDimensions,
          typename = enable_if_t<Vector::RowsAtCompileTime == NumDimensions && Vector::ColsAtCompileTime == 1>>
struct IsFixedSizeVector : std::true_type {};
template <class G>
struct IsUniformRandomBitGenerator {
  static const bool value = std::is_unsigned<typename G::result_type>::value &&
                            std::is_unsigned<decltype(G::min())>::value && std::is_unsigned<decltype(G::max())>::value;
};

/// Takes in arbitrary square matrix and returns true if it is
/// orthogonal.
template <class D>
SOPHUS_FUNC bool isOrthogonal(Eigen::MatrixBase<D> const& R) {
  using Scalar = typename D::Scalar;
  static int const N = D::RowsAtCompileTime;
  static int const M = D::ColsAtCompileTime;

  static_assert(N == M, "must be a square matrix");
  static_assert(N >= 2, "must have compile time dimension >= 2");

  return (R * R.transpose() - Matrix<Scalar, N, N>::Identity()).norm() < Constants<Scalar>::epsilon();
}

}  // namespace Sophus
