//
// Created by leavesnight on 9/27/23.
//

#include "include/G2oTypes.h"
#include "common/camera_models/camera.h"
#include "common/mlog/timer.h"
using namespace VIEO_SLAM;
using namespace g2o;
using namespace std;
using namespace ORB_SLAM3;

bool TestVertexPose() {
  VertexNavStatePR *pr = new VertexNavStatePR();
  vector<camm::Camera::Ptr> CamInsts(
      1, camm::CreateCameraInstance(VIEO_SLAM::camm::Camera::kPinhole, 0, 0, 0,
                                    vector<camm::Camera::Tdata>(
                                        {456.7149963378906, 456.7149963378906, 364.4412078857422, 256.9516830444336})));
  Eigen::Matrix4d eigTbcr;
  eigTbcr << 0.016320884227752686, -0.99980568885803223, 0.011063323356211185, -0.021640146151185036,
      0.99971014261245728, 0.016513228416442871, 0.017522787675261497, -0.064676985144615173, -0.017702072858810425,
      0.010774127207696438, 0.99978524446487427, 0.0098107308149337769, 0, 0, 0, 1;
  Sophus::SE3exd Tcrb(eigTbcr);
  Tcrb = Tcrb.inverse();
  float bf = 0.11007784307003021 * 456.7149963378906;
  pr->SetParams(CamInsts, Tcrb, &bf);
  NavStateBased ns;
  {
    ns.pwb_ << (double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX;
    ns.Rwb_ = Sophus::SO3exd::exp(
        Eigen::Vector3d((double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX));
    ns.vwb_ << (double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX;
    ns.bg_ << (double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX;
    ns.ba_ << (double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX;
    ns.bg_ *= 1e-3;
    ns.ba_ *= 1e-2;
  }
  pr->SetNavState(ns);
  pr->setId(0);
  pr->setFixed(false);
  // P(xc,xp)=P(xc)*P(xp|xc), P(xc) is called marginalized/Schur elimination,
  // [B-E*C^(-1)*E.t()]*deltaXc=v-E*C^(-1)*w, H*deltaX=g=[v;w]; used in Sparse solver
  assert(!pr->marginalized());
  pr->setMarginalized(0);
  using Vector6d = Eigen::Matrix<double, 6, 1>;
  // ref ORB3 InertialOptimization times 200 * g2o LM inner default max iters num 10, normal times 20 is reasonable
  int num_tests = (int)2e3;
  mlog::Timer timer;
  for (int i = 0; i < num_tests; ++i) {
    Vector6d update;
    update << (double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX,
        (double)rand() / RAND_MAX, (double)rand() / RAND_MAX, (double)rand() / RAND_MAX;
    pr->oplus(update.data());
  }
  auto dt_test = timer.GetDTms(true);
  PRINT_INFO_MUTEX(mlog::FormatString("num_tests({}) Vertex oplus time cost={}ms, normally / 10", num_tests, dt_test)
                       << endl);

  delete pr;
  pr = nullptr;
  return true;
}

int main(int argc, char **argv) {
  PRINT_INFO_MUTEX("Hello, now test optimizer_ba!" << endl);

  PRINT_INFO_MUTEX("test VertexPose" << endl);
  TestVertexPose();

  PRINT_INFO_MUTEX("test g2o LBA" << endl);

  return 0;
}
