/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
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

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <ctime>
#include <sstream>
#include <memory>
#include <rapidjson/document.h>

#include <opencv2/core/core.hpp>


#include<System.h>
#include "ImuTypes.h"
#include "Optimizer.h"

using namespace std;
using namespace rapidjson;

void AlignImgs(vector<vector<double>> &vtmcam, vector<vector<string>> &vstrimg) {
  CV_Assert(vtmcam.size() == vstrimg.size());
  if (!vtmcam.size()) return;
  CV_Assert(vtmcam[0].size() == vstrimg[0].size());
  size_t n_cams = vtmcam.size();
  size_t i = 0;
  const double synch_allow = 0.010;
  while (i < vtmcam[0].size()) {
    bool bplus = true;
    // cout << "tm0:" << vtmcam[0][i] << " ";
    for (size_t j = 1; j < n_cams; ++j) {
      while (vtmcam[j].size() > i && (vtmcam[j][i] < vtmcam[0][i] - synch_allow || vtmcam[0].size() <= i)) {
        vtmcam[j].erase(vtmcam[j].begin() + i);
        vstrimg[j].erase(vstrimg[j].begin() + i);
      }
      if (vtmcam[j].size() <= i) {
        CV_Assert(i == vtmcam[j].size());
        if (vtmcam[0].size() > i) {
          vtmcam[0].resize(i);
          vstrimg[0].resize(i);
        }
        continue;
      }
      if (vtmcam[j][i] > vtmcam[0][i] + synch_allow) {
        vtmcam[0].erase(vtmcam[0].begin() + i);
        vstrimg[0].erase(vstrimg[0].begin() + i);
        bplus = false;
      }
      // cout << "tm" << j << ":" << vtmcam[j][i] << " ";
    }
    // cout << endl;
    if (bplus) ++i;
  }
  auto n0 = vtmcam[0].size();
  for (size_t j = 0; j < n_cams; ++j) {
    CV_Assert(vtmcam[j].size() >= n0);
    CV_Assert(vstrimg[j].size() == vtmcam[j].size());
    vtmcam[j].resize(n0);
    vstrimg[j].resize(n0);
    CV_Assert(vstrimg[j].size() == n0);
  }
  cout << "After align img size=" << vtmcam[0].size() << endl;
}

static void GetFileNames(const string& path, vector<string>& filenames, const string& suffix=".pgm", const string& prefix="");

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps, const string& suffix = ".pgm");

void LoadIMU(const vector<string> &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl << "Usage: ./stereo_inertial_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1(path_to_image_folder_2 ... path_to_image_folder_N) " << endl;
        return 1;
    }
    int dataset_type = 0;
    string mode="VIO";
    if (argc > 4) {
        mode = argv[4];
        for (int i = 4; i < argc - 1; ++i)
            argv[i] = argv[i + 1];
        --argc;
    }

    const int num_seq = (argc-3);
    cout << "num_seq = " << num_seq << endl;

    // Load all sequences:
    int seq;
    vector< vector<vector<string>> > vstrImageLeft;
    vector< vector<vector<double>> > vTimestampsCam;
    vector< vector<vector<cv::Point3f>> > vAcc, vGyro;
    vector< vector<vector<double>> > vTimestampsImu;
    vector<int> nImages, nImagesUsed;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageLeft.resize(num_seq, vector<vector<string>>(1));
    vTimestampsCam.resize(num_seq, vector<vector<double>>(1));
    vAcc.resize(num_seq, vector<vector<cv::Point3f>>(1));
    vGyro.resize(num_seq, vector<vector<cv::Point3f>>(1));
    vTimestampsImu.resize(num_seq, vector<vector<double>>(1));
    nImages.resize(num_seq);
    nImagesUsed.resize(num_seq, 0);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++) {
      cout << "Loading images for sequence " << seq << "...";

      string pathSeq(argv[(seq) + 3]);

      string pathCam0 = pathSeq + "/Camera8";
      vector<string> pathImu = {pathSeq + "/Sensors/gyroscope.xml",
                                pathSeq + "/Sensors/accelerometer.xml"};

      auto &vstrimg = vstrImageLeft[seq];
      auto &vtmcam = vTimestampsCam[seq];
      LoadImages(pathCam0, vstrimg[0], vtmcam[0]);
      if (vtmcam[0].empty()) {
        dataset_type = 1;
        int n_cams_max = 4;
        vstrimg.resize(n_cams_max);
        vtmcam.resize(n_cams_max);
        for (int i = 0; i < n_cams_max; ++i) {
          pathCam0 = pathSeq + "/Camera" + to_string(i) + "/images";
          LoadImages(pathCam0, vstrimg[i], vtmcam[i], ".bmp");
          if (vtmcam[i].empty()) {
            vtmcam.resize(i);
            vstrimg.resize(i);
            break;
          }
        }
      }
      cout << "LOADED!" << endl;

      cout << "Loading IMU for sequence " << seq << "...";
      auto &vacc = vAcc[seq], &vgyr = vGyro[seq];
      auto &vtmimu = vTimestampsImu[seq];
      LoadIMU(pathImu, vtmimu[0], vacc[0], vgyr[0]);
      if (vtmimu[0].empty()) {
        int n_imu_max = 3;
        vtmimu.resize(n_imu_max);
        vacc.resize(n_imu_max);
        vgyr.resize(n_imu_max);
        pathImu.resize(1);
        for (int i = 0; i < n_imu_max; ++i) {
          pathImu[0] = pathSeq + "/IMU" + to_string(i) + "/data.json";
          LoadIMU(pathImu, vtmimu[i], vacc[i], vgyr[i]);
          if (vtmimu[i].empty()) {
            vtmimu.resize(i);
            vacc.resize(i);
            vgyr.resize(i);
            break;
          }
        }
      }
      cout << "LOADED!" << endl;

//#define TRANS_IMU2TXT
#ifdef TRANS_IMU2TXT
    ofstream fout("./" + to_string(seq) + ".txt");
    for (int i = 0; i < vTimestampsImu[seq].size(); ++i) {
      if (vAcc[seq].size() <= i)
        break;
      if (vGyro[seq].size() <= i)
        break;
      fout << (unsigned long)(vTimestampsImu[seq][i] * 1e9) << " "
           << vAcc[seq][i].x << " " << vAcc[seq][i].y << " " << vAcc[seq][i].z
           << " " << vGyro[seq][i].x << " " << vGyro[seq][i].y << " "
           << vGyro[seq][i].z << endl;
    }
    fout.close();
#endif

    nImages[seq] = vstrimg[0].size();
    tot_images += nImages[seq];
    nImu[seq] = vtmimu[0].size();

    if ((nImages[seq] <= 0) || (nImu[seq] <= 0)) {
      cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
      return 1;
    }

    // Find first imu to be considered, supposing imu measurements start first

    while (vtmimu[0][first_imu[seq]] <= vtmcam[0][0])
      first_imu[seq]++;
    if (0 < first_imu[seq])
      first_imu[seq]--; // first imu measurement to be considered

    while (vtmcam[0][nImages[seq] - 1] > vtmimu[0].back()) {
      nImages[seq]--;
    }

    AlignImgs(vtmcam, vstrimg);
    nImages[seq] = vtmcam[0].size();

    cout << "first_imu[seq]=" << first_imu[seq]
         << ", after deleted, nImages[seq]=" << nImages[seq] << endl;
  }

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }
    cv::FileNode fnimumode = fsSettings["IMU.mode"];
    int mode_imu = fnimumode.empty() ? 0 : (int)fnimumode;
    if (1 == (int)mode_imu) { // 1 menas -gy,gxgz/-ay,axaz
      for (int seq = 0; seq < num_seq; ++seq) {
        auto &vacc = vAcc[seq], &vgyr = vGyro[seq];
        for (int i = 0; i < vacc.size(); ++i)
          for (int j = 0; j < vacc[i].size(); ++j) {
            swap(vacc[i][j].x, vacc[i][j].y);
            vacc[i][j].y = -vacc[i][j].y;
          }
        for (int i = 0; i < vgyr.size(); ++i)
          for (int j = 0; j < vgyr[i].size(); ++j) {
            swap(vgyr[i][j].x, vgyr[i][j].y);
            vgyr[i][j].y = -vgyr[i][j].y;
          }
      }
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images, 0);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    shared_ptr<ORB_SLAM3::System> pSLAM;
    if ("VIO" == mode)
        pSLAM=make_shared<ORB_SLAM3::System>(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO, true);
    else
        pSLAM=make_shared<ORB_SLAM3::System>(argv[1],argv[2],ORB_SLAM3::System::STEREO, true);
    ORB_SLAM3::System& SLAM=*pSLAM;

    cv::FileNode fnfps = fsSettings["Camera.fps"];
    for (seq = 0; seq<num_seq; seq++)
    {
      auto &vstrimg = vstrImageLeft[seq];
      auto &vtmcam = vTimestampsCam[seq];
      auto &vacc = vAcc[seq], &vgyr = vGyro[seq];
      auto &vtmimu = vTimestampsImu[seq];
      int fpsrat = 1;
      if (!fnfps.empty() && nImages[seq] > 1) {
        double fps = (double)fnfps;
        double fpsreal =
            vtmcam[0].size() / (vtmcam[0].back() - vtmcam[0].front());
        fpsrat = (int)(fpsreal / fps + 0.5);
        if (fpsrat < 1)
          fpsrat = 1;
        cout << "fps ratio: " << fpsrat << endl;
      }

        cv::Mat ims[2];
        // Seq loop
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        int proccIm = 0;
//        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        for(int ni=0; ni<nImages[seq]; ni+=fpsrat, proccIm++)
        {
          double tframe = vtmcam[0][ni];

            // Read left and right images from file
//            imLeft = cv::imread(vstrImageLeft[seq][ni],cv::IMREAD_UNCHANGED);
            ims[0] = cv::imread(vstrimg[0][ni],cv::IMREAD_GRAYSCALE);
            if (!dataset_type) {
              ims[1] = ims[0].colRange(ims[0].cols / 2, ims[0].cols);
              ims[0] = ims[0].colRange(0, ims[0].cols / 2);
            } else {
              //              for (int i = 1; i < vstrimg.size(); ++i) {
              //                ims[i] = cv::imread(vstrimg[i][ni], cv::IMREAD_GRAYSCALE);
              //              }
              ims[1] = cv::imread(vstrimg[3][ni], cv::IMREAD_GRAYSCALE);
            }
            // clahe
//            clahe->apply(imLeft,imLeft);
//            clahe->apply(imRight,imRight);

            if(ims[0].empty() || ims[1].empty()) {
              cerr << endl
                   << "Failed to load image at: " << vstrimg[0][ni] << endl;
              return 1;
            }

            // Load imu measurements from previous frame
            vImuMeas.clear();

            if(ni>0) {
              while (vtmimu[0][first_imu[seq]] <= vtmcam[0][ni]) {
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                    vacc[0][first_imu[seq]].x, vacc[0][first_imu[seq]].y,
                    vacc[0][first_imu[seq]].z, vgyr[0][first_imu[seq]].x,
                    vgyr[0][first_imu[seq]].y, vgyr[0][first_imu[seq]].z,
                    vtmimu[0][first_imu[seq]]));
                first_imu[seq]++;
              }
            }

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the images to the SLAM system
            if ("VIO" == mode)
                SLAM.TrackStereo(ims[0],ims[1],tframe,vImuMeas);
            else
                SLAM.TrackStereo(ims[0],ims[1],tframe);

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            vTimesTrack[ni]=ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vtmcam[0][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vtmcam[0][ni-1];
            T *= fpsrat;

            if(ttrack<T)
                usleep((T-ttrack)*1e6); // 1e6
            ++nImagesUsed[seq];
        }

        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }
    // Stop all threads
    SLAM.Shutdown();

    int ni = 0;
    for (int seq = 0; seq < num_seq; ++ seq) {
      // Tracking time statistics
      sort(vTimesTrack.begin(), vTimesTrack.end());
      float totaltime = 0;
      for (; ni < nImages[seq]; ni++) {
        totaltime += vTimesTrack[ni];
      }
      cout << "-------seq" << seq << endl << endl;
      cout << "mean tracking time: " << totaltime / nImagesUsed[seq] << endl;
      cout << "max tracking time: " << vTimesTrack[ni - 1] << endl;
    }

    // Save camera trajectory
//    SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt", 1.0);
//    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt", 1.0);
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveTrajectoryTUM("CameraTrajectoryCamPoseIMUBias.txt", 1);

    return 0;
}


#include <dirent.h>//for unix, directory entries related definition header
#include <libgen.h>

static void GetFileNames(const string& path, vector<string>& filenames, const string& suffix, const string& prefix) {
    DIR *pDir;
    struct dirent* ptr;
    if (!(pDir = opendir(path.c_str()))) {
        cerr<<path<<" opendir failed"<<endl;
        return;
    }

    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") && strcmp(ptr->d_name, "..")) {
            string d_name = string(ptr->d_name);
            size_t pos_suffix = d_name.rfind(suffix), pos_prefix = 0;
            if (!prefix.empty())
                pos_prefix = d_name.find(prefix);
            if (string::npos != pos_suffix && pos_suffix + suffix.length() == d_name.length()) {
                if (!pos_prefix)
                    filenames.push_back(path + "/" + ptr->d_name);
            }
        }
    }
    closedir(pDir);
}

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps, const string& suffix)
{
    ifstream fTimes;
    GetFileNames(strImagePath, vstrImages, suffix);
    sort(vstrImages.begin(),vstrImages.end());
    for (int i = 0; i < vstrImages.size(); ++i) {
        string dir_path = dirname(strdup(vstrImages[i].c_str()));
        int offset = dir_path.length();
        if (dir_path[offset - 1] != '/' and dir_path[offset - 1] != '\\')
            ++offset;
        double ftmp = strtod(vstrImages[i].substr(offset).c_str(), 0) * 1e-9;
        vTimeStamps.push_back(ftmp);
    }
}

typedef enum KeyStrType{
    kStrStart,
    kStrEnd,
    kStrDivide,
    kNumKeyStrType
};

static int GetFloatArray(const string &str_tmp, const string *keystr, size_t &last_pos, vector<double> &ret_vals) {
    size_t pos_keystr = str_tmp.find(keystr[kStrStart], last_pos);
    if (keystr[kStrStart] == "")
        pos_keystr = 0;
    if (string::npos != pos_keystr) {
        last_pos = pos_keystr + keystr[kStrStart].length();
        if (keystr[kStrEnd] == "")
            pos_keystr = str_tmp.length();
        else
            pos_keystr = str_tmp.find(keystr[kStrEnd], last_pos);
        string str_data = str_tmp.substr(last_pos, pos_keystr - last_pos);
        char *endptr = 0;
        last_pos = 0;
        while (last_pos < str_data.length()) {
            string str_data_tmp = str_data.substr(last_pos);
            ret_vals.push_back(strtod(str_data_tmp.c_str(), &endptr));
            last_pos += endptr - str_data_tmp.c_str() + keystr[kStrDivide].length();
        }
        last_pos = pos_keystr + keystr[kStrEnd].length();

        return 0;
    }

    return -1;
}

void LoadIMU(const vector<string> &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu, fAcc;
    fImu.open(strImuPath[0].c_str());
    if (!fImu.is_open()) return;
    int mode = 0;
    if (strImuPath.size() > 1) {
        fAcc.open(strImuPath[1].c_str());
        if (!fAcc.is_open()) return;
    } else
      mode = 1;
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);
    vector<double> ret_vals;

    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);

        if (!mode) {
          string keystr[kNumKeyStrType] = {"Data x='", "' time", "' *='"};
          size_t last_pos = 0;
          if (!GetFloatArray(s, keystr, last_pos, ret_vals)) {
            CV_Assert(3 == ret_vals.size());
            vGyro.push_back(cv::Point3f(ret_vals[0], ret_vals[1], ret_vals[2]));
            ret_vals.clear();
          }

          keystr[kStrStart] = "stamp='";
          keystr[kStrEnd] = "' index";
          if (!GetFloatArray(s, keystr, last_pos, ret_vals)) {
            CV_Assert(1 == ret_vals.size());
            vTimeStamps.push_back(ret_vals[0] / 1e9);
            ret_vals.clear();
          }
        } else {
          rapidjson::Document imu_doc;
          imu_doc.Parse<0>(s.c_str());
          Value &IMUData = imu_doc["Sequence"]["Dataset"]["Data"];
          for (std::size_t i = 0; i < IMUData.Size(); i++) {
            vTimeStamps.push_back(IMUData[i]["timestamp"].GetUint64() / 1.e9);
            vGyro.push_back(cv::Point3f(IMUData[i]["g_x"].GetFloat(), IMUData[i]["g_y"].GetFloat(), IMUData[i]["g_z"].GetFloat()));
            vAcc.push_back(cv::Point3f(IMUData[i]["a_x"].GetFloat(), IMUData[i]["a_y"].GetFloat(), IMUData[i]["a_z"].GetFloat()));
          }
        }
    }
    if (mode) return;

    int id_acc = 0;
    while(!fAcc.eof() && !fAcc.fail())
    {
        string s;
        getline(fAcc,s);

        string keystr[kNumKeyStrType] = {"Data x='", "' time", "' *='"};
        size_t last_pos = 0;
        if (!GetFloatArray(s, keystr, last_pos, ret_vals)) {
            CV_Assert(3 == ret_vals.size());
            vAcc.push_back(cv::Point3f(ret_vals[0],ret_vals[1],ret_vals[2]));
            ret_vals.clear();
        }

        keystr[kStrStart] = "stamp='";
        keystr[kStrEnd] = "' index";
        if (!GetFloatArray(s, keystr, last_pos, ret_vals)) {
            CV_Assert(1 == ret_vals.size());
            if (vTimeStamps.size() <= id_acc) {
              break;
            }
            CV_Assert(vTimeStamps[id_acc++] == ret_vals[0]/1e9);
            ret_vals.clear();
        }
        if (vTimeStamps.size() <= id_acc) {
          vAcc.resize(vTimeStamps.size());
          break;
        }
    }
    if (vAcc.size() < vTimeStamps.size()) {
      vTimeStamps.resize(vAcc.size());
      vGyro.resize(vAcc.size());
    }
}
