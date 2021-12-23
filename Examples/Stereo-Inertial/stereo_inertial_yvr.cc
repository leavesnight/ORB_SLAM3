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

#include <opencv2/core/core.hpp>


#include<System.h>
#include "ImuTypes.h"
#include "Optimizer.h"

using namespace std;

static void GetFileNames(const string& path, vector<string>& filenames, const string& suffix=".pgm", const string& prefix="");

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps);

void LoadIMU(const vector<string> &strImuPath, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl << "Usage: ./stereo_inertial_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1(path_to_image_folder_2 ... path_to_image_folder_N) " << endl;
        return 1;
    }
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
    vector< vector<string> > vstrImageLeft;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageLeft.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";

        string pathSeq(argv[(seq) + 3]);

        string pathCam0 = pathSeq + "/Camera8";
        vector<string> pathImu = {pathSeq + "/Sensors/gyroscope.xml", pathSeq + "/Sensors/accelerometer.xml"};

        LoadImages(pathCam0, vstrImageLeft[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(pathImu, vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
        cout << "LOADED!" << endl;

//#define TRANS_IMU2TXT
#ifdef TRANS_IMU2TXT
        ofstream fout("./"+to_string(seq)+".txt");
        for (int i =0; i < vTimestampsImu[seq].size();++i) {
          if (vAcc[seq].size()<=i) break;
          if (vGyro[seq].size()<=i) break;
          fout << (unsigned long)(vTimestampsImu[seq][i]*1e9) << " "<< vAcc[seq][i].x << " " << vAcc[seq][i].y<<" "<<vAcc[seq][i].z<<" "<<
            vGyro[seq][i].x<<" "<<vGyro[seq][i].y<<" "<<vGyro[seq][i].z<<endl;
        }
        fout.close();
#endif

        nImages[seq] = vstrImageLeft[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first

        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered

        while(vTimestampsCam[seq][nImages[seq] - 1] > vTimestampsImu[seq].back()) {
            nImages[seq]--;
        }

        cout<<"first_imu[seq]="<<first_imu[seq]<<", after delted, nImages[seq]="<<nImages[seq]<<endl;
    }

    // Read rectification parameters
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    shared_ptr<ORB_SLAM3::System> pSLAM;
    if ("VIO" == mode)
        pSLAM=make_shared<ORB_SLAM3::System>(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO, true);
    else
        pSLAM=make_shared<ORB_SLAM3::System>(argv[1],argv[2],ORB_SLAM3::System::STEREO, true);
    ORB_SLAM3::System& SLAM=*pSLAM;

    for (seq = 0; seq<num_seq; seq++)
    {
        cv::Mat imLeft, imRight;
        // Seq loop
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        int proccIm = 0;
//        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // Read left and right images from file
//            imLeft = cv::imread(vstrImageLeft[seq][ni],cv::IMREAD_UNCHANGED);
            imLeft = cv::imread(vstrImageLeft[seq][ni],cv::IMREAD_GRAYSCALE);
            imRight = imLeft.colRange(imLeft.cols / 2, imLeft.cols);
            imLeft = imLeft.colRange(0, imLeft.cols / 2);
            // clahe
//            clahe->apply(imLeft,imLeft);
//            clahe->apply(imRight,imRight);

            double tframe = vTimestampsCam[seq][ni];

            if(imLeft.empty() || imRight.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageLeft[seq][ni] << endl;
                return 1;
            }

            // Load imu measurements from previous frame
            vImuMeas.clear();

            if(ni>0)
                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni]) // while(vTimestampsImu[first_imu]<=vTimestampsCam[ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(vAcc[seq][first_imu[seq]].x,vAcc[seq][first_imu[seq]].y,vAcc[seq][first_imu[seq]].z,
                                                             vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                                                             vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the images to the SLAM system
            if ("VIO" == mode)
                SLAM.TrackStereo(imLeft,imRight,tframe,vImuMeas);
            else
                SLAM.TrackStereo(imLeft,imRight,tframe);

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
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6); // 1e6
        }

        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }


    }
    // Stop all threads
    SLAM.Shutdown();


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

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    GetFileNames(strImagePath, vstrImages);
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
    if (strImuPath.size() > 1) {
        fAcc.open(strImuPath[1].c_str());
    }
    vTimeStamps.reserve(5000);
    vAcc.reserve(5000);
    vGyro.reserve(5000);
    vector<double> ret_vals;

    while(!fImu.eof())
    {
        string s;
        getline(fImu,s);

        string keystr[kNumKeyStrType] = {"Data x='", "' time", "' *='"};
        size_t last_pos = 0;
        if (!GetFloatArray(s, keystr, last_pos, ret_vals)) {
            CV_Assert(3 == ret_vals.size());
            vGyro.push_back(cv::Point3f(ret_vals[0],ret_vals[1],ret_vals[2]));
            ret_vals.clear();
        }

        keystr[kStrStart] = "stamp='";
        keystr[kStrEnd] = "' index";
        if (!GetFloatArray(s, keystr, last_pos, ret_vals)) {
            CV_Assert(1 == ret_vals.size());
            vTimeStamps.push_back(ret_vals[0]/1e9);
            ret_vals.clear();
        }
    }

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
