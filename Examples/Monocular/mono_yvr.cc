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
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

#include <libgen.h>

using namespace std;

static void GetFileNames(const string& path, vector<string>& filenames, const string& suffix=".pgm", const string& prefix="");

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{  
    if(argc < 4)
    {
        cerr << endl << "Usage: ./mono_yvr path_to_vocabulary path_to_settings path_to_sequence_folder_1 (path_to_image_folder_2 ... path_to_image_folder_N) (trajectory_file_name)" << endl;
        return 1;
    }

    const int num_seq = (argc-3);
    cout << "num_seq = " << num_seq << endl;

    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector<int> nImages;

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        LoadImages(string(argv[(seq)+3]) + "/Camera8", vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, true);

    for (seq = 0; seq<num_seq; seq++)
    {

        // Main loop
        cv::Mat im;
        int proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {

            // Read image from file
            im = cv::imread(vstrImageFilenames[seq][ni],cv::IMREAD_UNCHANGED);
            im = im.colRange(0, im.cols / 2);

            double tframe = vTimestampsCam[seq][ni];

            if(im.empty())
            {
                cerr << endl << "Failed to load image at: "
                     <<  vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the image to the SLAM system
            // cout << "tframe = " << tframe << endl;
            SLAM.TrackMonocular(im,tframe); // TODO change to monocular_inertial

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


    return 0;
}

#include <dirent.h>//for unix, directory entries related definition header

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
