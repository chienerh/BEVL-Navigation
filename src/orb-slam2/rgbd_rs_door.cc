/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <python3.6/Python.h>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "opencv/cv-helpers.hpp"
#include <numpy/arrayobject.h>

#include <System.h>

using namespace std;
using namespace rs2;
using namespace cv;

int main(int argc, char **argv)
{
    string vocabPath = "Vocabulary/ORBvoc.txt";
	string settingsPath = "Examples/RGB-D/RealSense.yaml";
	if (argc == 1)
	{
		cout << "Default vocabPath: " << vocabPath << endl << "Default settingsPath: " << settingsPath << endl;
	}
	else if (argc == 2)
	{
		vocabPath = argv[1];
	}
	else if (argc == 3)
	{
		vocabPath = argv[1];
		settingsPath = argv[2];
	}
    else
    {
        cerr << endl << "Usage: mono_webcam.exe path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

    // Set up RealSense streaming 
    cout << "set up RealSense streaming" << endl;
    config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, -1, 640, 480, rs2_format::RS2_FORMAT_RGB8, 0);
    cfg.enable_stream(RS2_STREAM_DEPTH, -1, 640, 480, rs2_format::RS2_FORMAT_Z16, 0);
    pipeline pipe;
    auto config = pipe.start(cfg);
    auto profile = config.get_stream(RS2_STREAM_COLOR).as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    // Set up Embedded Python script 
    cout << "set up python script" << endl;
    PyObject *pName = nullptr, *pModule = nullptr, *pFunc = nullptr;
    PyObject *pImRGB = nullptr, *pImD = nullptr, *pReturn = nullptr;

    wchar_t *wcsProgram = Py_DecodeLocale("rgbd_rs_door", NULL);
    Py_SetProgramName(wcsProgram);
    Py_Initialize();

    pName = PyUnicode_DecodeFSDefault("detectntrack"); //for python3.6

    if ((pModule = PyImport_Import(pName)) == NULL) {
        printf("Error: PyImport_Import\n");
    }
    if ((pFunc = PyObject_GetAttrString(pModule, "detectntrack"))==NULL) {
        printf("Error: PyObject_GetAttrString\n");
    }

    const int ND = 2;
    npy_intp dimsRGB[2]{ 480, 640 * 3 };
    npy_intp dimsD[2]{ 480, 640 };

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    cout << "set up ORB-SLAM2 system" << endl;
    ORB_SLAM2::System SLAM(vocabPath, settingsPath, ORB_SLAM2::System::RGBD, true);

    vector<float> vTimesTrack;
    static unsigned int last_frame_number = 0;
    static unsigned int last_depth_frame_number = 0;
    static unsigned int no_depth = 0;

    // Pass image to python script
    if (pFunc && PyCallable_Check(pFunc)) {} else
    {
        cout << "Detect and tracking functions are not callable !" << endl;
    }

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    cv::Mat imRGB;
    cv::Mat imD;
    cv::Mat imD_mm;

    // Main loop
    while(true)
    {

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame, but the color did not update, continue
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();
        if (depth_frame.get_frame_number() == last_depth_frame_number){
            no_depth = 1;
        }
        else{
            no_depth = 0;
        }
        last_depth_frame_number = depth_frame.get_frame_number();

        // Convert RealSense frame to OpenCV matrix:
        import_array();
		imRGB = frame_to_mat(color_frame);
        pImRGB = PyArray_SimpleNewFromData(ND, dimsRGB, NPY_UBYTE, reinterpret_cast<void*>(imRGB.data));
        if (!pImRGB) {
            cerr << "PyArray_SimpleNewFromData failed." << endl;
            return -1;
        }

        if (no_depth){
            // Call python function to detect and track
            cerr << "No depth_frame" << endl;
            pReturn = PyObject_CallFunctionObjArgs(pFunc, pImRGB, NULL);
            
        }
        else{
            imD_mm = depth_frame_to_mm(pipe, depth_frame); // input to ORB-SLAM
            imD = depth_frame_to_meters(pipe, depth_frame); // input to python script

            cv::Scalar tempVal = mean( imD );
            float myMAtMean = tempVal.val[0];
            pImD = PyArray_SimpleNewFromData(ND, dimsD, NPY_DOUBLE, reinterpret_cast<void*>(imD.data)); // NPY_INT8 NPY_UBYTE
            if (!pImD) {
                cerr << "PyArray_SimpleNewFromData failed." << endl;
                return -1;
            }
            // Call python function to detect and track
            pReturn = PyObject_CallFunctionObjArgs(pFunc, pImRGB, pImD, NULL);
            double tframe = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
            SLAM.TrackRGBD(imRGB, imD_mm, tframe);
            imshow("imD", imD);
            int key = waitKey(1);

            if (key >= 0)
                break;
        }

        
        
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack.push_back(ttrack);
        
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(unsigned int ni=0; ni<vTimesTrack.size(); ni++)
    {
        totaltime+=vTimesTrack[ni];
    }

    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[vTimesTrack.size()/2] << endl;
    cout << "mean tracking time: " << totaltime/vTimesTrack.size() << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    // Clean Up
    Py_XDECREF(pFunc);
    Py_DECREF (pName);
    Py_DECREF (pModule);
    Py_DECREF (pReturn);
    Py_DECREF (pImRGB);
    Py_DECREF (pImD);
    // Finish the Python Interpreter
    Py_Finalize ();    

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
