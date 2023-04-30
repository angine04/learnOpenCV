#include<iostream>
#include<opencv2/opencv.hpp>
#include "yaml-cpp/yaml.h"

int main() {
    YAML::Node cfg = YAML::LoadFile("../config.yaml");
    cv::VideoCapture inputVideo(0);
    if (!inputVideo.isOpened()) {
        std::cout << "video is off\n\n" << std::endl;
    } else {
        std::cout << "video is on \n\n" << std::endl;
    }

    cv::Mat frame, frameCalibration;
    inputVideo >> frame;

    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = cfg["cameraMatrix"][0][0].as<double>();
    cameraMatrix.at<double>(0, 2) = cfg["cameraMatrix"][0][2].as<double>();
    cameraMatrix.at<double>(1, 1) = cfg["cameraMatrix"][1][1].as<double>();
    cameraMatrix.at<double>(1, 2) = cfg["cameraMatrix"][1][2].as<double>();
    cameraMatrix.at<double>(2, 2) = cfg["cameraMatrix"][2][2].as<double>();

    cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = cfg["distCoeffs"][0].as<double>();
    distCoeffs.at<double>(1, 0) = cfg["distCoeffs"][1].as<double>();
    distCoeffs.at<double>(2, 0) = cfg["distCoeffs"][2].as<double>();
    distCoeffs.at<double>(3, 0) = cfg["distCoeffs"][3].as<double>();
    distCoeffs.at<double>(4, 0) = cfg["distCoeffs"][4].as<double>();

    cv::Mat view, rview, map1, map2;
    cv::Size frameCalibration_Size;
    frameCalibration_Size = frame.size();
    initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, frameCalibration_Size, CV_16SC2, map1,
                            map2);
    while (true) {
        cfg = YAML::LoadFile("../config.yaml");
        inputVideo >> frame;
        if (frame.empty()) break;
        remap(frame, frameCalibration, map1, map2, cv::INTER_LINEAR);
        cv::Mat imgOriginal = frameCalibration;
        cv::Mat imgHSV, imgBGR;
        cv::Mat imgThresholded;
        std::vector<cv::Mat> hsvSplit;   //创建向量容器，存放HSV的三通道数据
        cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
        split(imgHSV, hsvSplit);            //分类原图像的HSV三通道
        equalizeHist(hsvSplit[2], hsvSplit[2]);    //对HSV的亮度通道进行直方图均衡
        merge(hsvSplit, imgHSV);                   //合并三种通道v
        cv::Mat gray, binary;
        cvtColor(imgHSV, gray, cv::COLOR_BGR2GRAY);
        equalizeHist(gray, gray);
        auto denoisingStrength = cfg["denoisingStrength"].as<float>();
        fastNlMeansDenoising(gray, gray, denoisingStrength);
        namedWindow("hough gray", cv::WINDOW_FREERATIO);
        imshow("hough gray", gray);
        std::vector<cv::Vec3f> circles;
        auto minDist = cfg["houghCircle"]["minDist"].as<double>();
        auto param1 = cfg["houghCircle"]["param1"].as<double>();
        auto param2 = cfg["houghCircle"]["param2"].as<double>();
        int min_r = cfg["houghCircle"]["minRadius"].as<int>();
        int max_r = cfg["houghCircle"]["maxRadius"].as<int>();
        HoughCircles(gray, circles, cv::HOUGH_GRADIENT_ALT, 1.5, minDist, param1, param2, min_r, max_r);
        for (auto &i: circles) {
            circle(frameCalibration, cv::Point2f(i[0], i[1]), static_cast<int>(i[2]), cv::Scalar(0, 255, 0), 2, 8);
            circle(frameCalibration, cv::Point2f(i[0], i[1]), 3, cv::Scalar(250, 0, 0), -1, 8);
            std::vector<cv::Point3f> obj = std::vector<cv::Point3f>{
                    cv::Point3f(-cfg["ballRadius"].as<float>(), 0, 0),
                    cv::Point3f(cfg["ballRadius"].as<float>(), 0, 0),
                    cv::Point3f(0, cfg["ballRadius"].as<float>(), 0),
                    cv::Point3f(0, -cfg["ballRadius"].as<float>(), 0)
            };
            std::vector<cv::Point2f> points = std::vector<cv::Point2f>{
                    cv::Point2f(i[0], i[1] + i[2]),
                    cv::Point2f(i[0], i[1] - i[2]),
                    cv::Point2f(i[0] + i[2], i[1]),
                    cv::Point2f(i[0] - i[2], i[1]),
            };
            cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1);//init rvec
            cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1);//init tvec
            solvePnP(obj, points, cameraMatrix, distCoeffs, rVec, tVec, false, cv::SOLVEPNP_ITERATIVE);
            double distance =
                    sqrt(pow(tVec.at<double>(0), 2) + pow(tVec.at<double>(1), 2) + pow(tVec.at<double>(2), 2)) -
                    cfg["ballRadius"].as<float>();
            putText(frameCalibration, std::to_string(distance), cv::Point2f(i[0] - i[2], i[1]), 0, 1,
                    cv::Scalar(255, 255, 0), 3);
        }
        namedWindow("hough circle", cv::WINDOW_FREERATIO);
        imshow("hough circle", frameCalibration);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }
    return 0;
}