#include<iostream>
#include <ctime>
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;
float KNOWN_RADIUS = 3;


int main()
{
    VideoCapture inputVideo(0);
    if(!inputVideo.isOpened()){
        std::cout << "video is off\n\n"<<endl;
    }
    else{
        std::cout << "video is on \n\n"<<endl;
    }

    Mat frame, frameCalibration;
    inputVideo >> frame;

    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0,0) = 465.785761871299;
    cameraMatrix.at<double>(0,2) = 325.9609788660892;
    cameraMatrix.at<double>(1,1) = 466.0322492604924;
    cameraMatrix.at<double>(1,2) = 228.3349447234544;
    cameraMatrix.at<double>(2,2) = 1;

    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
    distCoeffs.at<double>(0,0) = -0.003636243725168309;
    distCoeffs.at<double>(1,0) =  0.0009571786468545265;
    distCoeffs.at<double>(2,0) =  -0.006906820389750674;
    distCoeffs.at<double>(3,0) = -0.00745741077751897;
    distCoeffs.at<double>(4,0) = 0.2082387251633841;

    Mat view, rview, map1, map2;
    Size frameCalibration_Size;
    frameCalibration_Size = frame.size();
    initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), cameraMatrix, frameCalibration_Size, CV_16SC2, map1, map2);
    while(true){
        inputVideo >> frame;
        if(frame.empty()) break;
        remap(frame, frameCalibration, map1, map2, INTER_LINEAR);
        cv::Mat imgOriginal = frameCalibration;
        Mat imgHSV, imgBGR;
        Mat imgThresholded;
        vector <Mat> hsvSplit;   //创建向量容器，存放HSV的三通道数据
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
            split(imgHSV, hsvSplit);            //分类原图像的HSV三通道
            equalizeHist(hsvSplit[2], hsvSplit[2]);    //对HSV的亮度通道进行直方图均衡
            merge(hsvSplit, imgHSV);                   //合并三种通道v
        Mat gray, binary;
        cvtColor(imgHSV, gray, COLOR_BGR2GRAY);
        equalizeHist(gray,gray);
        fastNlMeansDenoising(gray,gray,20);
        namedWindow("hough gray", WINDOW_FREERATIO);
        imshow("hough gray", gray);
        vector<Vec3f> circles;
        double mindist = 30;
        int min_r = 40;
        int max_r = 200;
        HoughCircles(gray, circles, HOUGH_GRADIENT_ALT, 1.5, mindist, 92, 0.88, min_r, max_r);
        for (auto & i : circles) {
            circle(frameCalibration, Point(i[0], i[1]), i[2], Scalar(0, 255, 0), 3, 8);
            circle(frameCalibration, Point(i[0], i[1]), 10, Scalar(250, 0, 0), -1, 8);
            vector<Point3f> obj = vector<Point3f>{
                    cv::Point3f(-KNOWN_RADIUS, 0, 0),
                    cv::Point3f(KNOWN_RADIUS, 0, 0),
                    cv::Point3f(0, KNOWN_RADIUS, 0),
                    cv::Point3f(0, -KNOWN_RADIUS, 0)
            };
            vector<Point2f> points = vector<Point2f>{
                    cv::Point2f(i[0],i[1]+i[2]),
                    cv::Point2f(i[0],i[1]-i[2]),
                    cv::Point2f(i[0]+i[2],i[1]),
                    cv::Point2f(i[0]-i[2],i[1]),
            };
            cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1);//init rvec 
            cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1);//init tvec
            solvePnP(obj, points, cameraMatrix, distCoeffs, rVec, tVec, false, SOLVEPNP_ITERATIVE);
            double distance = sqrt(pow(tVec.at<double>(0),2)+pow(tVec.at<double>(1),2)+pow(tVec.at<double>(2),2));
            putText(frameCalibration,to_string(distance),Point(i[0]-i[2],i[1]),0,1,Scalar(255,255,0),3);
        }
        namedWindow("hough circle", WINDOW_FREERATIO);
        imshow("hough circle", frameCalibration);
        int key = waitKey(1);
        if(key == 27 || key == 'q' || key == 'Q') break;
    }
    return 0;
}