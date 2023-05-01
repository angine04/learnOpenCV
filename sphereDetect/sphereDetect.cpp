#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>


// Load configuration from yaml
YAML::Node cfg = YAML::LoadFile("../config.yaml");
bool fromFile = cfg["fromFile"]["enabled"].as<bool>();
bool saveOriginal = cfg["saveOriginal"]["enabled"].as<bool>();
bool saveResult = cfg["saveResult"]["enabled"].as<bool>();

int main() {

    // Initialize video
    cv::VideoCapture inputVideo;
    if (fromFile) {
        inputVideo = cv::VideoCapture(cfg["fromFile"]["fileName"].as<std::string>());
    } else {
        inputVideo = cv::VideoCapture(0);
    }
    if (!inputVideo.isOpened()) {
        std::cout << "video is off\n\n" << std::endl;
    } else {
        std::cout << "video is on \n\n" << std::endl;
    }

    // Prepare for calibration
    cv::Mat frame, frameCalibration;
    inputVideo >> frame;

    cv::VideoWriter originalOut, resultOut;

    if (saveOriginal) {
        originalOut = cv::VideoWriter(cfg["saveOriginal"]["fileName"].as<std::string>(), cv::VideoWriter::fourcc('X','V','I','D'), 3.0,
                                      cv::Size(frame.cols, frame.rows));
    }
    if (saveResult) {
        resultOut = cv::VideoWriter(cfg["saveResult"]["fileName"].as<std::string>(), cv::VideoWriter::fourcc('X','V','I','D'), 3.0,
                                    cv::Size(frame.cols, frame.rows));
    }

    // Load parameters for calibration
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

    // Calibration
    cv::Mat view, map1, map2;
    cv::Size frameCalibrationSize;
    frameCalibrationSize = frame.size();
    initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix, frameCalibrationSize, CV_16SC2, map1,
                            map2);
    while (true) {

        // Reload configuration every frame, activating dynamic parameter alteration
        cfg = YAML::LoadFile("../config.yaml");

        // Remap calibrated frame
        inputVideo >> frame;
        if (frame.empty()) break;
        remap(frame, frameCalibration, map1, map2, cv::INTER_LINEAR);
        if (saveOriginal) {
            originalOut.write(frame);
        }

        // Prepare for sphere detection
        cv::Mat imgOriginal = frameCalibration;
        cv::Mat imgHSV, imgBGR;
        cv::Mat imgThresholded;
        std::vector<cv::Mat> hsvSplit;

        // Convert the captured frame from BGR to HSV
        cvtColor(imgOriginal, imgHSV, cv::COLOR_BGR2HSV);
        // Split HSV channels
        split(imgHSV, hsvSplit);
        // Enhance contrast of lightness
        equalizeHist(hsvSplit[2], hsvSplit[2]);
        merge(hsvSplit, imgHSV);

        cv::Mat gray;
        // Convert to gray scale for Hough detecting
        cvtColor(imgHSV, gray, cv::COLOR_BGR2GRAY);

        // Denoising
        auto denoisingStrength = cfg["denoisingStrength"].as<float>();
        fastNlMeansDenoising(gray, gray, denoisingStrength);
//        equalizeHist(gray, gray);

        int KernelSize = cfg["kernelSize"].as<int>();
        cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(KernelSize, KernelSize), cv::Point(-1, -1));
        morphologyEx(gray, gray, cv::MORPH_OPEN, kernel);
        morphologyEx(gray, gray, cv::MORPH_CLOSE, kernel);

        namedWindow("hough gray", cv::WINDOW_FREERATIO);
        imshow("hough gray", gray);

        std::vector<cv::Vec3f> circles;
        // Load parameters for Hough circle detection
        auto minDist = cfg["houghCircle"]["minDist"].as<double>();
        auto param1 = cfg["houghCircle"]["param1"].as<double>();
        auto param2 = cfg["houghCircle"]["param2"].as<double>();
        int min_r = cfg["houghCircle"]["minRadius"].as<int>();
        int max_r = cfg["houghCircle"]["maxRadius"].as<int>();
        // Detect circles
        HoughCircles(gray, circles, cv::HOUGH_GRADIENT_ALT, 1.5, minDist, param1, param2, min_r, max_r);

        // For each circle detected
        for (auto &i: circles) {

            // Marking center and contour of the circle
            circle(frameCalibration, cv::Point2f(i[0], i[1]), static_cast<int>(i[2]), cv::Scalar(0, 255, 0), 2, 8);
            circle(frameCalibration, cv::Point2f(i[0], i[1]), 3, cv::Scalar(250, 0, 0), -1, 8);

            // Set real world points for solving PnP problem
            std::vector<cv::Point3f> obj = std::vector<cv::Point3f>{
                    cv::Point3f(-cfg["ballRadius"].as<float>(), 0, 0),
                    cv::Point3f(cfg["ballRadius"].as<float>(), 0, 0),
                    cv::Point3f(0, cfg["ballRadius"].as<float>(), 0),
                    cv::Point3f(0, -cfg["ballRadius"].as<float>(), 0)
            };

            // Collect in-image points for solving PnP problem
            std::vector<cv::Point2f> points = std::vector<cv::Point2f>{
                    cv::Point2f(i[0], i[1] + i[2]),
                    cv::Point2f(i[0], i[1] - i[2]),
                    cv::Point2f(i[0] + i[2], i[1]),
                    cv::Point2f(i[0] - i[2], i[1]),
            };

            // Solve PnP problem
            cv::Mat rVec = cv::Mat::zeros(3, 1, CV_64FC1);
            cv::Mat tVec = cv::Mat::zeros(3, 1, CV_64FC1);
            solvePnP(obj, points, cameraMatrix, distCoeffs, rVec, tVec, false, cv::SOLVEPNP_ITERATIVE);

            // Solve distance
            double distance =
                    sqrt(pow(tVec.at<double>(0), 2) + pow(tVec.at<double>(1), 2) + pow(tVec.at<double>(2), 2)) -
                    cfg["ballRadius"].as<float>();

            // Mark distance on the sphere
            putText(frameCalibration, std::to_string(distance), cv::Point2f(i[0] - i[2], i[1]), 0, 1,
                    cv::Scalar(255, 255, 0), 3);

            // Sample hue value of the sphere's center
            cv::Point2f center(i[0], i[1]);
            unsigned int hue = static_cast<unsigned int>(hsvSplit[0].at<uchar>(center));
            // Distinguish the color
            if (hue >= 120 && hue <= 150) {
                putText(frameCalibration, "purple", cv::Point2f(i[0] - i[2] / 2, i[1] - i[2] / 2), 0, 1,
                        cv::Scalar(255, 255, 255), 3);
            } else if (hue <= 25 && hue >= 7) {
                putText(frameCalibration, "orange", cv::Point2f(i[0] - i[2] / 2, i[1] - i[2] / 2), 0, 1,
                        cv::Scalar(255, 255, 255), 3);
            }
        }

        // Display result
        cv::namedWindow("hough circle", cv::WINDOW_FREERATIO);
        imshow("hough circle", frameCalibration);
        if (saveResult) {
            resultOut.write(frameCalibration);
        }

        // Wait key to quit
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }

    cv::destroyAllWindows();
    resultOut.release();
    originalOut.release();
    return 0;
}