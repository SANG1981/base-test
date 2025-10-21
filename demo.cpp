#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

class VINSDemo
{
private:
    // 相机内参（示例参数）
    Mat cameraMatrix;
    Mat distCoeffs;

    // 特征检测器
    Ptr<Feature2D> orb;

    // 存储图像和特征
    vector<Mat> images;
    vector<vector<KeyPoint>> allKeypoints;
    vector<Mat> allDescriptors;
    vector<vector<Point2f>> allPoints;

public:
    VINSDemo()
    {
        // 初始化相机参数（假设的相机参数）
        cameraMatrix = (Mat_<double>(3, 3) << 800.0, 0.0, 320.0,
                        0.0, 800.0, 240.0,
                        0.0, 0.0, 1.0);
        distCoeffs = Mat::zeros(4, 1, CV_64F);

        // 初始化ORB特征检测器
        orb = ORB::create(500);
    }

    // 加载图像
    bool loadImages(const vector<string> &imagePaths)
    {
        images.clear();
        for (const auto &path : imagePaths)
        {
            Mat img = imread(path);
            if (img.empty())
            {
                cout << "无法加载图像: " << path << endl;
                return false;
            }
            images.push_back(img);
            cout << "加载图像: " << path << " 尺寸: " << img.size() << endl;
        }
        return images.size() == 3;
    }

    // 图像预处理和特征提取
    void extractFeatures()
    {
        allKeypoints.clear();
        allDescriptors.clear();
        allPoints.clear();

        cout << "\n=== 开始特征提取 ===" << endl;

        for (size_t i = 0; i < images.size(); i++)
        {
            Mat gray;
            cvtColor(images[i], gray, COLOR_BGR2GRAY);

            // 畸变校正
            Mat undistorted;
            undistort(gray, undistorted, cameraMatrix, distCoeffs);

            vector<KeyPoint> keypoints;
            Mat descriptors;

            // 特征检测和描述子计算
            orb->detectAndCompute(undistorted, noArray(), keypoints, descriptors);

            // 转换为Point2f格式用于光流跟踪
            vector<Point2f> points;
            KeyPoint::convert(keypoints, points);

            allKeypoints.push_back(keypoints);
            allDescriptors.push_back(descriptors);
            allPoints.push_back(points);

            cout << "图像 " << i + 1 << " 提取到 " << keypoints.size() << " 个特征点" << endl;
        }
    }

    // 光流跟踪
    void opticalFlowTracking()
    {
        cout << "\n=== 开始光流跟踪 ===" << endl;

        for (size_t i = 0; i < images.size() - 1; i++)
        {
            vector<Point2f> prevPoints = allPoints[i];
            vector<Point2f> nextPoints;
            vector<uchar> status;
            vector<float> err;

            // 金字塔LK光流跟踪
            calcOpticalFlowPyrLK(
                images[i], images[i + 1],
                prevPoints, nextPoints,
                status, err,
                Size(21, 21), 3,
                TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01));

            // 过滤掉跟踪失败的点
            vector<Point2f> trackedPrev, trackedNext;
            for (size_t j = 0; j < status.size(); j++)
            {
                if (status[j] && err[j] < 20.0f)
                {
                    trackedPrev.push_back(prevPoints[j]);
                    trackedNext.push_back(nextPoints[j]);
                }
            }

            cout << "图像 " << i + 1 << " -> " << i + 2
                 << ": 成功跟踪 " << trackedPrev.size() << "/" << prevPoints.size() << " 个点" << endl;

            // 可视化光流跟踪结果
            visualizeOpticalFlow(i, trackedPrev, trackedNext);
        }
    }

    // 对极几何和位姿估计
    void estimatePose()
    {
        cout << "\n=== 开始位姿估计 ===" << endl;

        for (size_t i = 0; i < images.size() - 1; i++)
        {
            // 使用特征匹配找到对应点
            vector<DMatch> matches;
            BFMatcher matcher(NORM_HAMMING);
            matcher.match(allDescriptors[i], allDescriptors[i + 1], matches);

            // 筛选好的匹配
            vector<DMatch> goodMatches;
            for (const auto &match : matches)
            {
                if (match.distance < 50)
                {
                    goodMatches.push_back(match);
                }
            }

            // 获取匹配的点对
            vector<Point2f> pts1, pts2;
            for (const auto &match : goodMatches)
            {
                pts1.push_back(allPoints[i][match.queryIdx]);
                pts2.push_back(allPoints[i + 1][match.trainIdx]);
            }

            if (pts1.size() < 8)
            {
                cout << "图像 " << i + 1 << " -> " << i + 2 << ": 匹配点太少，跳过" << endl;
                continue;
            }

            // 使用RANSAC计算基础矩阵
            Mat fundamentalMatrix = findFundamentalMat(pts1, pts2, FM_RANSAC, 3.0, 0.99);

            // 从基础矩阵恢复相对位姿
            Mat essentialMatrix = cameraMatrix.t() * fundamentalMatrix * cameraMatrix;
            Mat R, t;
            recoverPose(essentialMatrix, pts1, pts2, cameraMatrix, R, t);

            cout << "图像 " << i + 1 << " -> " << i + 2 << " 的相对位姿:" << endl;
            cout << "旋转矩阵 R: " << endl
                 << R << endl;
            cout << "平移向量 t: " << t.t() << endl;

            // 可视化匹配结果
            visualizeMatches(i, goodMatches);
        }
    }

    // 三角测量
    void triangulatePoints()
    {
        cout << "\n=== 开始三角测量 ===" << endl;

        if (images.size() < 2)
            return;

        // 假设第一张图像在原点，第二张图像有相对运动
        Mat R1 = Mat::eye(3, 3, CV_64F);
        Mat t1 = Mat::zeros(3, 1, CV_64F);

        Mat R2 = (Mat_<double>(3, 3) << 0.98, 0.05, 0.1,
                  -0.05, 0.99, -0.05,
                  -0.1, 0.04, 0.99);
        Mat t2 = (Mat_<double>(3, 1) << 0.1, 0.01, 0.05);

        // 构建投影矩阵
        Mat P1 = cameraMatrix * Mat::eye(3, 4, CV_64F);
        Mat P2 = cameraMatrix * buildRTMatrix(R2, t2);

        // 使用特征匹配找到对应点
        vector<DMatch> matches;
        BFMatcher matcher(NORM_HAMMING);
        matcher.match(allDescriptors[0], allDescriptors[1], matches);

        vector<Point2f> pts1, pts2;
        for (const auto &match : matches)
        {
            if (match.distance < 30)
            {
                pts1.push_back(allPoints[0][match.queryIdx]);
                pts2.push_back(allPoints[1][match.trainIdx]);
            }
        }

        if (pts1.size() < 8)
        {
            cout << "匹配点太少，无法进行三角测量" << endl;
            return;
        }

        // 三角测量
        Mat points4D;
        cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

        // 转换为3D坐标
        vector<Point3f> points3D;
        for (int i = 0; i < points4D.cols; i++)
        {
            Mat x = points4D.col(i);
            x /= x.at<float>(3);
            points3D.push_back(Point3f(x.at<float>(0), x.at<float>(1), x.at<float>(2)));
        }

        cout << "三角测量生成 " << points3D.size() << " 个3D点" << endl;

        // 可视化3D点（简单的2D投影）
        visualize3DPoints(points3D);

        // 使用三角测量得到的3D点进行solvePnP演示
        demonstrateSolvePnP(points3D, pts1, pts2);
    }

    // 演示solvePnP功能
    void demonstrateSolvePnP(const vector<Point3f> &objectPoints,
                             const vector<Point2f> &imagePoints1,
                             const vector<Point2f> &imagePoints2)
    {
        cout << "\n=== 开始 solvePnP 演示 ===" << endl;

        if (objectPoints.size() < 4 || imagePoints1.size() != objectPoints.size())
        {
            cout << "3D-2D点对数量不足，无法进行solvePnP" << endl;
            return;
        }

        // 使用第一帧图像的2D点和对应的3D点进行PnP求解
        Mat rvec1, tvec1;
        bool success1 = solvePnP(objectPoints, imagePoints1, cameraMatrix, distCoeffs,
                                 rvec1, tvec1, false, SOLVEPNP_ITERATIVE);

        if (success1)
        {
            cout << "第一帧图像的solvePnP结果:" << endl;
            cout << "旋转向量 rvec: " << rvec1.t() << endl;
            cout << "平移向量 tvec: " << tvec1.t() << endl;

            // 将旋转向量转换为旋转矩阵
            Mat R1;
            Rodrigues(rvec1, R1);
            cout << "旋转矩阵 R: " << endl
                 << R1 << endl;

            // 验证重投影误差
            double error1 = calculateReprojectionError(objectPoints, imagePoints1, rvec1, tvec1);
            cout << "第一帧重投影误差: " << error1 << " 像素" << endl;
        }

        // 使用第二帧图像的2D点和对应的3D点进行PnP求解
        if (imagePoints2.size() == objectPoints.size())
        {
            Mat rvec2, tvec2;
            bool success2 = solvePnP(objectPoints, imagePoints2, cameraMatrix, distCoeffs,
                                     rvec2, tvec2, false, SOLVEPNP_EPNP);

            if (success2)
            {
                cout << "\n第二帧图像的solvePnP结果:" << endl;
                cout << "旋转向量 rvec: " << rvec2.t() << endl;
                cout << "平移向量 tvec: " << tvec2.t() << endl;

                // 将旋转向量转换为旋转矩阵
                Mat R2;
                Rodrigues(rvec2, R2);
                cout << "旋转矩阵 R: " << endl
                     << R2 << endl;

                // 验证重投影误差
                double error2 = calculateReprojectionError(objectPoints, imagePoints2, rvec2, tvec2);
                cout << "第二帧重投影误差: " << error2 << " 像素" << endl;
            }
        }

        // 使用RANSAC的solvePnP进行鲁棒估计
        Mat rvec_ransac, tvec_ransac;
        vector<int> inliers;
        bool success_ransac = solvePnPRansac(objectPoints, imagePoints1, cameraMatrix, distCoeffs,
                                             rvec_ransac, tvec_ransac, false, 100, 8.0, 0.99, inliers);

        if (success_ransac)
        {
            cout << "\n使用RANSAC的solvePnP结果:" << endl;
            cout << "内点数量: " << inliers.size() << "/" << objectPoints.size() << endl;
            cout << "旋转向量: " << rvec_ransac.t() << endl;
            cout << "平移向量: " << tvec_ransac.t() << endl;

            double error_ransac = calculateReprojectionError(objectPoints, imagePoints1, rvec_ransac, tvec_ransac);
            cout << "RANSAC重投影误差: " << error_ransac << " 像素" << endl;

            // 可视化PnP结果
            visualizePnPResult(objectPoints, imagePoints1, rvec_ransac, tvec_ransac, inliers);
        }
    }

    // 计算重投影误差
    double calculateReprojectionError(const vector<Point3f> &objectPoints,
                                      const vector<Point2f> &imagePoints,
                                      const Mat &rvec, const Mat &tvec)
    {
        vector<Point2f> projectedPoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

        double totalError = 0.0;
        for (size_t i = 0; i < imagePoints.size(); i++)
        {
            double error = norm(imagePoints[i] - projectedPoints[i]);
            totalError += error * error;
        }
        return sqrt(totalError / imagePoints.size());
    }

    // 可视化光流跟踪结果
    void visualizeOpticalFlow(int frameIdx, const vector<Point2f> &prevPts, const vector<Point2f> &nextPts)
    {
        Mat visImage = images[frameIdx + 1].clone();

        // 绘制跟踪的点
        for (size_t i = 0; i < nextPts.size(); i++)
        {
            circle(visImage, nextPts[i], 3, Scalar(0, 255, 0), -1);
            // 绘制运动轨迹
            if (i < prevPts.size())
            {
                arrowedLine(visImage, prevPts[i], nextPts[i], Scalar(0, 0, 255), 2);
            }
        }

        putText(visImage, "Optical Flow Tracking: Frame " + to_string(frameIdx + 1) + "->" + to_string(frameIdx + 2),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        string windowName = "Optical Flow " + to_string(frameIdx + 1);
        namedWindow(windowName, WINDOW_AUTOSIZE);
        imshow(windowName, visImage);
    }

    // 可视化特征匹配
    void visualizeMatches(int frameIdx, const vector<DMatch> &matches)
    {
        Mat matchImage;
        drawMatches(images[frameIdx], allKeypoints[frameIdx],
                    images[frameIdx + 1], allKeypoints[frameIdx + 1],
                    matches, matchImage,
                    Scalar::all(-1), Scalar::all(-1),
                    vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        putText(matchImage, "Feature Matching: Frame " + to_string(frameIdx + 1) + "->" + to_string(frameIdx + 2),
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        string windowName = "Feature Matching " + to_string(frameIdx + 1);
        namedWindow(windowName, WINDOW_AUTOSIZE);
        imshow(windowName, matchImage);
    }

    // 可视化3D点（简单的2D投影）
    void visualize3DPoints(const vector<Point3f> &points3D)
    {
        if (points3D.empty())
            return;

        Mat visImage = images[0].clone();

        // 简单地将3D点投影到图像平面（假设在原点观察）
        for (const auto &pt : points3D)
        {
            // 简单的透视投影
            Point2f imgPt(pt.x / pt.z * 800 + 320, pt.y / pt.z * 800 + 240);
            if (imgPt.x >= 0 && imgPt.x < visImage.cols && imgPt.y >= 0 && imgPt.y < visImage.rows)
            {
                circle(visImage, imgPt, 2, Scalar(0, 255, 255), -1);
            }
        }

        putText(visImage, "3D Points Projection (" + to_string(points3D.size()) + " points)",
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);

        namedWindow("3D Points", WINDOW_AUTOSIZE);
        imshow("3D Points", visImage);
    }

    // 可视化PnP结果
    void visualizePnPResult(const vector<Point3f> &objectPoints,
                            const vector<Point2f> &imagePoints,
                            const Mat &rvec, const Mat &tvec,
                            const vector<int> &inliers)
    {
        Mat visImage = images[0].clone();

        // 投影3D点
        vector<Point2f> projectedPoints;
        projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, projectedPoints);

        // 绘制原始点（红色）和重投影点（绿色）
        for (size_t i = 0; i < imagePoints.size(); i++)
        {
            // 检查是否为内点
            bool isInlier = false;
            for (int inlier : inliers)
            {
                if (static_cast<int>(i) == inlier)
                {
                    isInlier = true;
                    break;
                }
            }

            Scalar originalColor = isInlier ? Scalar(0, 0, 255) : Scalar(0, 0, 100);  // 红色
            Scalar projectedColor = isInlier ? Scalar(0, 255, 0) : Scalar(0, 100, 0); // 绿色

            // 绘制原始检测点
            circle(visImage, imagePoints[i], 4, originalColor, -1);
            // 绘制重投影点
            circle(visImage, projectedPoints[i], 3, projectedColor, -1);
            // 连接原始点和重投影点
            if (isInlier)
            {
                line(visImage, imagePoints[i], projectedPoints[i], Scalar(255, 255, 0), 1);
            }
        }

        putText(visImage, "solvePnP Result (Red: Original, Green: Reprojected)",
                Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        putText(visImage, "Inliers: " + to_string(inliers.size()) + "/" + to_string(objectPoints.size()),
                Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        namedWindow("solvePnP Result", WINDOW_AUTOSIZE);
        imshow("solvePnP Result", visImage);
    }

private:
    // 构建RT矩阵
    Mat buildRTMatrix(const Mat &R, const Mat &t)
    {
        Mat RT = Mat::eye(4, 4, CV_64F);
        R.copyTo(RT(Rect(0, 0, 3, 3)));
        t.copyTo(RT(Rect(3, 0, 1, 3)));
        return RT(Rect(0, 0, 4, 3));
    }
};

int main()
{
    cout << "VINS-Fusion OpenCV 3.4.16 Demo with solvePnP" << endl;
    cout << "=============================================" << endl;

    // 创建演示对象
    VINSDemo demo;

    // 加载测试图像（请替换为实际的图像路径）
    vector<string> imagePaths = {
        "image1.jpg", // 替换为你的第一张图像路径
        "image2.jpg", // 替换为你的第二张图像路径
        "image3.jpg"  // 替换为你的第三张图像路径
    };

    // 如果找不到图像，使用默认的示例
    if (!demo.loadImages(imagePaths))
    {
        cout << "无法加载指定图像，创建示例图像..." << endl;

        // 创建示例图像用于演示
        vector<Mat> exampleImages(3);
        for (int i = 0; i < 3; i++)
        {
            exampleImages[i] = Mat::zeros(480, 640, CV_8UC3);
            // 在不同位置绘制不同的形状来模拟相机运动
            rectangle(exampleImages[i], Point(100 + i * 50, 100 + i * 30),
                      Point(300 + i * 50, 300 + i * 30), Scalar(0, 255, 0), -1);
            circle(exampleImages[i], Point(200 + i * 20, 150 - i * 10), 40, Scalar(255, 0, 0), -1);
            putText(exampleImages[i], "Frame " + to_string(i + 1), Point(50, 400),
                    FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
        }

        // 保存示例图像
        for (int i = 0; i < 3; i++)
        {
            string filename = "example" + to_string(i + 1) + ".jpg";
            imwrite(filename, exampleImages[i]);
            imagePaths[i] = filename;
        }

        // 重新加载图像
        demo.loadImages(imagePaths);
    }

    // 执行完整的处理流程
    demo.extractFeatures();
    demo.opticalFlowTracking();
    demo.estimatePose();
    demo.triangulatePoints(); // 这个函数内部会调用solvePnP演示

    cout << "\n=== 演示完成 ===" << endl;
    cout << "按任意键退出..." << endl;

    waitKey(0);
    destroyAllWindows();

    return 0;
}