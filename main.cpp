#include <stdio.h>
#include <iostream>
#include <fstream>
#include <ctime>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>

#include "sfm_planar_motion.h"

cv::Mat read_matrix_from_file(std::string file_name, int rows, int cols)
{
    std::ifstream file_stream(file_name);

    cv::Mat K = cv::Mat::zeros(rows, cols, CV_64F);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double e;
            file_stream >> e;
            K.at<double>(i, j) = e;
        }
    }

    file_stream.close();

    return K;
}

cv::Mat read_matrix_from_file(std::string file_name, int cols)
{
    std::ifstream file_stream(file_name);

    std::vector<double> v;
    while (!file_stream.eof())
    {
        double e;
        file_stream >> e;
        v.push_back(e);
    }

    file_stream.close();

    int rows = v.size() / cols;

    cv::Mat K(rows, cols, CV_64F);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            K.at<double>(i, j) = v[i * cols + j];

    return K;
}

int main(int argc, char **argv)
{
    // Setting the random seed to get random results in each run.
    srand(time(NULL));

    const std::string keys =
        "{ @src_points_name |  |  }"
        "{ @dest_points_name |  |  }"
        "{ @calibration_matrix_name |  |  }";

    cv::CommandLineParser parser(argc, argv, keys);

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    std::string src_points_name = parser.get<std::string>("@src_points_name");
    std::string dest_points_name = parser.get<std::string>("@dest_points_name");
    std::string calibration_matrix_name = parser.get<std::string>("@calibration_matrix_name");

    cv::Mat src_points_mat = read_matrix_from_file(src_points_name, 2);
    cv::Mat dest_points_mat = read_matrix_from_file(dest_points_name, 2);

    // Calibration matrix
    cv::Mat K = read_matrix_from_file(calibration_matrix_name, 3, 3);

    std::vector<cv::Point2d> src_points, dest_points;
    for (int i = 0; i < src_points_mat.rows; i++)
        src_points.push_back(cv::Point2d(src_points_mat.row(i)));
    for (int i = 0; i < dest_points_mat.rows; i++)
        dest_points.push_back(cv::Point2d(dest_points_mat.row(i)));

    cv::Mat Kinv = K.inv();
    cv::Mat src_points_mat_normalized(3, src_points_mat.rows, CV_64F);
    src_points_mat_normalized.rowRange(0, 2) = 1.0 * src_points_mat.t();
    src_points_mat_normalized.row(2) = 1.0 * cv::Mat::ones(1, src_points_mat_normalized.cols, CV_64F);
    src_points_mat_normalized = Kinv * src_points_mat_normalized;
    src_points_mat_normalized = src_points_mat_normalized.t();
    src_points_mat_normalized.col(0) /= src_points_mat_normalized.col(2);
    src_points_mat_normalized.col(1) /= src_points_mat_normalized.col(2);
    src_points_mat_normalized = src_points_mat_normalized.colRange(0, 2);

    cv::Mat dest_points_mat_normalized(3, dest_points_mat.rows, CV_64F);
    dest_points_mat_normalized.rowRange(0, 2) = 1.0 * dest_points_mat.t();
    dest_points_mat_normalized.row(2) = 1.0 * cv::Mat::ones(1, dest_points_mat_normalized.cols, CV_64F);
    dest_points_mat_normalized = Kinv * dest_points_mat_normalized;
    dest_points_mat_normalized = dest_points_mat_normalized.t();
    dest_points_mat_normalized.col(0) /= dest_points_mat_normalized.col(2);
    dest_points_mat_normalized.col(1) /= dest_points_mat_normalized.col(2);
    dest_points_mat_normalized = dest_points_mat_normalized.colRange(0, 2);

    std::vector<cv::Point2d> src_points_normalized, dest_points_normalized;
    for (int i = 0; i < src_points_mat_normalized.rows; i++)
        src_points_normalized.push_back(cv::Point2d(src_points_mat_normalized.row(i)));
    for (int i = 0; i < dest_points_mat_normalized.rows; i++)
        dest_points_normalized.push_back(cv::Point2d(dest_points_mat_normalized.row(i)));

    // Since we use the normalized coordinates, we have to normalize the threshold
    // We scale the threshold with the average focal lengths
    double threshold = 2.;
    threshold = threshold / ((K.at<double>(0, 0) + K.at<double>(1, 1)) / 2.0);

    std::vector<int> inliers;
    std::vector<cv::Mat> Rs, ts;
    sfm_planar_motion_ellipse::ransac_2points(
        src_points_normalized,
        dest_points_normalized,
        100, // max iteration
        threshold,
        inliers,
        Rs,
        ts);

    std::cout << "inliers count: " << inliers.size() << std::endl;
    /*std::cout << "Rs: " << Rs[0] << std::endl;
    std::cout << "ts: " << ts[0] << std::endl;
    std::cout << "Rs: " << Rs[1] << std::endl;
    std::cout << "ts: " << ts[1] << std::endl;*/

    cv::Mat R, t;
    sfm_planar_motion_ellipse::get_correct_rotation_translation(
        src_points,
        dest_points,
        Rs,
        ts,
        K,
        inliers,
        R,
        t);

    std::cout << "R:\n"
              << R << std::endl;
    std::cout << "t:\n"
              << t << std::endl;
    std::cout << "angle: " << atan2(R.at<double>(0, 2), R.at<double>(0, 0)) << std::endl;

    return 0;
}