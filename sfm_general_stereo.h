#ifndef SFM_GENERAL_STEREO
#define SFM_GENERAL_STEREO

#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

namespace sfm_general_stereo
{

    /**
     * The objective: normalize the point set in each image by
     * translating the mass point to the origin and
     * the average distance from the mass point to be sqrt(2).
     * 
     * @param points input image
     * @param normalized_points output normalized points
     * @param T transformation matrix from orignal coordinates to the normalized one
     */
    void normalize_points(
        std::vector<cv::Point2d> &points,
        std::vector<cv::Point2d> &normalized_points,
        cv::Mat &T)
    {
        int n = points.size();

        T = cv::Mat::eye(3, 3, CV_64F);
        normalized_points.resize(n);

        // Calculate the mass point
        cv::Point2d mass(0, 0);

        for (auto i = 0; i < n; ++i)
        {
            mass = mass + points[i];
        }

        mass = mass * (1.0 / n);

        // Translate the point clouds to the origin
        for (auto i = 0; i < n; ++i)
        {
            normalized_points[i] = points[i] - mass;
        }

        // Calculate the average distances of the points from the origin
        double avg_distance = 0.0;

        for (auto i = 0; i < n; ++i)
        {
            avg_distance += cv::norm(normalized_points[i]);
        }

        avg_distance /= n;

        const double multiplier =
            sqrt(2) / avg_distance;

        for (auto i = 0; i < n; ++i)
        {
            normalized_points[i] *= multiplier;
        }

        T.at<double>(0, 0) = multiplier;
        T.at<double>(1, 1) = multiplier;
        T.at<double>(0, 2) = -multiplier * mass.x;
        T.at<double>(1, 2) = -multiplier * mass.y;

        // Reason: T1_ * point = (Scaling1 * Translation1) * point = Scaling1 * (Translation1 * point)
    }

    void get_fundamental_matrix_LSQ(
        const std::vector<cv::Point2d> &input_src_points,
        const std::vector<cv::Point2d> &input_dest_points,
        std::vector<int> &selected_samples,
        cv::Mat &fundamental_matrix)
    {
        int n = selected_samples.size();
        // Construct the coefficient matrix (A)
        cv::Mat A(n, 9, CV_64F);

        for (int i = 0; i < n; i++)
        {
            int idx = selected_samples[i];

            const double
                &x1 = input_src_points[idx].x,
                &y1 = input_src_points[idx].y,
                &x2 = input_dest_points[idx].x,
                &y2 = input_dest_points[idx].y;

            A.at<double>(i, 0) = x1 * x2;
            A.at<double>(i, 1) = x2 * y1;
            A.at<double>(i, 2) = x2;
            A.at<double>(i, 3) = y2 * x1;
            A.at<double>(i, 4) = y2 * y1;
            A.at<double>(i, 5) = y2;
            A.at<double>(i, 6) = x1;
            A.at<double>(i, 7) = y1;
            A.at<double>(i, 8) = 1;
        }

        // Solve Ax=0 where x = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
        cv::Mat evals, evecs;
        cv::Mat AtA = A.t() * A;
        cv::eigen(AtA, evals, evecs);

        cv::Mat x = evecs.row(evecs.rows - 1);
        fundamental_matrix.create(3, 3, CV_64F);
        memcpy(fundamental_matrix.data, x.data, sizeof(double) * 9);
    }

    int get_iteration_number(int point_number,
                             int inlier_number,
                             int sample_size,
                             double confidence)
    {
        const double inlier_ratio =
            static_cast<double>(inlier_number) / point_number;

        static const double log1 = log(1.0 - confidence);
        const double log2 = log(1.0 - pow(inlier_ratio, sample_size));

        const int k = log1 / log2;
        if (k < 0)
            return std::numeric_limits<int>::max();
        return k;
    }

    /**
     * Get fundmental matrix using 8 points method and RANSAC roubstification
     */
    void ransac_fundamental_matrix_8points_method(
        const std::vector<cv::Point2d> &input_src_points,
        const std::vector<cv::Point2d> &input_dest_points,
        cv::Mat &best_fundamental_matrix,
        std::vector<int> &best_inliers,
        int ransac_max_iteration,
        double ransac_threshold,
        double ransac_confidence)
    {
        // The number of correspondences
        const int n = input_src_points.size();

        // The size of a minimal sample
        const int sample_size = 8;

        int iteration = 0;
        int maximum_iterations = std::numeric_limits<int>::max(); // The maximum number of iterations set adaptively when a new best model is found
        while (iteration++ < std::min(ransac_max_iteration, maximum_iterations))
        {
            // Initializing the index pool from which the minimal samples are selected
            std::vector<bool> index_pool(n);
            for (int i = 0; i < n; ++i)
                index_pool[i] = true;
            // The minimal sample
            std::vector<int> selected_points_idx;

            // Select 8 random correspondeces
            do
            {
                // Select a random index from the pool
                int idx = rand() % index_pool.size();

                // In case it is not selected before
                if (index_pool[idx])
                {
                    // new index is selected
                    index_pool[idx] = false;
                    selected_points_idx.push_back(idx);
                }

            } while (selected_points_idx.size() != sample_size);

            // Estimate fundamental matrix
            cv::Mat fundamental_matrix(3, 3, CV_64F);
            get_fundamental_matrix_LSQ(
                input_src_points,
                input_dest_points,
                selected_points_idx,
                fundamental_matrix);

            // Count the inliers
            std::vector<int> inliers;
            for (int i = 0; i < input_src_points.size(); ++i)
            {
                // Symmetric epipolar distance
                cv::Mat pt1 = (cv::Mat_<double>(3, 1) << input_src_points[i].x, input_src_points[i].y, 1);
                cv::Mat pt2 = (cv::Mat_<double>(3, 1) << input_dest_points[i].x, input_dest_points[i].y, 1);

                // Calculate the error
                cv::Mat lL = fundamental_matrix.t() * pt2;
                cv::Mat lR = fundamental_matrix * pt1;

                // Calculate the distance of point pt1 from lL
                const double
                    &aL = lL.at<double>(0),
                    &bL = lL.at<double>(1),
                    &cL = lL.at<double>(2);

                double tL = abs(aL * input_src_points[i].x + bL * input_src_points[i].y + cL);
                double dL = sqrt(aL * aL + bL * bL);
                double distanceL = tL / dL;

                // Calculate the distance of point pt2 from lR
                const double
                    &aR = lR.at<double>(0),
                    &bR = lR.at<double>(1),
                    &cR = lR.at<double>(2);

                double tR = abs(aR * input_dest_points[i].x + bR * input_dest_points[i].y + cR);
                double dR = sqrt(aR * aR + bR * bR);
                double distanceR = tR / dR;

                double dist = 0.5 * (distanceL + distanceR);

                if (dist < ransac_threshold)
                    inliers.push_back(i);
            }

            // Update if the new model is better than the previous so-far-the-best.
            if (best_inliers.size() < inliers.size())
            {
                // Update the set of inliers
                best_inliers.clear();
                best_inliers.assign(inliers.begin(), inliers.end());
                // Update fundamental matrix
                fundamental_matrix.copyTo(best_fundamental_matrix);
                // Update the iteration number
                maximum_iterations = get_iteration_number(n,
                                                          best_inliers.size(),
                                                          sample_size,
                                                          ransac_confidence);
            }
        }
    }

    void decompose_essential_matrix(
        const cv::Mat &E,
        cv::Mat &R1,
        cv::Mat &R2,
        cv::Mat &t)
    {
        cv::SVD svd(E, cv::SVD::FULL_UV);
        // It gives matrices U D Vt

        if (cv::determinant(svd.u) < 0)
            svd.u.col(2) *= -1;
        if (cv::determinant(svd.vt) < 0)
            svd.vt.row(2) *= -1;

        cv::Mat w = (cv::Mat_<double>(3, 3) << 0, -1, 0,
                     1, 0, 0,
                     0, 0, 1);

        cv::Mat rotation_1 = svd.u * w * svd.vt;
        cv::Mat rotation_2 = svd.u * w.t() * svd.vt;
        cv::Mat translation = svd.u.col(2) / cv::norm(svd.u.col(2));
        rotation_1.copyTo(R1);
        rotation_2.copyTo(R2);
        translation.copyTo(t);
    }

    void linear_triangulation(
        const cv::Mat &projection_1,
        const cv::Mat &projection_2,
        const cv::Point2d &src_point,
        const cv::Point2d &dst_point,
        cv::Point3d &points3d)
    {
        cv::Mat A(4, 3, CV_64F);
        cv::Mat b(4, 1, CV_64F);

        {
            const double
                &px = src_point.x,
                &py = src_point.y,
                &p1 = projection_1.at<double>(0, 0),
                &p2 = projection_1.at<double>(0, 1),
                &p3 = projection_1.at<double>(0, 2),
                &p4 = projection_1.at<double>(0, 3),
                &p5 = projection_1.at<double>(1, 0),
                &p6 = projection_1.at<double>(1, 1),
                &p7 = projection_1.at<double>(1, 2),
                &p8 = projection_1.at<double>(1, 3),
                &p9 = projection_1.at<double>(2, 0),
                &p10 = projection_1.at<double>(2, 1),
                &p11 = projection_1.at<double>(2, 2),
                &p12 = projection_1.at<double>(2, 3);

            A.at<double>(0, 0) = px * p9 - p1;
            A.at<double>(0, 1) = px * p10 - p2;
            A.at<double>(0, 2) = px * p11 - p3;
            A.at<double>(1, 0) = py * p9 - p5;
            A.at<double>(1, 1) = py * p10 - p6;
            A.at<double>(1, 2) = py * p11 - p7;

            b.at<double>(0) = p4 - px * p12;
            b.at<double>(1) = p8 - py * p12;
        }

        {
            const double
                &px = dst_point.x,
                &py = dst_point.y,
                &p1 = projection_2.at<double>(0, 0),
                &p2 = projection_2.at<double>(0, 1),
                &p3 = projection_2.at<double>(0, 2),
                &p4 = projection_2.at<double>(0, 3),
                &p5 = projection_2.at<double>(1, 0),
                &p6 = projection_2.at<double>(1, 1),
                &p7 = projection_2.at<double>(1, 2),
                &p8 = projection_2.at<double>(1, 3),
                &p9 = projection_2.at<double>(2, 0),
                &p10 = projection_2.at<double>(2, 1),
                &p11 = projection_2.at<double>(2, 2),
                &p12 = projection_2.at<double>(2, 3);

            A.at<double>(2, 0) = px * p9 - p1;
            A.at<double>(2, 1) = px * p10 - p2;
            A.at<double>(2, 2) = px * p11 - p3;
            A.at<double>(3, 0) = py * p9 - p5;
            A.at<double>(3, 1) = py * p10 - p6;
            A.at<double>(3, 2) = py * p11 - p7;

            b.at<double>(2) = p4 - px * p12;
            b.at<double>(3) = p8 - py * p12;
        }

        //cv::Mat x = (A.t() * A).inv() * A.t() * b;
        cv::Mat x = A.inv(cv::DECOMP_SVD) * b;
        points3d = cv::Point3d(x);
    }

    void check_points_infront_of_camera(
        const std::vector<cv::Point2d> &src_points,
        const std::vector<cv::Point2d> &dest_points,
        const std::vector<int> &inliers,
        const cv::Mat &P0,
        const cv::Mat &P1,
        std::vector<cv::Point3d> &traingulated_inliers_points,
        int &infront_camera_cnt,
        double &average_reprojection_error)
    {
        int n = inliers.size();

        infront_camera_cnt = 0;
        average_reprojection_error = 0.;
        for (int i = 0; i < n; i++)
        {
            int idx = inliers[i];

            // Triangulate the point
            cv::Point3d point3d;
            linear_triangulation(P0, P1, src_points[idx], dest_points[idx], point3d);

            // infront of the camera
            if (point3d.z > 0)
                infront_camera_cnt++;
            // Add the triangulated point
            traingulated_inliers_points.push_back(point3d);

            // calculate the reporjection error
            cv::Mat projection1 = P0 * (cv::Mat_<double>(4, 1) << point3d.x, point3d.y, point3d.z, 1.);
            cv::Mat projection2 = P1 * (cv::Mat_<double>(4, 1) << point3d.x, point3d.y, point3d.z, 1.);
            projection1 /= projection1.at<double>(2);
            projection2 /= projection2.at<double>(2);

            // cv::norm(projection1 - src_point_)
            double dx1 = projection1.at<double>(0) - src_points[idx].x;
            double dy1 = projection1.at<double>(1) - src_points[idx].y;
            double squaredDist1 = dx1 * dx1 + dy1 * dy1;

            // cv::norm(projection2 - dst_point_)
            double dx2 = projection2.at<double>(0) - dest_points[idx].x;
            double dy2 = projection2.at<double>(1) - dest_points[idx].y;
            double squaredDist2 = dx2 * dx2 + dy2 * dy2;

            double dst = sqrt(squaredDist1) + sqrt(squaredDist2);
            //std::cout << dst << std::endl;
            average_reprojection_error += dst;
        }

        average_reprojection_error /= n;
    }

    void get_projection_matrix(
        const std::vector<cv::Point2d> &src_points,
        const std::vector<cv::Point2d> &dest_points,
        const std::vector<int> &inliers,
        const cv::Mat &E,            // essential matrix
        const cv::Mat &K1,           // camera 1 calibration matrix
        const cv::Mat &K2,           // camera 2 calibration matrix
        cv::Mat &projection_matrix1, // output camera 1 projection matrix
        cv::Mat &projection_matrix2, // output camera 2 projection matrix
        cv::Mat &R,
        cv::Mat &_t,
        std::vector<cv::Point3d> &tringulated_inliers_points)
    {
        // Decopose essential matrix
        cv::Mat R1, R2, t;
        decompose_essential_matrix(E, R1, R2, t);

        // Consider the first camera is the origin
        cv::Mat P0 = K1 * cv::Mat::eye(3, 4, CV_64F);

        // 4 soultions of second camera projection matrix
        cv::Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
        P1(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
        P1.col(3) = t * 1.0;
        P2(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0;
        P2.col(3) = t * 1.0;
        P3(cv::Range::all(), cv::Range(0, 3)) = R1 * 1.0;
        P3.col(3) = -t * 1.0;
        P4(cv::Range::all(), cv::Range(0, 3)) = R2 * 1.0;
        P4.col(3) = -t * 1.0;
        P1 = K2 * P1;
        P2 = K2 * P2;
        P3 = K2 * P3;
        P4 = K2 * P4;

        //
        std::vector<cv::Point3d> tringulated_points1, tringulated_points2, tringulated_points3, tringulated_points4;
        int cnt1, cnt2, cnt3, cnt4;
        double average_reprojection_error1, average_reprojection_error2, average_reprojection_error3, average_reprojection_error4;
        check_points_infront_of_camera(src_points, dest_points, inliers, P0, P1, tringulated_points1, cnt1, average_reprojection_error1);
        check_points_infront_of_camera(src_points, dest_points, inliers, P0, P2, tringulated_points2, cnt2, average_reprojection_error2);
        check_points_infront_of_camera(src_points, dest_points, inliers, P0, P3, tringulated_points3, cnt3, average_reprojection_error3);
        check_points_infront_of_camera(src_points, dest_points, inliers, P0, P4, tringulated_points4, cnt4, average_reprojection_error4);

        std::cout << "cnt1: " << cnt1 << ", average_reprojection_error1: " << average_reprojection_error1 << std::endl;
        std::cout << "cnt2: " << cnt2 << ", average_reprojection_error2: " << average_reprojection_error2 << std::endl;
        std::cout << "cnt3: " << cnt3 << ", average_reprojection_error3: " << average_reprojection_error3 << std::endl;
        std::cout << "cnt4: " << cnt4 << ", average_reprojection_error4: " << average_reprojection_error4 << std::endl;

        if (cnt1 >= cnt2 && cnt1 >= cnt3 && cnt1 >= cnt4)
        {
            P1.copyTo(projection_matrix2);
            tringulated_inliers_points.assign(tringulated_points1.begin(), tringulated_points1.end());
            R1.copyTo(R);
            t.copyTo(_t);
            _t *= -1.0;
        }
        else if (cnt2 >= cnt1 && cnt2 >= cnt3 && cnt2 >= cnt4)
        {
            P2.copyTo(projection_matrix2);
            tringulated_inliers_points.assign(tringulated_points2.begin(), tringulated_points2.end());
            R2.copyTo(R);
            t.copyTo(_t);
            _t *= -1.0;
        }
        else if (cnt3 >= cnt1 && cnt3 >= cnt2 && cnt3 >= cnt4)
        {
            P3.copyTo(projection_matrix2);
            tringulated_inliers_points.assign(tringulated_points3.begin(), tringulated_points3.end());
            R1.copyTo(R);
            t.copyTo(_t);
        }
        else
        {
            P4.copyTo(projection_matrix2);
            tringulated_inliers_points.assign(tringulated_points4.begin(), tringulated_points4.end());
            R2.copyTo(R);
            t.copyTo(_t);
        }

        P0.copyTo(projection_matrix1);
    }

}

#endif