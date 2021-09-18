#ifndef SFM_PLANAR_MOTION
#define SFM_PLANAR_MOTION

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "sfm_general_stereo.h"

namespace sfm_planar_motion_ellipse
{

#define EPS 1e-6

    void get_estimated_translation_rotation(
        const std::vector<cv::Point2d> &src_points,
        const std::vector<cv::Point2d> &dest_points,
        const std::vector<int> &indices,
        std::vector<cv::Mat> &Rs,
        std::vector<cv::Mat> &ts)
    {
        // a = [cos(\phi) sin(\phi)]^T
        // b = [cos(\theta - \phi) sin(\theta - \phi)]^T

        cv::Mat A(indices.size(), 2, CV_64F), B(indices.size(), 2, CV_64F);
        for (int i = 0; i < indices.size(); i++)
        {
            int idx = indices[i];

            A.at<double>(i, 0) = src_points[idx].x * dest_points[idx].y;
            A.at<double>(i, 1) = -dest_points[idx].y;

            B.at<double>(i, 0) = dest_points[idx].x * src_points[idx].y;
            B.at<double>(i, 1) = src_points[idx].y;
        }

        cv::Mat Binv;
        cv::invert(B, Binv, cv::DECOMP_SVD);

        cv::Mat C = Binv * A;

        /*std::cout << "A: " << A << std::endl;
        std::cout << "B: " << B << std::endl;
        std::cout << "Binv: " << Binv << std::endl;
        std::cout << "C: " << C << std::endl;*/

        cv::Mat S, U, Vt;
        cv::SVD::compute(C.t() * C, S, U, Vt);

        /*std::cout << "S: " << S << std::endl;
        std::cout << "U: " << U << std::endl;
        std::cout << "Vt: " << Vt << std::endl;*/

        double s1 = S.at<double>(0);
        double s2 = S.at<double>(1);

        if (abs(s1 - s2) < EPS)
            return;

        std::vector<cv::Mat> ys;
        if (s1 < 1.)
        { // degenerate case
            ys.push_back(1. * (cv::Mat_<double>(2, 1) << 1., 0.));
            ys.push_back(1. * (cv::Mat_<double>(2, 1) << -1., 0.));
        }
        else if (s2 > 1.)
        { // degenerate case
            ys.push_back(1. * (cv::Mat_<double>(2, 1) << 0., 1.));
            ys.push_back(1. * (cv::Mat_<double>(2, 1) << 0., -1.));
        }
        else
        {
            double y_1 = sqrt((1. - s2) / (s1 - s2));
            double y_2 = sqrt((s1 - 1.) / (s1 - s2));

            ys.push_back(1. * (cv::Mat_<double>(2, 1) << y_1, y_2));
            ys.push_back(1. * (cv::Mat_<double>(2, 1) << -y_1, -y_2));
            ys.push_back(1. * (cv::Mat_<double>(2, 1) << y_1, -y_2));
            ys.push_back(1. * (cv::Mat_<double>(2, 1) << -y_1, y_2));
        }

        std::vector<cv::Mat> as;
        for (auto y : ys)
        {
            as.push_back(U * y);
        }

        std::vector<cv::Mat> bs;
        for (auto a : as)
        {
            bs.push_back(C * a);
        }

        for (auto b : bs)
        {
            // t = [sin(\theta - \phi) 0 -cos(\theta - \phi)]^T
            cv::Mat t = 1.0 * (cv::Mat_<double>(3, 1) << b.at<double>(1), 0., -b.at<double>(0));
            ts.push_back(t);
        }

        std::vector<cv::Mat> thetas;
        for (int i = 0; i < as.size(); i++)
        {
            // Derived from cos((\theta - \phi) + \phi), sin((\theta - \phi) + \phi)
            // by applying the trigonometric identities
            double costeta = bs[i].at<double>(0) * as[i].at<double>(0) - bs[i].at<double>(1) * as[i].at<double>(1);
            double sinteta = bs[i].at<double>(0) * as[i].at<double>(1) + bs[i].at<double>(1) * as[i].at<double>(0);

            thetas.push_back(1. * (cv::Mat_<double>(2, 1) << costeta, sinteta));
        }

        for (auto theta : thetas)
        {
            // R =
            // [cos(\theta) 0 -sin(\theta)
            //       0      1     0
            //  sin(\theta) 0 cos(\theta)]

            cv::Mat R = (cv::Mat_<double>(3, 3) << theta.at<double>(0), 0, -theta.at<double>(1),
                         0., 1., 0.,
                         theta.at<double>(1), 0, theta.at<double>(0));

            Rs.push_back(R);
        }

        /*std::cout << "ts: " << std::endl;
        for (auto t : ts)
        {
            std::cout << t << std::endl;
        }

        std::cout << "Rs: " << std::endl;
        for (auto R : Rs)
        {
            std::cout << R << std::endl;
        }*/
    }

    void check_inliers(
        const std::vector<cv::Point2d> &src_points,
        const std::vector<cv::Point2d> &dest_points,
        const cv::Mat &R,
        const cv::Mat &t,
        const double threshold,
        std::vector<int> &inliers)
    {
        cv::Mat t_cross = (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2), t.at<double>(1),
                           t.at<double>(2), 0, -t.at<double>(0),
                           -t.at<double>(1), t.at<double>(0), 0);

        cv::Mat E = t_cross * R;

        // Count the inliers
        for (int i = 0; i < src_points.size(); ++i)
        {
            // Symmetric epipolar distance
            cv::Mat pt1 = (cv::Mat_<double>(3, 1) << src_points[i].x, src_points[i].y, 1);
            cv::Mat pt2 = (cv::Mat_<double>(3, 1) << dest_points[i].x, dest_points[i].y, 1);

            // Calculate the error
            cv::Mat lL = E.t() * pt2;
            cv::Mat lR = E * pt1;

            // Calculate the distance of point pt1 from lL
            const double
                &aL = lL.at<double>(0),
                &bL = lL.at<double>(1),
                &cL = lL.at<double>(2);

            double tL = abs(aL * src_points[i].x + bL * src_points[i].y + cL);
            double dL = sqrt(aL * aL + bL * bL);
            double distanceL = tL / dL;

            // Calculate the distance of point pt2 from lR
            const double
                &aR = lR.at<double>(0),
                &bR = lR.at<double>(1),
                &cR = lR.at<double>(2);

            double tR = abs(aR * dest_points[i].x + bR * dest_points[i].y + cR);
            double dR = sqrt(aR * aR + bR * bR);
            double distanceR = tR / dR;

            double dist = 0.5 * (distanceL + distanceR);

            if (dist < threshold)
                inliers.push_back(i);
        }
    }

    void ransac_2points(
        const std::vector<cv::Point2d> &src_points,
        const std::vector<cv::Point2d> &dest_points,
        const int max_iteration,
        const double threshold,
        std::vector<int> &best_inliers,
        std::vector<cv::Mat> &best_Rs,
        std::vector<cv::Mat> &best_ts)
    {
        int n = src_points.size();

        for (int i = 0; i < max_iteration; i++)
        {
            int idx1 = std::rand() % n;
            int idx2 = std::rand() % n;
            while (idx1 == idx2)
                idx2 = std::rand() % n;

            std::vector<int> indices{idx1, idx2};
            std::vector<cv::Mat> Rs, ts;
            get_estimated_translation_rotation(src_points, dest_points, indices, Rs, ts);

            for (int j = 0; j < Rs.size() / 2; j++)
            {
                std::vector<int> inliers;
                check_inliers(
                    src_points,
                    dest_points,
                    Rs[j * 2],
                    ts[j * 2],
                    threshold,
                    inliers);

                if (inliers.size() > best_inliers.size())
                {
                    best_Rs.clear();
                    best_ts.clear();
                    best_inliers.clear();

                    best_inliers.assign(inliers.begin(), inliers.end());
                    best_Rs.assign(Rs.begin() + j * 2, Rs.begin() + j * 2 + 2);
                    best_ts.assign(ts.begin() + j * 2, ts.begin() + j * 2 + 2);
                }
            }
        }
    }

    int count_points_infront_camera(
        const std::vector<cv::Point2d> &src_points,
        const std::vector<cv::Point2d> &dest_points,
        const std::vector<int> &inliers,
        const cv::Mat &P0,
        const cv::Mat &P1)
    {
        int n = inliers.size();

        int infront_camera_cnt = 0;

        for (int i = 0; i < n; i++)
        {
            int idx = inliers[i];

            // Triangulate the point
            cv::Point3d point3d;
            sfm_general_stereo::linear_triangulation(P0, P1, src_points[idx], dest_points[idx], point3d);

            // infront of the camera
            if (point3d.z > 0)
                infront_camera_cnt++;
        }

        return infront_camera_cnt;
    }

    void get_correct_rotation_translation(
        const std::vector<cv::Point2d> &src_points,
        const std::vector<cv::Point2d> &dest_points,
        const std::vector<cv::Mat> &Rs,
        const std::vector<cv::Mat> &ts,
        const cv::Mat &K,
        const std::vector<int> &inliers,
        cv::Mat &R,
        cv::Mat &t)
    {
        cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);

        int best_cnt = 0;
        for (int i = 0; i < Rs.size(); i++)
        {
            cv::Mat P2(3, 4, CV_64F);
            P2.colRange(0, 3) = 1.0 * Rs[i];
            P2.col(3) = 1.0 * ts[i];
            P2 = K * P2;

            int cnt = count_points_infront_camera(
                src_points,
                dest_points,
                inliers,
                P1,
                P2);

            std::cout << "count points infront: " << cnt << std::endl;

            if (cnt > best_cnt)
            {
                Rs[i].copyTo(R);
                ts[i].copyTo(t);
            }
        }
    }

} // namespace sfm

#endif