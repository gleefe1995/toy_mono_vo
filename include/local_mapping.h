#pragma once

#include "Frame.h"
#include "Feature.h"
#include "bundle.h"
#include <alike.hpp>

namespace local_mapping
{

int ALIKE_p3p_tracking(cv::Mat img, Frame::Frame &prev_frame, Frame::Frame &curr_frame);
double get_pose_from_p3p(cv::Mat curr_image, Frame::Frame &prev_frame, Frame::Frame &curr_frame, const cv::Mat &intrinsic_param, cv::Mat &rvec, cv::Mat &tvec);


void system(cv::Mat &curr_image, Frame::Frame &prev_frame, Frame::Frame &curr_frame, auto &alike, bool use_cuda, bool &is_keyframe, int &N_matches,
            int &keyframe_num, int &frame_num, double &pnp_inlier_ratio, const cv::Mat &intrinsic_param, 
            Frame::Frame &detect_prev_frame, Frame::Frame &prev_keyframe, map_database::map_database &map_data)
{
    // p3p
    // std::cout << "p3p" << "\n";
    Feature::ALIKE_feature_detection(curr_image, curr_frame, 
                                alike, 
                                use_cuda,is_keyframe);
    Frame::Frame curr_frame_tmp = curr_frame;
    Frame::Frame detect_curr_frame = curr_frame;
    cv::Mat rvec,tvec;
    N_matches = local_mapping::ALIKE_p3p_tracking(curr_image, prev_frame, curr_frame);
    pnp_inlier_ratio = local_mapping::get_pose_from_p3p(curr_image, prev_frame, curr_frame, intrinsic_param, rvec, tvec);
    
    detect_curr_frame.set_camera_pose_from_frame(curr_frame);


    int making_3d_points_N_matches = Feature::ALIKE_feature_tracking(curr_image, detect_prev_frame, detect_curr_frame, prev_keyframe);


    // =============================> add 3d point 
    //(detect_curr_frame.get_good_points_2d().size()<500)
    // (pnp_inlier_ratio < 0.7) || 
    if ( detect_curr_frame.get_good_points_2d().size() < 300 )
    {
        Feature::make_3d_points(curr_image, prev_keyframe, detect_curr_frame, intrinsic_param);
        
        is_keyframe=true;

        if (is_keyframe==true)
        {
            // curr_frame.visualize_2d_3d_points(curr_image, intrinsic_param);
            curr_frame.add_detect_frame(detect_curr_frame);
            // curr_frame.print_data();
            // cv::waitKey();
            // curr_frame = detect_curr_frame;
            // curr_frame.visualize_2d_3d_points(curr_image, intrinsic_param);
            
            if (map_data.get_local_ba_rvec().size()==1)
            {
                map_data.erase_front_keyframe();
            }

            map_data.add_local_ba(curr_frame);
            

            std::vector<Frame::point_3d_pair> curr_3d_points = curr_frame.get_3d_points_with_id();
            
            bundle::localBA(map_data, intrinsic_param, curr_3d_points, rvec, tvec);
            
            
            curr_frame.set_camera_pose(rvec, tvec);
            curr_frame.set_points_3d_with_id(curr_3d_points);
            curr_frame_tmp.set_camera_pose_from_frame(curr_frame);
            curr_frame_tmp.set_points_2d_with_id(keyframe_num);
            
            detect_prev_frame = curr_frame_tmp;
            prev_keyframe = curr_frame_tmp;
            
            

            keyframe_num++;
        }
    }
    else
    {
        detect_prev_frame = detect_curr_frame;
    }

    prev_frame = curr_frame;
    
    frame_num++;
}


int ALIKE_p3p_tracking(cv::Mat img, Frame::Frame &prev_frame, Frame::Frame &curr_frame)
{
    float mMth = 0.7;
    int N_matches;
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;

    // std::cout << "p3p: " << prev_frame.get_desc().size() << "\n";
    // std::cout << "p3p tracking" << "\n";
    // std::cout << prev_frame.get_good_points_2d().size() << "\n";
    // std::cout << curr_frame.get_good_points_2d().size() << "\n";


    matcher.knnMatch(prev_frame.get_desc(), curr_frame.get_desc(), knn_matches, 2);

    int channel = prev_frame.get_desc().cols;

    std::vector<cv::DMatch> good_matches;
    for (auto i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < mMth * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    N_matches = good_matches.size();

    std::vector<cv::Point2f> prev_points;
    std::vector<cv::Point2f> curr_points;
    std::vector<Frame::point_3d_pair> prev_3d_points_with_id;
    std::vector<Frame::point_2d_pair> prev_2d_points_with_id;
    

    cv::Mat prev_desc_slice(0, N_matches, CV_32FC1);
    cv::Mat curr_desc_slice(0, N_matches, CV_32FC1);

    // std::cout << prev_frame.get_good_points_2d().size() << "\n";
    // std::cout << prev_frame.get_3d_points_with_id().size() << "\n";

    for (int i=0; i < N_matches; ++i)
    {
        auto match = good_matches[i];
        auto p1 = prev_frame.get_good_points_2d()[match.queryIdx];
        auto p2 = curr_frame.get_good_points_2d()[match.trainIdx];
        auto p3 = prev_frame.get_3d_points_with_id()[match.queryIdx];
        auto p4 = prev_frame.get_2d_points_with_id()[match.queryIdx];

        prev_desc_slice.push_back(prev_frame.get_desc().row(match.queryIdx));
        curr_desc_slice.push_back(curr_frame.get_desc().row(match.trainIdx));

        prev_points.push_back(p1);
        curr_points.push_back(p2);
        prev_3d_points_with_id.push_back(p3);
        prev_2d_points_with_id.push_back(p4);

        cv::line(img, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(img, p2, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
    // cv::imshow("tracked image", img);
    // cv::waitKey();
    


    prev_frame.set_good_points_2d(prev_points);
    curr_frame.set_good_points_2d(curr_points);

    prev_frame.set_descriptors(prev_desc_slice);
    curr_frame.set_descriptors(curr_desc_slice);

    prev_frame.set_points_2d_with_id(prev_2d_points_with_id);
    curr_frame.set_points_2d_with_id(prev_frame);

    prev_frame.set_point_3d(prev_3d_points_with_id);
    curr_frame.set_point_3d(prev_3d_points_with_id);

    // std::cout << prev_frame.get_good_points_2d().size() << "\n";
    // std::cout << curr_frame.get_good_points_2d().size() << "\n";

    //   prev_keyframe.set_good_points_2d(keyframe_triangulate_points);
    //   prev_keyframe.set_points_2d_with_id(prev_keyframe_2d_points_with_id);

    return N_matches;

}


double get_pose_from_p3p(cv::Mat curr_image, Frame::Frame &prev_frame, Frame::Frame &curr_frame, const cv::Mat &intrinsic_param, cv::Mat &rvec, cv::Mat &tvec)
{
    cv::Mat inlier_array;
    cv::Mat R,t;
    std::vector<cv::Point2f> corr_2d_points_float = curr_frame.get_good_points_2d();
    std::vector<cv::Point2d> corr_2d_points(corr_2d_points_float.begin(),corr_2d_points_float.end());
    std::vector<cv::Point3d> corr_3d_points;
    for (auto &p1 : prev_frame.get_3d_points_with_id())
        corr_3d_points.push_back(p1.second);
    // std::cout << curr_frame.get_good_points_2d().size() << "\n";
    // std::cout << corr_3d_points.size() << "\n";
    // std::cout << corr_2d_points.size() << "\n";
    int corr_3d_point_number=corr_3d_points.size();
    cv::solvePnPRansac(corr_3d_points, corr_2d_points, intrinsic_param, cv::noArray(), rvec, tvec, false, 100, 3.0F, 0.99, inlier_array, cv::SOLVEPNP_P3P);

    double inlier_ratio=double(inlier_array.rows)/double(corr_3d_point_number);

    for (int i=0;i<inlier_array.rows;i++){
      corr_3d_points.push_back(corr_3d_points[inlier_array.at<int>(i)]);
      corr_2d_points.push_back(corr_2d_points[inlier_array.at<int>(i)]);
    }
    
    corr_3d_points.erase(corr_3d_points.begin(),corr_3d_points.begin()+corr_3d_point_number);
    corr_2d_points.erase(corr_2d_points.begin(),corr_2d_points.begin()+corr_3d_point_number);

    double focal = intrinsic_param.at<double>(0,0);
    cv::Point2d pp = cv::Point2d(intrinsic_param.at<double>(0,2), intrinsic_param.at<double>(1,2));

    
    // curr_frame.visualize_2d_3d_points(curr_image, intrinsic_param);
    bundle::motion_only_BA(rvec, tvec, corr_2d_points, corr_3d_points, focal, pp);
    


    curr_frame.set_camera_pose(rvec, tvec);
    // std::cout << "after motion only BA" << "\n";
    // curr_frame.visualize_2d_3d_points(curr_image, intrinsic_param);


    return inlier_ratio;
}





}