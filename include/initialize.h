#pragma once

#include "Frame.h"
#include "Feature.h"
#include <string>
#include <alike.hpp>

namespace initialize{


void initialize_from_essential_mat(cv::Mat curr_image, Frame::Frame &prev_frame, Frame::Frame &curr_frame, cv::Mat intrinsic_param, auto alike,
                                    int &N_matches, const int frame_num, int &keyframe_num, bool use_cuda)
{
    Feature::ALIKE_feature_detection(curr_image, curr_frame, 
                                        keyframe_num, alike, 
                                        use_cuda,true);

    if (frame_num>0)
    {
        N_matches = Feature::ALIKE_feature_tracking(curr_image, prev_frame, curr_frame);
        Feature::get_pose_from_essential_mat(prev_frame, curr_frame, intrinsic_param);
    }
}



cv::Mat get_intrinsic_mat(std::string scene_num)
{
    double focal;
    cv::Point2d pp;

    if (scene_num=="00"){
        focal = 718.8560; //00-02
        pp = cv::Point2d(607.1928, 185.2157);
    }
    else if (scene_num=="03"){
        focal = 721.5377; //03
        pp = cv::Point2d(609.5593, 172.854);
    }
    else{
        focal = 707.0912; //04-12
        pp = cv::Point2d(601.8873, 183.1104);
    }

    cv::Mat Kd = (cv::Mat_<double>(3,3)<< focal, 0, pp.x,
                                0, focal, pp.y,
                                0,  0,   1);

    return Kd;
}



}