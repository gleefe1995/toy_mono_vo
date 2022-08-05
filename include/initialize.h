#pragma once

#include "Frame.h"
#include "Feature.h"
#include "map_database.h"
#include <string>
#include <alike.hpp>

namespace initialize{


void initialize_from_essential_mat(cv::Mat &curr_image, Frame::Frame &prev_frame, Frame::Frame &curr_frame, cv::Mat &intrinsic_param, auto &alike,
                                    int &N_matches, int &frame_num, int &keyframe_num, bool &use_cuda, bool &is_keyframe,
                                    Frame::Frame &prev_keyframe, Frame::Frame &detect_prev_frame, 
                                    bool &init, map_database::map_database &map_data)
{
    Feature::ALIKE_feature_detection(curr_image, curr_frame, 
                                        alike, 
                                        use_cuda,is_keyframe);

    Frame::Frame curr_frame_tmp = curr_frame;

    if (frame_num>0)
    {
        N_matches = Feature::ALIKE_feature_tracking(curr_image, prev_frame, curr_frame, prev_keyframe);

        Feature::get_pose_from_essential_mat(prev_frame, curr_frame, intrinsic_param);
    
    
        if (curr_frame.get_good_points_2d().size()<500)
        {
            
            Feature::make_3d_points(curr_image, prev_keyframe, curr_frame, intrinsic_param);

            is_keyframe = true;
            init = true;

            if (is_keyframe == true)
            {
                std::cout<<"init"<<"\n";

                // local ba points
                map_data.add_local_ba(curr_frame);

                curr_frame_tmp.set_points_2d_with_id(keyframe_num);
                curr_frame_tmp.set_camera_pose_from_frame(curr_frame);

                detect_prev_frame = curr_frame_tmp;
                prev_keyframe = curr_frame_tmp;
                
                keyframe_num++;
            }
        }
    }
    else
    {
        curr_frame.set_points_2d_with_id(keyframe_num);
        keyframe_num++;
        prev_keyframe = curr_frame;
        // is_keyframe=true;
    }

    prev_frame = curr_frame;
    // std::cout << "init: " << prev_frame.get_good_points_2d().size() << "\n";
    // std::cout << "init: " << curr_frame.get_good_points_2d().size() << "\n";
    frame_num++;

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