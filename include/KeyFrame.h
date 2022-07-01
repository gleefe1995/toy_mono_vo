#pragma once

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/core/eigen.hpp>



namespace Toy{

typedef std::pair<int, std::pair<int, cv::Point3d>> point_3d_pair;
typedef std::pair<int, std::pair<int, cv::Point2f>> point_2d_pair;


class KeyFrame
{
private:
    std::vector<point_3d_pair> points_3d_;
    std::vector<point_2d_pair> points_2d_;
    int number_of_3d_points_;
    int keyframe_num_;
    cv::Mat rvec_, tvec_;
    std::vector<cv::KeyPoint> keypoints_loop_;
    cv::Mat desc_loop_;
    


public:
    KeyFrame(){};
    
    void set_point_3d(int keyframe_number, int keypoint_number, cv::Point3d point_3d)
    {
        points_3d_.push_back(std::make_pair(keyframe_number, std::make_pair(keypoint_number,point_3d)));
    }


    cv::Mat get_rvec()
    {
        return rvec_;
    }

    cv::Mat get_tvec()
    {
        return tvec_;
    }

    int get_keyframe_num()
    {
        return keyframe_num_;
    }

    int get_num_3d_points()
    {
        return number_of_3d_points_;
    }

    std::vector<point_3d_pair> get_3d_points()
    {
        return points_3d_;
    }

    std::vector<point_2d_pair> get_2d_points()
    {
        return points_2d_;
    }

    std::vector<cv::KeyPoint> get_keypoints_loop()
    {
        return keypoints_loop_;
    }

    cv::Mat get_desc_loop()
    {
        return desc_loop_;
    }
    
};

}