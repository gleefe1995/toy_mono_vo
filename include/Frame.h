#pragma once

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/core/eigen.hpp>



namespace Frame{

typedef std::pair<std::pair<int, int>, cv::Point3d> point_3d_pair;
typedef std::pair<std::pair<int, int>, cv::Point2f> point_2d_pair;


class Frame
{
private:
    std::vector<point_3d_pair> points_3d_with_id_;
    std::vector<point_2d_pair> points_2d_with_id_;
    std::vector<cv::Point2f> points_2d_;
    std::vector<cv::Point2f> tracking_points_2d;
    // std::vector<point_2d_pair> triangulate_points_2d_with_id_;
    cv::Mat descriptor_;

    int number_of_3d_points_;
    int frame_num_;
    cv::Mat R, t; // camera pose
    std::vector<cv::KeyPoint> keypoints_loop_;
    cv::Mat desc_loop_;
    


public:
    Frame(){};
    
    void set_point_3d(int frame_number, int keypoint_number, cv::Point3d point_3d)
    {
        points_3d_with_id_.push_back( std::make_pair(std::make_pair(frame_number, keypoint_number), point_3d));
    }

    void set_points_2d(std::vector<cv::KeyPoint> sscKP, const int frame_number)
    {
        frame_num_ = frame_number;

        points_2d_.clear();
        points_2d_with_id_.clear();

        cv::KeyPoint::convert(sscKP, points_2d_, std::vector<int>());

        for (int i = 0; i < points_2d_.size(); i++)
            points_2d_with_id_.push_back(std::make_pair(std::make_pair(frame_num_, i), points_2d_.at(i)));

        // triangulate_points_2d_with_id_ = points_2d_with_id_;
    }

    void set_points_2d(std::vector<cv::Point2f> detected_points)
    {
        points_2d_.clear();
        points_2d_ = detected_points;
    }

    void set_good_points_2d(std::vector<cv::Point2f> tracked_points)
    {
        tracking_points_2d.clear();
        tracking_points_2d = tracked_points;
    }

    std::vector<cv::Point2f> get_good_points_2d()
    {
        return tracking_points_2d;
    }


    void set_descriptors(cv::Mat descriptors)
    {
        descriptor_ = descriptors.clone();
    }

    void set_points_2d_with_id(Frame prev_frame)
    {
        points_2d_with_id_.clear();
        for (int i=0; i<points_2d_.size();++i)
            points_2d_with_id_.push_back(std::make_pair(std::make_pair(prev_frame.get_2d_points_with_id()[i].first.first, prev_frame.get_2d_points_with_id()[i].first.second), points_2d_.at(i)));
    }

    void set_points_2d_with_id(const int keyframe_number)
    {
        points_2d_with_id_.clear();
        for (int i = 0; i < points_2d_.size(); i++)
            points_2d_with_id_.push_back(std::make_pair(std::make_pair(keyframe_number, i), points_2d_.at(i)));
    }


    void erase_points_with_index(int i)
    {
        points_2d_.erase(points_2d_.begin() + i);
        points_2d_with_id_.erase(points_2d_with_id_.begin() + i);
        // triangulate_points_2d_with_id_.erase(triangulate_points_2d_with_id_.begin() +i);
    }

    void set_camera_pose(Frame &prev_frame, cv::Mat &rel_rot, cv::Mat &rel_trans)
    {
        t = prev_frame.get_translation_mat() + prev_frame.get_rotation_mat() * rel_trans;
        R = prev_frame.get_rotation_mat() * rel_rot;
    }
    
    void set_camera_pose(cv::Mat rot, cv::Mat trans)
    {
        t = trans;
        R = rot;
    }


    cv::Mat get_desc()
    {
        return descriptor_;
    }


    cv::Mat get_rotation_mat()
    {
        return R;
    }

    cv::Mat get_translation_mat()
    {
        return t;
    }

    int get_keyframe_num()
    {
        return frame_num_;
    }



    int get_num_3d_points()
    {
        return number_of_3d_points_;
    }

    std::vector<point_3d_pair> get_3d_points_with_id()
    {
        return points_3d_with_id_;
    }

    std::vector<point_2d_pair> get_2d_points_with_id()
    {
        return points_2d_with_id_;
    }

    std::vector<cv::Point2f> get_2d_points()
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

    Frame& operator = (Frame curr_frame)
    {
        // std::cout << "Assignment operator" << "\n";

        if (this == &curr_frame)
            return *this;

        this->points_2d_ = curr_frame.get_2d_points();
        this->points_2d_with_id_ = curr_frame.get_2d_points_with_id();
        this->R = curr_frame.get_rotation_mat();
        this->t = curr_frame.get_translation_mat();
        this->descriptor_ = curr_frame.get_desc();

    }
    
};

}