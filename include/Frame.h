#pragma once

#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
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
    
    void print_data()
    {
        for (int i=0;i<points_3d_with_id_.size();++i)
        {
            std::cout << points_3d_with_id_[i].first.first << " " << points_3d_with_id_[i].first.second << "\n";
            std::cout << points_2d_with_id_[i].first.first << " " << points_2d_with_id_[i].first.second << "\n";
            // std::cout << points_2d_with_id_[i].second << "\n";
            // std::cout << tracking_points_2d[i] << "\n";
        }

        // std::cout << points_3d_with_id_.size() << "\n";
        // std::cout << points_2d_with_id_.size() << "\n";
        // std::cout << tracking_points_2d.size() << "\n";
    }


    void set_point_3d(std::vector<point_3d_pair> made_points_3d)
    {
        points_3d_with_id_ = made_points_3d;
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

    void set_points_2d(std::vector<cv::Point2f> &detected_points)
    {
        points_2d_ = detected_points;
    }

    void set_good_points_2d(std::vector<cv::Point2f> &tracked_points)
    {
        tracking_points_2d = tracked_points;
    }

    
    void set_descriptors(cv::Mat &descriptors)
    {
        descriptor_ = descriptors.clone();
    }

    void set_points_2d_with_id(Frame &prev_frame)
    {
        std::vector<point_2d_pair> points_2d_with_id_tmp;
        for (int i=0; i<tracking_points_2d.size();++i)
            points_2d_with_id_tmp.push_back(std::make_pair(prev_frame.get_2d_points_with_id()[i].first, tracking_points_2d[i]));
        points_2d_with_id_ = points_2d_with_id_tmp;
    }

    void set_points_2d_with_id(const int keyframe_number)
    {
        std::vector<point_2d_pair> points_2d_with_id_tmp;
        for (int i = 0; i < tracking_points_2d.size(); i++)
            points_2d_with_id_tmp.push_back(std::make_pair(std::make_pair(keyframe_number, i), tracking_points_2d[i]));
        points_2d_with_id_ = points_2d_with_id_tmp;
    }

    void set_points_2d_with_id(std::vector<point_2d_pair> &points_2d_with_id)
    {
        points_2d_with_id_ = points_2d_with_id;
    }

    void set_points_3d_with_id(std::vector<point_3d_pair> &points_3d_with_id)
    {
        points_3d_with_id_ = points_3d_with_id;
    }

    void add_detect_frame(Frame &detect_curr_frame)
    {
        std::vector<cv::Point2f> detect_2d_points = detect_curr_frame.get_good_points_2d();
        // int num_of_tracking_points_2d = tracking_points_2d.size();
        int tracking_points_size = tracking_points_2d.size();
        

        for (int i=0; i<detect_2d_points.size();++i)
        {
            int count = 0;
            cv::Point2f pt2 = detect_2d_points[i];
            for (int j=0; j<tracking_points_size;++j)
            {
                float x1,x2,y1,y2;
                cv::Point2f pt1 = tracking_points_2d[j];
                x1 = pt1.x;
                y1 = pt1.y;
                x2 = pt2.x;
                y2 = pt2.y;
                float distance = std::sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));
                
                if (distance==0)
                {
                    count++;
                    break;
                }
            }
            if (count==0)
            {
                points_2d_with_id_.push_back(detect_curr_frame.get_2d_points_with_id()[i]);
                points_3d_with_id_.push_back(detect_curr_frame.get_3d_points_with_id()[i]);
                descriptor_.push_back(detect_curr_frame.get_desc().row(i));
                tracking_points_2d.push_back(pt2); 
            }
        }
        
    }

    void visualize_2d_3d_points(cv::Mat &curr_image, const cv::Mat &intrinsic_mat)
    {
        for (auto &pt : tracking_points_2d)
        {
            cv::circle(curr_image, cv::Point((int)pt.x, (int)pt.y) ,2, CV_RGB(255,0,0), 2);
        }

        cv::Mat Rt1 = cv::Mat::eye(3, 4, CV_64FC1);
        cv::Mat currRotProj = R.t();
        cv::Mat currTransProj = -currRotProj*t;

        currRotProj.copyTo(Rt1.rowRange(0,3).colRange(0,3));
        currTransProj.copyTo(Rt1.rowRange(0,3).col(3));

        for (auto &pt : points_3d_with_id_)
        {
            cv::Mat point_3d(4,1,CV_64F);
            point_3d.at<double>(0)=pt.second.x;
            point_3d.at<double>(1)=pt.second.y;
            point_3d.at<double>(2)=pt.second.z;   
            point_3d.at<double>(3)=1;

            cv::Mat three_to_p=intrinsic_mat * Rt1 * point_3d;
            int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
            int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
            cv::circle(curr_image, cv::Point(c,d),2,CV_RGB(0,0,255),2);
        }
        cv::imshow("vis 2d 3d points", curr_image);
        cv::waitKey();

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
    
    void set_camera_pose(cv::Mat &rvec, cv::Mat &tvec)
    {
        cv::Mat R_, t_;
        cv::Rodrigues(rvec, R_);
        cv::Mat R_inv = R_.t();
        t_ = -R_inv * tvec;
        
        t = t_.clone();
        R = R_inv.clone();
    }

    void set_camera_pose_from_frame(Frame &curr_frame)
    {
        R = curr_frame.get_rotation_mat().clone();
        t = curr_frame.get_translation_mat().clone();
    }

    std::vector<cv::Point2f> get_good_points_2d()
    {
        return tracking_points_2d;
    }


    cv::Mat get_desc()
    {
        return descriptor_;
    }

    cv::Mat get_proj_rvec()
    {
        cv::Mat R_proj = R.t();
        cv::Mat rvec;
        cv::Rodrigues(R_proj, rvec);
        return rvec;
    }

    cv::Mat get_proj_tvec()
    {
        cv::Mat R_proj = R.t();
        cv::Mat tvec = -R_proj * t;
        return tvec;
        
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

    Frame& operator = (Frame &curr_frame)
    {
        // std::cout << "Assignment operator" << "\n";

        if (this == &curr_frame)
        {   
            return *this;
        }

        // points_2d_ = new std::vector<cv::Point2f>;
        // points_2d_with_id_ = new std::vector<point_2d_pair>;

        // this->points_2d_ = curr_frame.get_2d_points();
        this->points_2d_with_id_ = curr_frame.get_2d_points_with_id();
        this->points_3d_with_id_ = curr_frame.get_3d_points_with_id();
        this->descriptor_ = curr_frame.get_desc().clone();
        this->tracking_points_2d = curr_frame.get_good_points_2d();

        this->R = curr_frame.get_rotation_mat().clone();
        this->t = curr_frame.get_translation_mat().clone();
        

    }
    
};

}