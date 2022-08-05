#pragma once
#include "Frame.h"

namespace map_database
{



class map_database
{
private:
    std::vector<cv::Point3d> local_ba_rvec;
    std::vector<cv::Point3d> local_ba_tvec;
    std::vector<Frame::point_2d_pair> local_ba_points_2d_with_id;
    std::vector<Frame::point_3d_pair> local_ba_points_3d_with_id;
    std::vector<int> local_ba_number_of_points;
    
    

public:

    void add_local_ba(Frame::Frame &keyframe)
    {
        cv::Mat rvec = keyframe.get_proj_rvec();
        cv::Mat tvec = keyframe.get_proj_tvec();

        local_ba_rvec.push_back(cv::Point3d(rvec.at<double>(0),rvec.at<double>(1),rvec.at<double>(2)));
        local_ba_tvec.push_back(cv::Point3d(tvec.at<double>(0),tvec.at<double>(1),tvec.at<double>(2)));

        int number_of_points = keyframe.get_3d_points_with_id().size();

        local_ba_number_of_points.push_back(number_of_points);

        for (int i=0; i < keyframe.get_3d_points_with_id().size();++i)
        {
            local_ba_points_3d_with_id.push_back(keyframe.get_3d_points_with_id()[i]);
        }

        for (int i=0; i < keyframe.get_2d_points_with_id().size();++i)
        {
            local_ba_points_2d_with_id.push_back(keyframe.get_2d_points_with_id()[i]);
        }

        // for (int i=0;i<number_of_points;++i)
        // {
        //     std::cout << keyframe.get_3d_points_with_id()[i].first << " " << keyframe.get_2d_points_with_id()[i].first << "\n";
        //     std::cout << keyframe.get_3d_points_with_id()[i].second << " " << keyframe.get_2d_points_with_id()[i].second << "\n";
        // }
        // // cv::waitKey();

        // std::cout << keyframe.get_3d_points_with_id().size() << "\n";
        // std::cout << keyframe.get_2d_points_with_id().size() << "\n";
        // std::cout << keyframe.get_good_points_2d().size() << "\n";
        
        // std::cout << rvec << "\n";
        // std::cout << tvec << "\n";
        // cv::waitKey();
    }

    void erase_front_keyframe()
    {
        local_ba_rvec.erase(local_ba_rvec.begin());
        local_ba_tvec.erase(local_ba_tvec.begin());
        int number_of_points = local_ba_number_of_points[0];
        local_ba_number_of_points.erase(local_ba_number_of_points.begin());
        local_ba_points_3d_with_id.erase(local_ba_points_3d_with_id.begin(), local_ba_points_3d_with_id.begin() + number_of_points);
        local_ba_points_2d_with_id.erase(local_ba_points_2d_with_id.begin(), local_ba_points_2d_with_id.begin() + number_of_points);

    }

    std::vector<cv::Point3d> get_local_ba_rvec()
    {
        return local_ba_rvec;
    }

    std::vector<cv::Point3d> get_local_ba_tvec()
    {
        return local_ba_tvec;
    }

    void set_local_ba_3d_points_and_pose(std::vector<cv::Point3d> &rvec_local_ba, std::vector<cv::Point3d> &tvec_local_ba, std::vector<Frame::point_3d_pair> &points_3d_with_id_local_ba)
    {
        local_ba_rvec = rvec_local_ba;
        local_ba_tvec = tvec_local_ba;
        local_ba_points_3d_with_id = points_3d_with_id_local_ba;
    }

    auto get_local_ba_points_3d_with_id()
    {
        return local_ba_points_3d_with_id;
    }

    auto get_local_ba_points_2d_with_id()
    {
        return local_ba_points_2d_with_id;
    }

    auto get_local_ba_number_of_points()
    {
        return local_ba_number_of_points;
    }
};




}