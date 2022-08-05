#pragma once

#include <pangolin/pangolin.h>
#include <mutex>
#include "Frame.h"

namespace alike
{

class Viewer
{
private:
    int window_width_;
    int window_height_;



    std::mutex mMutexFinish;

    double mT = 1e-3/30;

    std::vector<cv::Point3d> our_traj_;
    std::vector<Frame::point_3d_pair> points_3d_with_id_;
    cv::Mat curr_image_;

public:
    float window_ratio;
public:
    Viewer(){}
    
    void init()
    {
        pangolin::CreateWindowAndBind("ALIKE: Map Viewer",1024,768);
        glEnable(GL_DEPTH_TEST);
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }


    void set_traj_points_3d(std::vector<cv::Point3d> &our_traj, std::vector<Frame::point_3d_pair> points_3d_with_id)
    {
        our_traj_ = our_traj;
        points_3d_with_id_ = points_3d_with_id;
    }

    void set_image(cv::Mat &curr_image)
    {
        curr_image_ = curr_image.clone();
    }

    void draw_points(std::vector<cv::Point3d> &our_traj, std::vector<cv::Point3d> &curr_gt_pose_vec, std::vector<Frame::point_3d_pair> points_3d_with_id)
    {
        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0);

        for (auto &pt : our_traj)
            glVertex3d(pt.x, pt.y, pt.z);
        glEnd();

        glPointSize(3);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,1.0);

        for (auto &pt : curr_gt_pose_vec)
            glVertex3d(pt.x, pt.y, pt.z);
        glEnd();

        glPointSize(1);
        glBegin(GL_POINTS);
        glColor3f(1.0,1.0,0.0);
        
        for (auto &pt : points_3d_with_id)
            glVertex3d(pt.second.x, pt.second.y, pt.second.z);
        
        
        glEnd();
    }




};


}