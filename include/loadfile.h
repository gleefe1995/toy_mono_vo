#pragma once

#include <iostream>
#include <fstream>
#include "Feature.h"

namespace loadfile
{

    std::vector<cv::Point3d> get_gt(std::string path_to_pose)
    {

        std::string line;
        double gt_x, gt_y, gt_z;
        int i = 0;
        std::ifstream myfile(path_to_pose);
        double x = 0, y = 0, z = 0;
        std::vector<cv::Point3d> gt_pose_vec;
        if (myfile.is_open())
        {
        while (getline(myfile, line))
        {
            std::istringstream in(line);
            //cout << line << '\n';
            for (int j = 0; j < 12; j++)
            {
            in >> z;
            gt_z = z;
            if (j == 7)
                gt_y = z;
            if (j == 3)
                gt_x = z;
            }
            
            gt_pose_vec.push_back(cv::Point3d(gt_x,gt_y,gt_z));

            i++;
        }
        myfile.close();
        }
        else
        {
        std::cout << "Unable to open file";
        }
        
        return gt_pose_vec;

    }
}
