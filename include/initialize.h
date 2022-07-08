#pragma once

#include "Frame.h"
#include <string>

namespace initialize{

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