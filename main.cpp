#include "Frame.h"
#include "initialize.h"
#include "Feature.h"


int main(int argc, char **argv)
{
    std::string pti = "/home/gleefe/Downloads/dataset/sequences/" + (std::string)argv[1]+"/image_0/%06d.png";
    std::string path_to_pose = "/home/gleefe/Downloads/dataset/poses/" + (std::string)argv[1] + ".txt";
    const char* path_to_image = pti.c_str();
    
    cv::Mat intrinsic_param = initialize::get_intrinsic_mat( (std::string)argv[1]);

    char filename1[200];
    char filename2[200];
  
    sprintf(filename1, path_to_image, 0);
    sprintf(filename2, path_to_image, 1);

    //read the first two frames from the dataset
    cv::Mat image1_c = cv::imread(filename1);
    cv::Mat image2_c = cv::imread(filename2);

    if ( !image1_c.data || !image2_c.data ) { 
        std::cout<< " --(!) Error reading images " << std::endl; return -1;
    }
    
    cv::Mat image1;
    cv::Mat image2;

    cvtColor(image1_c, image1, cv::COLOR_BGR2GRAY);
    cvtColor(image2_c, image2, cv::COLOR_BGR2GRAY);

    Frame::Frame prev_frame;
    Frame::Frame curr_frame;

    cv::Mat mat_identity=cv::Mat::eye(cv::Size(3,3),CV_64FC1);
    cv::Mat zero_trans = cv::Mat(3,1, CV_64FC1, 0.0);
    prev_frame.set_camera_pose(mat_identity, zero_trans);

    int keyframe_num = 0;

    std::vector<cv::Point2f> curr_points_before_erase;

    Feature::featureDetection(image1, prev_frame, keyframe_num);
    Feature::featureTracking(image1, image2, prev_frame, curr_frame);
    
    Feature::get_pose_from_essential_mat(prev_frame, curr_frame, intrinsic_param);

    // Feature::vis_frame_2d_points(curr_frame, image2_c, true);

    bool is_initialize = false;
    int numFrame = 2;

    cv::Mat prev_Image, curr_Image;
    prev_Image = image2;
    prev_frame = curr_frame;


    cv::Mat traj = cv::Mat::zeros(600, 800, CV_8UC3);

    while (is_initialize == false)
    {
        // std::cout << "Frame start" << "\n";

        sprintf(filename1, path_to_image, numFrame);
        cv::Mat curr_Image_c = cv::imread(filename1);
        cvtColor(curr_Image_c, curr_Image, cv::COLOR_BGR2GRAY);

        
        
        
        // std::cout << " before tracking" << "\n";

        Feature::featureTracking(prev_Image, curr_Image, prev_frame, curr_frame);

        // std::cout << "vis " << "\n";

        Feature::vis_frame_2d_points(curr_frame, curr_Image_c, true);


        Feature::get_pose_from_essential_mat(prev_frame, curr_frame, intrinsic_param);


        prev_Image = curr_Image;
        prev_frame = curr_frame;

        int x = int(curr_frame.get_translation_mat().at<double>(0)) + 300;
        int y = -int(curr_frame.get_translation_mat().at<double>(2)) + 400;

        cv::circle(traj, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);
        // cv::imshow("traj", traj);
        cv::waitKey(1);

        numFrame++;
    }

    return 0;
}