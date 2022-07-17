#include "Frame.h"
#include "initialize.h"
#include "Feature.h"

#include <chrono>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <args.hxx>
#include <image_loader.hpp>
#include <alike.hpp>
#include <simple_tracker.hpp>
#include <utils.h>
#include <iostream>
#include <sstream>
#include <string>

using std::stringstream;

using namespace alike;


int main(int argc, char **argv)
{
    // ===============> args
    args::ArgumentParser parser("ALIKE-demo");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Positional<std::string> file_path_parser(parser,
                                                   "input",
                                                   "Image directory or movie file or 'camera0' (for webcam0).",
                                                   args::Options::Required);
    args::Positional<std::string> model_path_parser(parser,
                                                    "model",
                                                    "Path of alike torchscript model.",
                                                    args::Options::Required);

    args::Flag use_cuda_parser(parser, "cuda", "Use cuda or not.", {"cuda"});
    args::ValueFlag<int> top_k_parser(parser,
                                      "top_k",
                                      "Detect top K keypoints. <=0 for threshold based mode, >0 for top K mode. [default: -1]",
                                      {"top_k"},
                                      -1);
    args::ValueFlag<float> scores_th_parser(parser,
                                            "scores_th",
                                            "Detector score threshold. [default: 0.2]",
                                            {"scores_th"},
                                            0.2);
    args::ValueFlag<int> n_limit_parser(parser,
                                        "n_limit",
                                        "Maximum number of keypoints to be detected. [default: 5000]",
                                        {"n_limit"},
                                        5000);
    args::ValueFlag<float> ratio_parser(parser,
                                        "ratio",
                                        "Ratio in FLANN matching process. [default: 0.7]",
                                        {"ratio"},
                                        0.7);
    args::ValueFlag<int> max_size_parser(parser,
                                         "max_size",
                                         "Maximum image size. (<=0 original; >0 for maximum image size). [default: -1]",
                                         {"max_size"},
                                         -1);
    args::Flag no_display_parser(parser,
                                 "no_display",
                                 "Do not display images to screen. Useful if running remotely.",
                                 {"no_display"});
    args::Flag no_subpixel_parser(parser,
                                  "no_subpixel",
                                  "Do not detect sub-pixel keypoints.",
                                  {"no_subpixel"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    catch (args::ValidationError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    std::string file_path = args::get(file_path_parser);
    std::string model_path = args::get(model_path_parser);
    bool use_cuda = args::get(use_cuda_parser);
    int top_k = args::get(top_k_parser);
    float scores_th = args::get(scores_th_parser);
    int n_limit = args::get(n_limit_parser);
    int max_size = args::get(max_size_parser);
    float ratio = args::get(ratio_parser);
    bool no_display = args::get(no_display_parser);
    bool no_subpixel = args::get(no_subpixel_parser);

    std::cout << "=======================" << std::endl;
    std::cout << "Running with " << ((use_cuda) ? "CUDA" : "CPU") << "!" << std::endl;
    std::cout << "=======================" << std::endl;

    auto loader = ImageLoader(file_path);
    auto alike = ALIKE(model_path, use_cuda, 2, top_k, scores_th, n_limit, !no_subpixel);

    std::string scene_num = file_path.substr(file_path.length()-10,2);
    cv::Mat intrinsic_param = initialize::get_intrinsic_mat(scene_num);
    // ===============> main loop
    cv::Mat image;
    auto device = (use_cuda) ? torch::kCUDA : torch::kCPU;

    Frame::Frame prev_frame;
    Frame::Frame curr_frame;

    cv::Mat mat_identity=cv::Mat::eye(cv::Size(3,3),CV_64FC1);
    cv::Mat zero_trans = cv::Mat(3,1, CV_64FC1, 0.0);
    curr_frame.set_camera_pose(mat_identity, zero_trans);

    int keyframe_num = 0;
    cv::Mat prev_image;
    cv::Mat curr_image;

    int frame_num = 0;

    cv::namedWindow("win");
    cv::Mat traj = cv::Mat::zeros(600, 800, CV_8UC3);

    bool init = false;

    while (loader.read_next(image, max_size))
    {
        curr_image = image.clone();
        int N_matches;
        
        if (init==false)
        {
            initialize::initialize_from_essential_mat(curr_image, prev_frame, curr_frame, intrinsic_param, alike,
                                    N_matches, frame_num, keyframe_num, use_cuda);
        }
        

        


        stringstream fmt;
        double tracking_ratio = N_matches / (double)curr_frame.get_2d_points().size();
        fmt << "Keypoints/Matches: " << curr_frame.get_2d_points().size() << "/" << N_matches << " , tracking ratio : " << tracking_ratio*100 << "%" ;
        std::string status = fmt.str();

        if (!no_display)
        {   
            int x = 300;
            int y = 400;
            if (frame_num>0)
            {
                x += int(curr_frame.get_translation_mat().at<double>(0));
                y += -int(curr_frame.get_translation_mat().at<double>(2));
            
            }

            cv::circle(traj, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);
            cv::imshow("traj", traj);

            cv::setWindowTitle("win", status);
            cv::imshow("win", curr_image);
            auto c = cv::waitKey(1);
            if (c=='q') break;
        }

        prev_image = curr_image.clone();
        prev_frame = curr_frame;
        frame_num++;
    }
    

    

    

    

    // Feature::featureDetection(image1, prev_frame, keyframe_num);
    // Feature::featureTracking(image1, image2, prev_frame, curr_frame);
    
    // Feature::get_pose_from_essential_mat(prev_frame, curr_frame, intrinsic_param);

    // // Feature::vis_frame_2d_points(curr_frame, image2_c, true);

    // bool is_initialize = false;
    // int numFrame = 2;

    // cv::Mat prev_Image, curr_Image;
    // prev_Image = image2.clone();
    // prev_frame = curr_frame;


    // cv::Mat traj = cv::Mat::zeros(600, 800, CV_8UC3);

    // while (is_initialize == false)
    // {
    //     // std::cout << "Frame start" << "\n";

    //     sprintf(filename1, path_to_image, numFrame);
    //     cv::Mat curr_Image_c = cv::imread(filename1);
    //     cvtColor(curr_Image_c, curr_Image, cv::COLOR_BGR2GRAY);

    //     // std::cout << " before tracking" << "\n";

    //     Feature::featureTracking(prev_Image, curr_Image, prev_frame, curr_frame);

    //     // std::cout << "vis " << "\n";

    //     Feature::vis_frame_2d_points(curr_frame, curr_Image_c, false);


    //     Feature::get_pose_from_essential_mat(prev_frame, curr_frame, intrinsic_param);


    //     if (prev_frame.get_2d_points().size()<200)
    //     {
    //         Feature::featureDetection(prev_Image, prev_frame, keyframe_num);
    //         Feature::featureTracking(prev_Image, curr_Image, prev_frame, curr_frame);
    //     }



    //     prev_Image = curr_Image.clone();
    //     prev_frame = curr_frame;

    //     int x = int(curr_frame.get_translation_mat().at<double>(0)) + 300;
    //     int y = -int(curr_frame.get_translation_mat().at<double>(2)) + 400;

    //     cv::circle(traj, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);
    //     cv::imshow("traj", traj);
    //     cv::waitKey(1);

    //     numFrame++;
    // }

    return 0;
}