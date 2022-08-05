#include "Frame.h"
#include "initialize.h"
#include "Feature.h"
#include "local_mapping.h"
#include "Viewer.h"
#include "loadfile.h"
#include "map_database.h"

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
#include <thread>

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
    std::string path_to_pose = "/home/gleefe/Downloads/dataset/poses/" + scene_num + ".txt";
    cv::Mat intrinsic_param = initialize::get_intrinsic_mat(scene_num);
    
    std::vector<cv::Point3d> gt_pose_vec;
    std::vector<cv::Point3d> curr_gt_pose_vec;
    gt_pose_vec = loadfile::get_gt(path_to_pose);

    // ===============> main loop
    cv::Mat image;
    auto device = (use_cuda) ? torch::kCUDA : torch::kCPU;

    Frame::Frame prev_frame;
    Frame::Frame curr_frame;

    Frame::Frame detect_prev_frame;
    // Frame::Frame detect_curr_frame;

    cv::Mat mat_identity=cv::Mat::eye(cv::Size(3,3),CV_64FC1);
    cv::Mat zero_trans = cv::Mat(3,1, CV_64FC1, 0.0);
    cv::Mat rvec, tvec;
    cv::Rodrigues(mat_identity, rvec);
    curr_frame.set_camera_pose(rvec, zero_trans);

    int keyframe_num = 0;
    cv::Mat prev_image;
    cv::Mat curr_image;

    int frame_num = 0;

    cv::namedWindow("win");
    bool init = false;
    bool is_keyframe = false;
    // std::vector<Frame::Frame> keyframe_vec;
    Frame::Frame prev_keyframe;
    map_database::map_database map_data;
    
    std::vector<cv::Point3d> our_traj;
    our_traj.push_back(cv::Point3d(0.0,0.0,0.0));

    alike::Viewer mpViewer;
    mpViewer.init();
    float mViewpointX = 0;
    float mViewpointY = -200;
    float mViewpointZ = -0.1;
    float mViewpointF = 100;
    pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
    pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (loader.read_next(image, max_size))
    {
        curr_image = image.clone();
        int N_matches;
        double pnp_inlier_ratio=0;
        
        if (init==false)
        {
            initialize::initialize_from_essential_mat(curr_image, prev_frame, curr_frame, intrinsic_param, alike,
                                    N_matches, frame_num, keyframe_num, use_cuda, is_keyframe, prev_keyframe, detect_prev_frame, 
                                    init, map_data);
        }
        else
        {
            local_mapping::system(curr_image, prev_frame, curr_frame, alike, use_cuda, is_keyframe, N_matches,
            keyframe_num, frame_num, pnp_inlier_ratio, intrinsic_param, 
            detect_prev_frame, prev_keyframe, map_data);
        }

        stringstream fmt;
        fmt << "Frame: " << frame_num << " Keypoints/Matches: " << curr_frame.get_good_points_2d().size() << "/" << N_matches << " ,inlier ratio : " << pnp_inlier_ratio*100 << "% ";
        std::string status = fmt.str();

        if (!no_display)
        {   
            if (is_keyframe)
            {
                cv::Point3d point_traj(cv::Point3d(curr_frame.get_translation_mat().at<double>(0), curr_frame.get_translation_mat().at<double>(1), curr_frame.get_translation_mat().at<double>(2)));
                // std::cout << "Current t: " << point_traj.x <<" "<< point_traj.y << " " <<point_traj.z << "\n";
                our_traj.push_back(point_traj);
                
                //traj update
                int local_ba_vec_size = map_data.get_local_ba_rvec().size();
                for (int i=0;i<local_ba_vec_size;++i)
                {
                    our_traj.pop_back();
                }

                for (int i=0;i<local_ba_vec_size;++i)
                {
                    cv::Mat R,t, R_inv;
                    cv::Mat rvec_tmp(3,1, CV_64FC1, 0.0);
                    cv::Mat tvec_tmp(3,1, CV_64FC1, 0.0);
                    rvec_tmp.at<double>(0) = map_data.get_local_ba_rvec()[i].x;
                    rvec_tmp.at<double>(1) = map_data.get_local_ba_rvec()[i].y;
                    rvec_tmp.at<double>(2) = map_data.get_local_ba_rvec()[i].z;
                    
                    tvec_tmp.at<double>(0) = map_data.get_local_ba_tvec()[i].x;
                    tvec_tmp.at<double>(1) = map_data.get_local_ba_tvec()[i].y;
                    tvec_tmp.at<double>(2) = map_data.get_local_ba_tvec()[i].z;
                    cv::Rodrigues(rvec_tmp, R);
                    R_inv = R.t();
                    t = -R_inv * tvec_tmp;
                    
                    our_traj.push_back(cv::Point3d(t.at<double>(0), t.at<double>(1), t.at<double>(2)));
                }

            

            // std::cout << our_traj.size() << "\n";

                curr_gt_pose_vec.push_back(gt_pose_vec[frame_num]);
                is_keyframe=false;
            }
            

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            glClearColor(0.0f,0.0f,0.0f,1.0f);
            mpViewer.draw_points(our_traj, curr_gt_pose_vec, curr_frame.get_3d_points_with_id());
            pangolin::FinishFrame();
            // std::thread mptViewer = std::thread(&Viewer::Run, &mpViewer);
            cv::setWindowTitle("win", status);
            cv::imshow("win", curr_image);
            auto c = cv::waitKey(1);
            if (c=='q') break;
        }

        prev_image = curr_image.clone();
        
        

    }
    

    return 0;
}