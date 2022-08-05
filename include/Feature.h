#pragma once
#include "Frame.h"
#include <iostream>
#include <numeric>
#include <math.h>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <args.hxx>
#include "image_loader.hpp"
#include "alike.hpp"
#include "simple_tracker.hpp"
#include "utils.h"



namespace Feature{




void ALIKE_feature_detection(cv::Mat &img, Frame::Frame &curr_frame, alike::ALIKE &alike, bool use_cuda ,bool is_keyframe)
{
  torch::Tensor score_map, descriptor_map;
  torch::Tensor keypoints_t, dispersitys_t, kptscores_t, descriptors_t;
  std::vector<cv::Point2f> keypoints;
  cv::Mat descriptors;
  auto device = (use_cuda) ? torch::kCUDA : torch::kCPU;

  auto img_tensor = alike::mat2Tensor(img).permute({2, 0, 1}).unsqueeze(0).to(device).to(torch::kFloat) / 255;

  alike.extract(img_tensor, score_map, descriptor_map);
  alike.detectAndCompute(score_map, descriptor_map, keypoints_t, dispersitys_t, kptscores_t, descriptors_t);
  alike.toOpenCVFormat(keypoints_t, dispersitys_t, kptscores_t, descriptors_t, keypoints, descriptors);

  // curr_frame.set_points_2d(keypoints);
  curr_frame.set_good_points_2d(keypoints);
  curr_frame.set_descriptors(descriptors);

}



int ALIKE_feature_tracking(cv::Mat &img, Frame::Frame &prev_frame, Frame::Frame &curr_frame, Frame::Frame &prev_keyframe)
{
  float mMth = 0.7;
  int N_matches;
  cv::FlannBasedMatcher matcher;
  std::vector<std::vector<cv::DMatch>> knn_matches;

  matcher.knnMatch(prev_frame.get_desc(), curr_frame.get_desc(), knn_matches, 2);

  // std::cout << prev_frame.get_good_points_2d().size() << "\n";


  int channel = prev_frame.get_desc().cols;

  std::vector<cv::DMatch> good_matches;
  for (auto i = 0; i < knn_matches.size(); i++)
  {
      if (knn_matches[i][0].distance < mMth * knn_matches[i][1].distance)
      {
          good_matches.push_back(knn_matches[i][0]);
      }
  }

  N_matches = good_matches.size();

  std::vector<cv::Point2f> prev_points;
  std::vector<cv::Point2f> curr_points;
  std::vector<cv::Point2f> keyframe_triangulate_points;
  std::vector<Frame::point_2d_pair> prev_2d_points_with_id;
  std::vector<Frame::point_2d_pair> prev_keyframe_2d_points_with_id;
  // std::vector<Frame::point_2d_pair> curr_frame_2d_points_with_id;

  cv::Mat prev_desc_slice(0, N_matches, CV_32FC1);
  cv::Mat curr_desc_slice(0, N_matches, CV_32FC1);

  // std::cout << curr_frame.get_good_points_2d().size() << "\n";
  // std::cout << prev_keyframe.get_good_points_2d().size() << "\n";
  // std::cout << prev_keyframe.get_2d_points_with_id().size() << "\n";

  // std::cout << "Tracking" << "\n";
  // std::cout << prev_frame.get_2d_points_with_id().size() << "\n";

  for (int i=0; i < N_matches; ++i)
  {
    auto match = good_matches[i];
    auto p1 = prev_frame.get_good_points_2d()[match.queryIdx];
    auto p2 = curr_frame.get_good_points_2d()[match.trainIdx];
    auto p3 = prev_keyframe.get_good_points_2d()[match.queryIdx];

    auto p4 = prev_keyframe.get_2d_points_with_id()[match.queryIdx];
    auto p5 = prev_frame.get_2d_points_with_id()[match.queryIdx];

    prev_desc_slice.push_back(prev_frame.get_desc().row(match.queryIdx));
    curr_desc_slice.push_back(curr_frame.get_desc().row(match.trainIdx));

    prev_points.push_back(p1);
    curr_points.push_back(p2);
    keyframe_triangulate_points.push_back(p3);
    prev_keyframe_2d_points_with_id.push_back(p4);
    prev_2d_points_with_id.push_back(p5);
    
    // cv::line(img, p1, p2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    // cv::circle(img, p1, 1, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    // cv::circle(img, p2, 1, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
  }
  // cv::imshow("make 3d points", img);
  


  prev_frame.set_good_points_2d(prev_points);
  curr_frame.set_good_points_2d(curr_points);
  prev_keyframe.set_good_points_2d(keyframe_triangulate_points);
  
  prev_frame.set_descriptors(prev_desc_slice);
  curr_frame.set_descriptors(curr_desc_slice);

  prev_frame.set_points_2d_with_id(prev_2d_points_with_id);
  curr_frame.set_points_2d_with_id(prev_frame);
  prev_keyframe.set_points_2d_with_id(prev_keyframe_2d_points_with_id);
  
  return N_matches;

}

void make_3d_points(cv::Mat &curr_image, Frame::Frame &prev_keyframe, Frame::Frame &curr_frame, const cv::Mat &intrinsic_param)
{
  cv::Mat Rt0 = cv::Mat::eye(3, 4, CV_64FC1);
  cv::Mat Rt1 = cv::Mat::eye(3, 4, CV_64FC1);
  
  cv::Mat prevRotProj = prev_keyframe.get_rotation_mat().t();
  cv::Mat prevTransProj = -prevRotProj*prev_keyframe.get_translation_mat();

  cv::Mat currRotProj = curr_frame.get_rotation_mat().t();
  cv::Mat currTransProj = -currRotProj*curr_frame.get_translation_mat();

  prevRotProj.copyTo(Rt0.rowRange(0,3).colRange(0,3));
  prevTransProj.copyTo(Rt0.rowRange(0,3).col(3));
  
  currRotProj.copyTo(Rt1.rowRange(0,3).colRange(0,3));
  currTransProj.copyTo(Rt1.rowRange(0,3).col(3));

  // std::cout << Rt0 << "\n";
  // std::cout << Rt1 << "\n";



  cv::Mat points_3d_homo;

  cv::triangulatePoints(intrinsic_param*Rt0,intrinsic_param*Rt1,prev_keyframe.get_good_points_2d(),curr_frame.get_good_points_2d(),points_3d_homo);

  std::vector<cv::Point2f> curr_points;
  std::vector<Frame::point_2d_pair> curr_points_2d_with_id;
  std::vector<Frame::point_3d_pair> curr_points_3d_with_id;
  std::vector<Frame::point_2d_pair> prev_keyframe_points_2d_with_id;
  
  // cv::Mat prev_desc_slice(0, N_matches, CV_32FC1);
  cv::Mat curr_desc_slice(0, points_3d_homo.cols, CV_32FC1);

  for(int i = 0; i < points_3d_homo.cols; i++)
  {
    int m = curr_frame.get_good_points_2d()[i].x;
    int n = curr_frame.get_good_points_2d()[i].y;
    // cv::circle(curr_image, cv::Point(m, n) ,1, CV_RGB(255,0,0), 2);

    
    cv::Mat point_3d_homo = points_3d_homo.col(i); 
    point_3d_homo /= point_3d_homo.at<float>(3);
    point_3d_homo.convertTo(point_3d_homo, CV_64F);
    
    cv::Mat three_to_p=intrinsic_param*Rt1*point_3d_homo;


    int c = int(three_to_p.at<double>(0) / three_to_p.at<double>(2));
    int d = int(three_to_p.at<double>(1) / three_to_p.at<double>(2));
    
    // cv::circle(curr_image, cv::Point(c,d),2,CV_RGB(0,0,255),-1);

    int point_diff_x = (m-c)*(m-c);
    int point_diff_y = (n-d)*(n-d);
    int reprojectionError = 10;

    
    float x_para2=curr_frame.get_good_points_2d()[i].x-prev_keyframe.get_good_points_2d()[i].x;
    float y_para2=curr_frame.get_good_points_2d()[i].y-prev_keyframe.get_good_points_2d()[i].y;

    float parallax2=std::sqrt(x_para2*x_para2+y_para2*y_para2);

    if((c>0)&&(d>0)&&(c<curr_image.cols)&&(d<curr_image.rows)&&(sqrt(point_diff_x+point_diff_y)<reprojectionError)
        &&(parallax2>10))
    {
      cv::circle(curr_image, cv::Point(c,d),2,CV_RGB(0,255,255),-1);

      auto p1 = prev_keyframe.get_2d_points_with_id()[i];
      auto p2 = curr_frame.get_good_points_2d()[i];
      // auto p3 = curr_frame.get_2d_points_with_id()[i];

      // prev_desc_slice.push_back(prev_frame.get_desc().row(i));
      curr_desc_slice.push_back(curr_frame.get_desc().row(i));
      
      curr_points.push_back(p2);
      prev_keyframe_points_2d_with_id.push_back(p1);
      curr_points_2d_with_id.push_back(std::make_pair(p1.first, p2));
      // std::cout << p1.first << " " << p3.first << "\n";
      // std::cout << p2 << " " << p3.second << "\n";
      curr_points_3d_with_id.push_back(std::make_pair(p1.first, cv::Point3d(point_3d_homo.at<double>(0),point_3d_homo.at<double>(1),point_3d_homo.at<double>(2))));

    }

  }
  // cv::waitKey();
  // std::cout << curr_desc_slice.size() << "\n";

  curr_frame.set_good_points_2d(curr_points);
  curr_frame.set_points_2d_with_id(curr_points_2d_with_id);
  curr_frame.set_points_3d_with_id(curr_points_3d_with_id);
  prev_keyframe.set_points_3d_with_id(curr_points_3d_with_id);
  prev_keyframe.set_points_2d_with_id(prev_keyframe_points_2d_with_id);
  // prev_frame.set_descriptors(prev_desc_slice);
  curr_frame.set_descriptors(curr_desc_slice);
  
  // cv::imshow("3d_points", curr_image);
  // cv::waitKey();
}





std::vector<cv::KeyPoint> ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints, float tolerance, int cols, int rows);
void erase_not_tracked_points(Frame::Frame &prev_frame, Frame::Frame &curr_frame, std::vector<uchar> status);

void featureDetection(cv::Mat img, Frame::Frame &curr_Frame, int &keyframe_number)
{

  // points1_map.clear();
  // vector <pair<int,Point2f>>().swap(points1_map);
  // goodFeaturesToTrack(img_1, points1, MAX_CORNERS, 0.01, 10);
  //  Size winSize = Size( 5, 5 );
  //  Size zeroZone = Size( -1, -1 );
  //  TermCriteria criteria = TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 40, 0.001 );
  //  cornerSubPix( img_1, points1, winSize, zeroZone, criteria );
  //***********************************************************************************

  std::vector<cv::KeyPoint> keyPoints;
  int fast_threshold = 1;
  bool nonmaxSuppression = true;
  cv::FAST(img, keyPoints, fast_threshold, nonmaxSuppression);
  
  cv::Mat mask;

  //detector->detect(img_1, keyPoints,mask);

  // KeyPoint::convert(keypoints_1, points1, vector<int>());
  int numRetPoints = 1500; //choose exact number of return points
  //float percentage = 0.1; //or choose percentage of points to be return
  //int numRetPoints = (int)keyPoints.size()*percentage;

  float tolerance = 0.1; // tolerance of the number of return points

  //Sorting keypoints by deacreasing order of strength
  std::vector<float> responseVector;
  for (unsigned int i = 0; i < keyPoints.size(); i++)
    responseVector.push_back(keyPoints[i].response);
  std::vector<int> Indx(responseVector.size());
  std::iota(std::begin(Indx), std::end(Indx), 0);
  cv::sortIdx(responseVector, Indx, cv::SORT_DESCENDING);
  std::vector<cv::KeyPoint> keyPointsSorted;
  for (unsigned int i = 0; i < keyPoints.size(); i++)
    keyPointsSorted.push_back(keyPoints[Indx[i]]);

  std::vector<cv::KeyPoint> sscKP = ssc(keyPointsSorted, numRetPoints, tolerance, img.cols, img.rows);
  
  curr_Frame.set_points_2d(sscKP, keyframe_number);
  

  // cv::Ptr<cv::ORB> compute_desc = cv::ORB::create();
  // cv::Mat descriptor;

  // compute_desc->compute(img,sscKP, descriptor);
  // curr_Frame.set_tracking_descriptor(descriptor);
  
  keyframe_number++;
}



std::vector<cv::KeyPoint> ssc(std::vector<cv::KeyPoint> keyPoints, int numRetPoints, float tolerance, int cols, int rows)
{
  // several temp expression variables to simplify solution equation
  int exp1 = rows + cols + 2 * numRetPoints;
  long long exp2 = ((long long)4 * cols + (long long)4 * numRetPoints + (long long)4 * rows * numRetPoints + (long long)rows * rows + (long long)cols * cols - (long long)2 * rows * cols + (long long)4 * rows * cols * numRetPoints);
  double exp3 = sqrt(exp2);
  double exp4 = numRetPoints - 1;

  double sol1 = -round((exp1 + exp3) / exp4); // first solution
  double sol2 = -round((exp1 - exp3) / exp4); // second solution

  int high = (sol1 > sol2) ? sol1 : sol2; //binary search range initialization with positive solution
  int low = floor(sqrt((double)keyPoints.size() / numRetPoints));

  int width;
  int prevWidth = -1;

  std::vector<int> ResultVec;
  bool complete = false;
  unsigned int K = numRetPoints;
  unsigned int Kmin = round(K - (K * tolerance));
  unsigned int Kmax = round(K + (K * tolerance));

  std::vector<int> result;
  result.reserve(keyPoints.size());
  while (!complete)
  {
    width = low + (high - low) / 2;
    if (width == prevWidth || low > high)
    {                     //needed to reassure the same radius is not repeated again
      ResultVec = result; //return the keypoints from the previous iteration
      break;
    }
    result.clear();
    double c = width / 2; //initializing Grid
    int numCellCols = floor(cols / c);
    int numCellRows = floor(rows / c);
    std::vector<std::vector<bool>> coveredVec(numCellRows + 1, std::vector<bool>(numCellCols + 1, false));

    for (unsigned int i = 0; i < keyPoints.size(); ++i)
    {
      int row = floor(keyPoints[i].pt.y / c); //get position of the cell current point is located at
      int col = floor(keyPoints[i].pt.x / c);
      if (coveredVec[row][col] == false)
      { // if the cell is not covered
        result.push_back(i);
        int rowMin = ((row - floor(width / c)) >= 0) ? (row - floor(width / c)) : 0; //get range which current radius is covering
        int rowMax = ((row + floor(width / c)) <= numCellRows) ? (row + floor(width / c)) : numCellRows;
        int colMin = ((col - floor(width / c)) >= 0) ? (col - floor(width / c)) : 0;
        int colMax = ((col + floor(width / c)) <= numCellCols) ? (col + floor(width / c)) : numCellCols;
        for (int rowToCov = rowMin; rowToCov <= rowMax; ++rowToCov)
        {
          for (int colToCov = colMin; colToCov <= colMax; ++colToCov)
          {
            if (!coveredVec[rowToCov][colToCov])
              coveredVec[rowToCov][colToCov] = true; //cover cells within the square bounding box with width w
          }
        }
      }
    }

    if (result.size() >= Kmin && result.size() <= Kmax)
    { //solution found
      ResultVec = result;
      complete = true;
    }
    else if (result.size() < Kmin)
      high = width - 1; //update binary search range
    else
      low = width + 1;
    prevWidth = width;
  }
  // retrieve final keypoints
  std::vector<cv::KeyPoint> kp;
  for (unsigned int i = 0; i < ResultVec.size(); i++)
    kp.push_back(keyPoints[ResultVec[i]]);

  return kp;
}


void featureTracking(cv::Mat img_1, cv::Mat img_2, Frame::Frame &prev_frame, Frame::Frame &curr_frame)
{

  //this function automatically gets rid of points for which tracking fails
  // points2_map.clear();
  // vector <pair<int,Point2f>>().swap(points2_map);
  // std::vector<std::pair<int, std::pair<int, cv::Point2f>>> points2_map_tmp;
  std::vector<float> err;
  cv::Size winSize = cv::Size(21, 21);
  cv::TermCriteria termcrit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 40, 0.01);
  std::vector<uchar> status;

  std::vector<cv::Point2f> curr_frame_2d_points;

  cv::calcOpticalFlowPyrLK(img_1, img_2, prev_frame.get_2d_points(), curr_frame_2d_points, status, err, winSize, 3, termcrit, 0, 0.001);
  
  // std::cout << "befroe tracking prev: " << prev_frame.get_2d_points().size() << "\n";
  // std::cout << "bf curr: " << curr_frame_2d_points.size() << "\n";

  curr_frame.set_points_2d(curr_frame_2d_points);
  // std::cout << curr_frame_2d_points.size() << "\n";
  // std::cout << curr_frame.get_2d_points().size() << "\n";

  // for (int i = 0; i < status.size(); i++)
  //   curr_frame.get_2d_points_with_id().push_back(std::make_pair(std::make_pair(prev_frame.get_2d_points_with_id()[i].first.first, prev_frame.get_2d_points_with_id()[i].first.second), curr_frame.get_2d_points().at(i)));

  curr_frame.set_points_2d_with_id(prev_frame);
  // std::cout << curr_frame.get_2d_points_with_id().size() << "\n";

  

  erase_not_tracked_points(prev_frame, curr_frame, status);

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  
}

void erase_not_tracked_points(Frame::Frame &prev_frame, Frame::Frame &curr_frame, std::vector<uchar> status)
{
    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++)
    {
        cv::Point2f pt = curr_frame.get_2d_points().at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0))
        {
            if ((pt.x < 0) || (pt.y < 0))
                status.at(i) = 0;

        prev_frame.erase_points_with_index(i - indexCorrection);
        curr_frame.erase_points_with_index(i - indexCorrection);

        indexCorrection++;
        }
    }
}





void get_pose_from_essential_mat(Frame::Frame &prev_frame, Frame::Frame &curr_frame, const cv::Mat &intrinsic_param)
{
  cv::Mat R, t, E, mask;

  double focal = intrinsic_param.at<double>(0,0);
  cv::Point2d pp(intrinsic_param.at<double>(0,2), intrinsic_param.at<double>(1,2));

  // std::cout << curr_frame.get_2d_points().size() << "\n";
  // std::cout << prev_frame.get_2d_points().size() << "\n";


  E = findEssentialMat(curr_frame.get_good_points_2d(), prev_frame.get_good_points_2d(), focal, pp, cv::RANSAC, 0.999, 1.0, mask);
  recoverPose(E, curr_frame.get_good_points_2d(), prev_frame.get_good_points_2d(), R, t, focal, pp, mask);

  // std::cout << prev_frame.get_rotation_mat().type() << "\n";
  // std::cout << R.type() << "\n";


  curr_frame.set_camera_pose(prev_frame, R, t);

}


void vis_frame_2d_points(Frame::Frame &frame, cv::Mat image_c,bool stop)
{
  const float r = 5;
    for (int i=0;i<frame.get_2d_points().size();++i)
    {
        cv::Point2f pt1,pt2;
        // prev_frame.get_2d_points()[i].x
        pt1.x=frame.get_2d_points()[i].x-r;
        pt1.y=frame.get_2d_points()[i].y-r;
        pt2.x=frame.get_2d_points()[i].x+r;
        pt2.y=frame.get_2d_points()[i].y+r;

        cv::rectangle(image_c,pt1,pt2,cv::Scalar(0,255,0));
        cv::circle(image_c,frame.get_2d_points()[i],2,cv::Scalar(0,255,0),-1);
    }
    if (stop==true)
    {
      cv::imshow("image1_c", image_c);
      cv::waitKey();
    }
    else
    {
      cv::imshow("image1_c", image_c);
      cv::waitKey(1);
    }

}








}