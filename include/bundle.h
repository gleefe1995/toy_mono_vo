#pragma once
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "ceres/ceres.h"
#include "map_database.h"
#include "Frame.h"

namespace bundle{

struct SnavelyReprojectionError
{
  SnavelyReprojectionError(double observed_x, double observed_y, Eigen::Vector4d point_3d_homo_eig, double focal, double ppx, double ppy)
      : observed_x(observed_x), observed_y(observed_y), point_3d_homo_eig(point_3d_homo_eig), focal(focal), ppx(ppx), ppy(ppy) {}

  template <typename T>
  bool operator()(const T *const rvec_eig,
                  const T *const tvec_eig,
                  T *residuals) const
  {
    // camera[0,1,2] are the angle-axis rotation.

    // const T theta = sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2]);
    const T theta = ceres::sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2] + 1e-8);
    // std::cout << theta << "\n";
    const T tvec_eig_0 = tvec_eig[0];
    const T tvec_eig_1 = tvec_eig[1];
    const T tvec_eig_2 = tvec_eig[2];

    const T w1 = rvec_eig[0] / theta;
    const T w2 = rvec_eig[1] / theta;
    const T w3 = rvec_eig[2] / theta;

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    // Eigen::Matrix<T,3,3> R_solve_homo;
    // R_solve_homo << cos+w1*w1*(1-cos), w1*w2*(1-cos)-w3*sin, w1*w3*(1-cos)+w2*sin,
    //                 w1*w2*(1-cos)+w3*sin, cos+w2*w2*(1-cos), w2*w3*(1-cos)-w1*sin,
    //                 w1*w3*(1-cos)-w2*sin, w2*w3*(1-cos)+w1*sin, cos+w3*w3*(1-cos);

    Eigen::Matrix<T, 3, 4> Relative_homo_R;
    Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;

    Eigen::Matrix<T, 3, 1> three_to_p_eig;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;
    // Kd = Kd.cast<T>();

    //Eigen::Matrix<T,4,1> point_3d_homo;
    //point_3d_homo_eig = point_3d_homo_eig.cast<T>();
    //point_3d_homo<<point_3d_homo_eig[0],point_3d_homo_eig[1],point_3d_homo_eig[2],point_3d_homo_eig[3];

    three_to_p_eig = Kd.cast<T>() * Relative_homo_R * point_3d_homo_eig.cast<T>();
    // cv2eigen(three_to_p,three_to_p_eig);

    // three_to_p_eig[0]=three_to_p.at<double>(0);
    // three_to_p_eig[1]=three_to_p.at<double>(1);
    // three_to_p_eig[2]=three_to_p.at<double>(2);

    T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
    T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  // static ceres::CostFunction* Create(const double observed_x,
  //                                    const double observed_y,
  //                                    const Eigen::Vector4d point_3d_homo_eig,
  //                                    const double focal,
  //                                    const double ppx,
  //                                    const double ppy) {
  //   return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3,3>(
  //       new SnavelyReprojectionError(observed_x, observed_y,point_3d_homo_eig,focal,ppx,ppy)));
  // }

  double observed_x;
  double observed_y;
  const Eigen::Vector4d point_3d_homo_eig;
  double focal;
  double ppx;
  double ppy;
};



void motion_only_BA(cv::Mat &rvec, cv::Mat &tvec, std::vector<cv::Point2d> &corr_2d_pointd, 
                    std::vector<cv::Point3d> &corr_3d_point, const double focal, cv::Point2d pp)
{
    Eigen::Vector3d rvec_eig;
    Eigen::Vector3d tvec_eig;
    rvec_eig[0]=rvec.at<double>(0);
    rvec_eig[1]=rvec.at<double>(1);
    rvec_eig[2]=rvec.at<double>(2);
    
    tvec_eig[0]=tvec.at<double>(0);
    tvec_eig[1]=tvec.at<double>(1);
    tvec_eig[2]=tvec.at<double>(2);
    
    Eigen::MatrixXd corr_2d_point_eig(2,corr_2d_pointd.size());

    for (int i=0;i<corr_2d_pointd.size();i++){
      corr_2d_point_eig(0,i)=corr_2d_pointd[i].x;
      corr_2d_point_eig(1,i)=corr_2d_pointd[i].y;
    }
    

    Eigen::MatrixXd corr_3d_point_eig(4,corr_3d_point.size());

    for (int i=0;i<corr_3d_point.size();i++){
      corr_3d_point_eig(0,i)=corr_3d_point[i].x;
      corr_3d_point_eig(1,i)=corr_3d_point[i].y;
      corr_3d_point_eig(2,i)=corr_3d_point[i].z;
      corr_3d_point_eig(3,i)=1;
    }
    
    ceres::Problem problem;

    for (int i = 0; i < corr_3d_point.size(); i++) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.

    
    ceres::CostFunction* cost_function = 
          new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 3,3>(
            new SnavelyReprojectionError(corr_2d_point_eig(0,i), corr_2d_point_eig(1,i),corr_3d_point_eig.col(i),focal,pp.x,pp.y)
            );

    problem.AddResidualBlock(cost_function,
                             NULL ,
                             rvec_eig.data(),
                             tvec_eig.data());

  }
  // cv::waitKey();

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 12;
  options.max_num_iterations=100;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  //std::cout << summary.FullReport() << "\n";

 

   for (int i=0;i<3;i++){
      rvec.at<double>(i)=double(rvec_eig[i]);
      tvec.at<double>(i)=double(tvec_eig[i]);
    }
}


struct SnavelyReprojectionError_Local
{
  SnavelyReprojectionError_Local(double observed_x, double observed_y, double focal, double ppx, double ppy, int j, Eigen::VectorXi number_of_3d_points_eig)
      : observed_x(observed_x), observed_y(observed_y), focal(focal), ppx(ppx), ppy(ppy), j(j), number_of_3d_points_eig(number_of_3d_points_eig) {}

  template <typename T>
  bool operator()(const T *const rvec_eig,
                  const T *const tvec_eig,
                  const T *const point_3d_homo_eig,
                  T *residuals) const
  {
    // camera[0,1,2] are the angle-axis rotation.

    int number_of_3d_points = 0;

    int count = 0;
    

    const T theta = ceres::sqrt(rvec_eig[0] * rvec_eig[0] + rvec_eig[1] * rvec_eig[1] + rvec_eig[2] * rvec_eig[2] + 1e-8);
    // std::cout << theta << "\n";
    const T tvec_eig_0 = tvec_eig[0];
    const T tvec_eig_1 = tvec_eig[1];
    const T tvec_eig_2 = tvec_eig[2];

    const T w1 = rvec_eig[0] / theta;
    const T w2 = rvec_eig[1] / theta;
    const T w3 = rvec_eig[2] / theta;

    const T cos = ceres::cos(theta);
    const T sin = ceres::sin(theta);

    

    // Eigen::Matrix<T,3,3> R_solve_homo;
    // R_solve_homo << cos+w1*w1*(1-cos), w1*w2*(1-cos)-w3*sin, w1*w3*(1-cos)+w2*sin,
    //                 w1*w2*(1-cos)+w3*sin, cos+w2*w2*(1-cos), w2*w3*(1-cos)-w1*sin,
    //                 w1*w3*(1-cos)-w2*sin, w2*w3*(1-cos)+w1*sin, cos+w3*w3*(1-cos);

    Eigen::Matrix<T, 3, 4> Relative_homo_R;
    Relative_homo_R << cos + w1 * w1 * (static_cast<T>(1) - cos), w1 * w2 * (static_cast<T>(1) - cos) - w3 * sin, w1 * w3 * (static_cast<T>(1) - cos) + w2 * sin, tvec_eig_0,
        w1 * w2 * (static_cast<T>(1) - cos) + w3 * sin, cos + w2 * w2 * (static_cast<T>(1) - cos), w2 * w3 * (static_cast<T>(1) - cos) - w1 * sin, tvec_eig_1,
        w1 * w3 * (static_cast<T>(1) - cos) - w2 * sin, w2 * w3 * (static_cast<T>(1) - cos) + w1 * sin, cos + w3 * w3 * (static_cast<T>(1) - cos), tvec_eig_2;
    
    Eigen::Matrix<T, 1, 3> three_to_p_eig;

    Eigen::Matrix<double, 3, 3> Kd;
    Kd << focal, 0, ppx,
        0, focal, ppy,
        0, 0, 1;
    // Kd = Kd.cast<T>();

    //Eigen::Matrix<T,4,1> point_3d_homo;
    //point_3d_homo_eig = point_3d_homo_eig.cast<T>();
    //point_3d_homo<<point_3d_homo_eig[0],point_3d_homo_eig[1],point_3d_homo_eig[2],point_3d_homo_eig[3];

    Eigen::Matrix<T, 4, 1> p3he(point_3d_homo_eig[0], point_3d_homo_eig[1], point_3d_homo_eig[2], static_cast<T>(1));
    // std::cout<<p3he(0,0)<<"\n";
    // std::cout<<p3he(1,0)<<"\n";
    // std::cout<<p3he(2,0)<<"\n";
    // std::cout<<p3he(3,0)<<"\n";

    three_to_p_eig = Kd.cast<T>() * Relative_homo_R * p3he;

    // cv2eigen(three_to_p,three_to_p_eig);
    //cout<<p3he<<"\n";
    // waitKey();
    //cout<<Kd.cast<T>()<<"\n";

      // std::cout<<three_to_p_eig<<"\n";

    // std::cout<<Relative_homo_R(0,3)<<"\n";
    // std::cout<<Relative_homo_R(1,3)<<"\n";
    // std::cout<<Relative_homo_R(2,3)<<"\n";

    // three_to_p_eig[0]=three_to_p.at<double>(0);
    // three_to_p_eig[1]=three_to_p.at<double>(1);
    // three_to_p_eig[2]=three_to_p.at<double>(2);

    T predicted_x = (three_to_p_eig[0] / three_to_p_eig[2]);
    T predicted_y = (three_to_p_eig[1] / three_to_p_eig[2]);
    // std::cout << three_to_p_eig[2] << "\n";

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);

    // std::cout << predicted_x << " " << observed_x << "\n";
    // std::cout << predicted_y << " " << observed_y << "\n";
    // std::cout << predicted_x << "\n";
    // std::cout << predicted_y << "\n";

    // std::cout<<residuals[0]<<"\n";
    // std::cout<<residuals[1]<<"\n";
    // cv::waitKey();
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  // static ceres::CostFunction* Create(const double observed_x,
  //                                    const double observed_y,
  //                                    const double focal,
  //                                    const double ppx,
  //                                    const double ppy) {
  //   return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local, 2, 3,3,4>(
  //       new SnavelyReprojectionError_Local(observed_x, observed_y,focal,ppx,ppy)));
  // }

  double observed_x;
  double observed_y;
  int j;
  double focal;
  double ppx;
  double ppy;
  Eigen::VectorXi number_of_3d_points_eig;
};




bool compare_point (std::pair<std::pair<int,int>,cv::Point3d> a,
                    std::pair<std::pair<int,int>,cv::Point3d> b){
                      if(a.first.first == b.first.first){
                        return a.first.second<b.first.second;
                      }
                        return a.first.first < b.first.first;
                    }




void localBA(map_database::map_database &map_data, const cv::Mat &intrinsic_param, std::vector<Frame::point_3d_pair> &point_3d_map, cv::Mat &rvec, cv::Mat &tvec)
{
  const double focal = intrinsic_param.at<double>(0,0);
  cv::Point2d pp(intrinsic_param.at<double>(0,2),intrinsic_param.at<double>(1,2));

  std::vector<Frame::point_3d_pair> local_ba_3d_points_unique = map_data.get_local_ba_points_3d_with_id();
                
  sort(local_ba_3d_points_unique.begin(),local_ba_3d_points_unique.end(), compare_point);
  
  local_ba_3d_points_unique.erase(unique(local_ba_3d_points_unique.begin(),local_ba_3d_points_unique.end()),local_ba_3d_points_unique.end());

  std::vector<int> local_ba_points_id;
  for (int i=0;i<local_ba_3d_points_unique.size();i++){
      //cout<<i<<" "<<BA_3d_points_map_tmp[i].first<<" "<<BA_3d_points_map_tmp[i].second.first<<"\n";
      local_ba_points_id.push_back(10000*local_ba_3d_points_unique[i].first.first+local_ba_3d_points_unique[i].first.second);
  }
  
  std::vector<cv::Point3d> rvec_vec = map_data.get_local_ba_rvec();
  std::vector<cv::Point3d> tvec_vec = map_data.get_local_ba_tvec();
  std::vector<int32_t> number_of_3d_points = map_data.get_local_ba_number_of_points();
  std::vector<Frame::point_2d_pair> BA_2d_points_map = map_data.get_local_ba_points_2d_with_id();
  std::vector<Frame::point_3d_pair> BA_3d_points_map = map_data.get_local_ba_points_3d_with_id();
  std::vector<int32_t> BA_3d_map_points = local_ba_points_id;

  // std::cout << rvec_vec.size() << "\n";
  // std::cout << tvec_vec.size() << "\n";
  // for (int i=0;i<number_of_3d_points.size();++i)
  //   std::cout << number_of_3d_points[i] << "\n";
  // std::cout << BA_2d_points_map.size() << "\n";
  // std::cout << BA_3d_points_map.size() << "\n";
  // std::cout << local_ba_3d_points_unique.size() << "\n";
  // cv::waitKey();

  // for (int i=0;i<BA_3d_points_map.size();++i)
  // {
  //   std::cout << BA_2d_points_map[i].second << " " << BA_3d_points_map[i].second << "\n";
  // }
  // cv::waitKey();
  // std::cout << BA_2d_points_map.size() << "\n";
  // std::cout << BA_3d_points_map.size() << "\n";

  // cv::waitKey();



    int rvec_eig_local_size=rvec_vec.size();
   
    Eigen::MatrixXd rvec_eig_local(3,rvec_eig_local_size);
    Eigen::MatrixXd tvec_eig_local(3,rvec_eig_local_size);


    for (int i=0; i<rvec_eig_local_size;i++){
      rvec_eig_local(0,i)=rvec_vec[i].x;
      rvec_eig_local(1,i)=rvec_vec[i].y;
      rvec_eig_local(2,i)=rvec_vec[i].z;
      tvec_eig_local(0,i)=tvec_vec[i].x;
      tvec_eig_local(1,i)=tvec_vec[i].y;
      tvec_eig_local(2,i)=tvec_vec[i].z;
    }
    
   
    Eigen::VectorXd BA_2d_points_eig(2);
    Eigen::MatrixXd BA_3d_points_eig(3,local_ba_3d_points_unique.size());
    Eigen::VectorXi number_of_3d_points_eig(number_of_3d_points.size());
   
    for (int i=0;i<local_ba_3d_points_unique.size();i++){
      BA_3d_points_eig(0,i)=local_ba_3d_points_unique[i].second.x;
      BA_3d_points_eig(1,i)=local_ba_3d_points_unique[i].second.y;
      BA_3d_points_eig(2,i)=local_ba_3d_points_unique[i].second.z;
    }
    
    
    
    ceres::Problem problem2;
    
    int index_vec=0;
    
    //cout<<half_3d_points<<"\n";
    //-number_of_3d_points[local_ba_frame-1]
    for (int i = 0; i < BA_2d_points_map.size(); i++) {
          
          //cout<<BA_2d_points_map.at(i).first*10000+BA_2d_points_map.at(i).second.first<<"\n";
          auto it =std::find(BA_3d_map_points.begin(), BA_3d_map_points.end(), BA_2d_points_map.at(i).first.first*10000+BA_2d_points_map.at(i).first.second);
          
          //auto it = BA_3d_map_points.find(BA_2d_points_map[j].at(i).first*1000+BA_2d_points_map[j].at(i).second.first);
          BA_2d_points_eig[0]=(double)BA_2d_points_map.at(i).second.x;
          BA_2d_points_eig[1]=(double)BA_2d_points_map.at(i).second.y;
          

          int index_vec_num=0;
          for (int j = 0; j < number_of_3d_points.size(); j++)
            {
              index_vec_num += number_of_3d_points[j];
              if (i < index_vec_num)
              {
                index_vec = j;
                break;
              }
            }
          //cout<<it-BA_3d_map_points.begin()<<"\n";
          
    //       if (i<half_3d_points){
    //             // cout<<i<<"\n";
    //           ceres::CostFunction* cost_function2 = 
    //       new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local_pose_fixed, 2,3>(
    //         new SnavelyReprojectionError_Local_pose_fixed(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,i,number_of_3d_points_eig,rvec_eig_local.col(index_vec),tvec_eig_local.col(index_vec))
    //       );
       
    
    // problem2.AddResidualBlock(cost_function2,
    //                          NULL ,
    //                          BA_3d_points_eig.col(it-BA_3d_map_points.begin()).data());

    //       }
    //       else{
           ceres::CostFunction* cost_function2 = 
          new ceres::AutoDiffCostFunction<SnavelyReprojectionError_Local, 2, 3,3,3>(
            new SnavelyReprojectionError_Local(BA_2d_points_eig[0],BA_2d_points_eig[1],focal,pp.x,pp.y,i,number_of_3d_points_eig)
           );
       
    
    problem2.AddResidualBlock(cost_function2,
                             NULL ,
                             rvec_eig_local.col(index_vec).data(),
                             tvec_eig_local.col(index_vec).data(),
                             BA_3d_points_eig.col(it-BA_3d_map_points.begin()).data());
          //}
      
      }
      
    //   cout<<"BA_3d_points_map_tmp size: "<<BA_3d_points_map_tmp.size()<<"\n";
    //  cv::waitKey();
      
  //cout<<"local BA solver start"<<"\n";
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 12;
  options.max_num_iterations=100;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem2, &summary);
  
  // cv::waitKey();
  
  rvec_vec.clear();
  std::vector<cv::Point3d>().swap(rvec_vec);
  tvec_vec.clear();
  std::vector<cv::Point3d>().swap(tvec_vec);
   for (int i=0;i<rvec_eig_local_size;i++){
     double rvec_eig_1=rvec_eig_local(0,i);
     double rvec_eig_2=rvec_eig_local(1,i);
     double rvec_eig_3=rvec_eig_local(2,i);
     double tvec_eig_1=tvec_eig_local(0,i);
     double tvec_eig_2=tvec_eig_local(1,i);
     double tvec_eig_3=tvec_eig_local(2,i);
     
     
      rvec_vec.push_back(cv::Point3d(rvec_eig_1,rvec_eig_2,rvec_eig_3));
      tvec_vec.push_back(cv::Point3d(tvec_eig_1,tvec_eig_2,tvec_eig_3));
      
    }
    
    
    int BA_3d_points_map_size=BA_3d_points_map.size();
    // vector <pair<int,pair<int,Point3d>>>().swap(BA_3d_points_map);
    int point_3d_map_size=point_3d_map.size();
  for (int i=0;i<BA_3d_points_map_size;i++){
      int map_first=BA_3d_points_map[i].first.first;
      int map_second_first=BA_3d_points_map[i].first.second;
      
      auto it =std::find(BA_3d_map_points.begin(), BA_3d_map_points.end(), map_first*10000+map_second_first);
      if (it==BA_3d_map_points.end()){
            std::cout<<"fail"<<"\n";
            
            cv::waitKey();
          }
      int eig_index=it-BA_3d_map_points.begin();
      BA_3d_points_map.push_back(std::make_pair(BA_3d_points_map[i].first, cv::Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index))));
      
      if (i>=BA_3d_points_map_size-number_of_3d_points[number_of_3d_points.size()-1]){
        point_3d_map.push_back(std::make_pair(BA_3d_points_map[i].first, cv::Point3d(BA_3d_points_eig(0,eig_index),BA_3d_points_eig(1,eig_index),BA_3d_points_eig(2,eig_index))));
      }
    }
    
BA_3d_points_map.erase(BA_3d_points_map.begin(),BA_3d_points_map.begin()+BA_3d_points_map_size);
point_3d_map.erase(point_3d_map.begin(),point_3d_map.begin()+point_3d_map_size);

  

  rvec.at<double>(0)=rvec_vec[rvec_eig_local_size-1].x;
  rvec.at<double>(1)=rvec_vec[rvec_eig_local_size-1].y;
  rvec.at<double>(2)=rvec_vec[rvec_eig_local_size-1].z;

  tvec.at<double>(0)=tvec_vec[rvec_eig_local_size-1].x;
  tvec.at<double>(1)=tvec_vec[rvec_eig_local_size-1].y;
  tvec.at<double>(2)=tvec_vec[rvec_eig_local_size-1].z;



map_data.set_local_ba_3d_points_and_pose(rvec_vec, tvec_vec, BA_3d_points_map);


}






}
//namespace