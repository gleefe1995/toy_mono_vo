#include "KeyFrame.h"
#include "initialize.h"


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



    return 0;
}