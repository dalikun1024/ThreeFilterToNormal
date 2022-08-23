#include <cstdlib>
#include <iostream>
#include <eigen3/Eigen/Core>
#include "tftn/tftn.h"
#include "cvrgbd/rgbd.hpp"
#include <chrono>


/**
  * @brief Read depth images (.bin files)
  * */
cv::Mat LoadDepthImage(const std::string &path, const size_t width = 640,
                       const size_t height = 480){
  const int buffer_size = sizeof(float) * height * width;
  //char *buffer = new char[buffer_size];

  cv::Mat mat(cv::Size(width, height), CV_32FC1);

  // open filestream && read buffer
  std::ifstream fs_bin_(path, std::ios::binary);
  fs_bin_.read(reinterpret_cast<char*>(mat.data), buffer_size);
  fs_bin_.close();
  return mat;
}

cv::Mat convertNormalToU16(const cv::Mat &normal_map) {
    cv::Mat normal_res(normal_map.size(), CV_8UC3);
    for (int r = 0; r < normal_map.rows; ++r) {
        for (int c = 0; c < normal_map.cols; ++c) {
            cv::Vec3f normal_vec = normal_map.at<cv::Vec3f>(r, c);
            float theta = atan2(normal_vec[0], normal_vec[2]);
            theta *= (180.0/3.141592653589793238463);
            float phi = asin(normal_vec[1]);
            phi *= (180.0/3.141592653589793238463);
            normal_res.at<cv::Vec3b>(r,c) = cv::Vec3b(round(theta) + 128, round(phi) + 128, 0);
        }
    }
    return normal_res;
}

int saveToFile(std::string save_path, const cv::Mat &normal_res) {
  std::ofstream normal_bin_stream;
  normal_bin_stream.open(save_path, std::ofstream::binary);
  if (normal_bin_stream.good()) {
    normal_bin_stream.write((char*)normal_res.data, normal_res.cols*normal_res.rows*normal_res.channels()*sizeof(float));
  }
  normal_bin_stream.close();
  return 0;
}

cv::Mat convertUVtoNormalMap(const cv::Mat &normal_uv) {
    cv::Mat normal_map(normal_uv.size(), CV_32FC3);
    for (int r = 0; r < normal_uv.rows; ++r) {
        for (int c = 0; c < normal_uv.cols; ++c) {
            auto uv = normal_uv.at<cv::Vec3b>(r, c);
            char ch;
            float theta = char(uv[0]-128) * (3.141592653589793238463/180.0);
            float phi = char(uv[1]-128) * (3.141592653589793238463/180.0);
            float y = sin(phi);
            float z = cos(theta) * cos(phi);
            float x = sin(theta) * cos(phi);
            normal_map.at<cv::Vec3f>(r,c) = cv::Vec3f(z, y, x);
        }
    }
    cv::Mat normal_show;
    normal_map.convertTo(normal_show, CV_8UC3, 128, 128);
    return normal_show;
}

int main(){
  int n; //the number of depth images.
  std::string param = "../data/android/params_2.txt";
  FILE *f = fopen(param.c_str(), "r");

  cv::Matx33d camera(0,0,0,0,0,0,0,0,1);
  fscanf(f, "%lf %lf %lf %lf %d", &camera(0,0),
         &camera(1,1), &camera(0, 2), &camera(1,2), &n);
  camera(0,2)--;  camera(1,2)--;

  // auto depth_image = LoadDepthImage("../data/android/depth/000001.bin", 640, 480);
  cv::Mat depth_image = cv::imread("/mnt/hgfs/vmshare/depth_filtered/demo_006-depth_raw_gauss.png", cv::IMREAD_UNCHANGED);
  std::cout << "xxxxxxxxxxxxxxxxxxxxxxxxxx" << std::endl;
  cv::imshow("depth", depth_image);
  // depth_image.convertTo(depth_image, CV_32FC1, -1);
  double min, max;
  cv::minMaxIdx(depth_image, &min, &max);
  std::cout << min << " " << max << std::endl;
  std::cout << depth_image.type() << std::endl;
  std::cout << depth_image.channels() << std::endl;
  cv::Mat_<float> s(depth_image);
  for (auto &it : s){
    if (fabs(it) < 1e-7){ //If the value equals 0, the point is infinite
      it = 1e10;
    }
  }

  //convert depth image to range image. watch out the problem of bgr and rgb.
  cv::Mat range_image, result;
  cv::rgbd::depthTo3d(depth_image, camera, range_image);
  std::vector<cv::Mat> matpart(3);
  cv::split(range_image, matpart);
  result.create(matpart[0].rows, matpart[0].cols, CV_32FC3);

  /*******************the core code*********************/
  auto t1 = std::chrono::steady_clock::now();
  TFTN(range_image, camera, R_MEANS_4, &result);
  std::chrono::duration<double> dur = std::chrono::steady_clock::now() - t1;
  std::cout << "process time: " << dur.count() << std::endl;
  /*****************************************/
  cv::Mat showNormal;
  cv::Mat output;
  for (int i = 0; i < result.rows; ++ i){
    for (int j = 0; j < result.cols; ++ j){
      result.at<cv::Vec3f>(i, j) = result.at<cv::Vec3f>(i, j) / cv::norm(result.at<cv::Vec3f>(i, j));
      if (result.at<cv::Vec3f>(i, j)[2] > 0) {
        std::cout << "z: " << result.at<cv::Vec3f>(i, j)[2] << " larger than 0" << std::endl;
        result.at<cv::Vec3f>(i, j)[2]  = -result.at<cv::Vec3f>(i, j)[2];
      }
      result.at<cv::Vec3f>(i,j)[1] *= -1;
      result.at<cv::Vec3f>(i,j)[2] *= -1;
    }
  }
  result.convertTo(showNormal, CV_8UC3, 128.0, 128);
  cv::cvtColor(showNormal, showNormal, cv::COLOR_RGB2BGR);
  cv::imshow("normal",showNormal);

  cv::Mat uv_out = convertNormalToU16(result);
  cv::Mat recovered_normal = convertUVtoNormalMap(uv_out);
  
  saveToFile("/mnt/hgfs/vmshare/depth_filtered/006_normal.bin", result);
  cv::imshow("result", uv_out);
  cv::imshow("recover", recovered_normal);
  cv::imwrite("/mnt/hgfs/vmshare/depth_filtered/006_normal.png", showNormal);
  cv::imwrite("/mnt/hgfs/vmshare/depth_filtered/006_normal_uv.png", uv_out);
  cv::imwrite("/mnt/hgfs/vmshare/depth_filtered/006_normal_recover.png", recovered_normal);
  cv::waitKey();
  return 0;
}

