#include "depth_to_mesh.hpp"
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    std::string depth_map_path(argv[1]);

    cv::Mat depth_mat = cv::imread(depth_map_path, cv::IMREAD_UNCHANGED);

    std::cout << "type: " << depth_mat.type() << " " << depth_mat.depth() << std::endl;

    

    depthToMesh depth_to_mesh(depth_mat, 60.0);

    depth_to_mesh.generateMesh_v2();

    depth_to_mesh.saveToObj("output.obj");

    return 0;
}