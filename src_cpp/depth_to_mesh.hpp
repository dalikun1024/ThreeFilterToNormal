#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class Edge {
public:
    Edge():id_a(-1), id_b(-1), face_id(-1), splited(false), center_id(-1) {}
    Edge(int32_t id_a, int32_t id_b, int32_t face = -1, int32_t center_id = -1):
        id_a(id_a), id_b(id_b), face_id(face), 
        center_id(center_id), splited(center_id!=-1) {}
    std::string getKey() {
        std::string key;
        if (id_a < id_b) {
        key = std::to_string(id_a) + "_" + std::to_string(id_b);
    } else {
        key = std::to_string(id_b) + "_" + std::to_string(id_a);
    }
    return key;
    }
public:
    int32_t id_a;
    int32_t id_b;
    int32_t face_id;
    bool splited;
    int32_t center_id;
};

class depthToMesh {
public:
    depthToMesh(const cv::Mat &depth, float fov);
    int generateMesh();
    int generateMesh_v2();
    int trianglelate(std::vector<int> grib_coor);
    int saveToObj(std::string obj_path);
private:
    bool needSplit(int idx[4]);
    int generateTriangles(int idx[4], float dx, float dy);
    bool repairGap(std::vector<int> grid_coor);
    int trianglelation(std::vector<Eigen::Vector2f>grid_coor);
    int triangleNeedSplit(int idx[3]);
    float depthDiff(int id_a, int id_b);
    float checkShape(std::vector<Eigen::Vector2f>grid_coor);
    int32_t addVertices(float u, float v);
    Eigen::Vector3f calcVertices(float u, float v);
    Eigen::Vector3f calcVertices(int u, int v);
    std::vector<Eigen::Vector2f> splitAngle(std::vector<Eigen::Vector2f> grid_coor, std::vector<int> idx);
    int addFace(std::vector<int> idx);
    bool edgeExist(int id_a, int id_b);
    int edgeErase(int id_a, int id_b);
    int addEdge(int id_a, int id_b,  int32_t face = -1, int32_t center_id = -1);
    int addEdgeRecurive(int id_a, int id_b,  int32_t face);
    int splitEdge(Edge edge, uint16_t center_id);
    int drawFace(std::vector<Eigen::Vector2f> grid_coor);
public:
    int min_dis = 2;
    float focal;
    float cx, cy;
    cv::Mat depth;
    cv::Mat index_map;
    float diff_threshold = 10;
    float area_threshold = 8;
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector2i> uv;
    std::vector<uint16_t> faces;
    std::unordered_map<std::string, Edge> edges;
    cv::Mat triangles_map;
};

