#include "depth_to_mesh.hpp"
#include <fstream>
#include <chrono>

depthToMesh::depthToMesh(const cv::Mat &depth_data, float fov) {
    depth = depth_data.clone();
    index_map = cv::Mat(depth_data.size(), CV_32SC1, -1);
    int height = depth.rows;
    int width = depth.cols;
    focal = std::max(height, width) * 0.5 / tan(fov*0.5*3.1415926/180.0);
    cx = width * 0.5;
    cy = height * 0.5;
    double min, max;
    cv::minMaxLoc(depth, &min, &max);
    diff_threshold = 0.1 * (max - min);
    triangles_map = cv::Mat(depth_data.size(), CV_8UC1, 200);
}

int depthToMesh::generateMesh() {
    auto t1 = std::chrono::steady_clock::now();
    int height = depth.rows;
    int width = depth.cols;
    int step_h = 0.1 * height;
    int step_w = 0.1 * width;
    int grib_row = (height-1) / step_h;
    int grib_col = (width-1) / step_w;
    for (int r = 0; r < grib_row; r++) {
        for (int c = 0; c < grib_col; c++) {
            std::vector<int> grid_coor = { r*step_h,     c*step_w, 
                                          (r+1)*step_h, (c+1)*step_w };
            grid_coor[2] = grid_coor[2] >= height ? height-1 : grid_coor[2];
            grid_coor[3] = grid_coor[3] >= width ? width-1 : grid_coor[3];
            trianglelate(grid_coor);
        }
    }
    std::chrono::duration<double> dur = std::chrono::steady_clock::now() - t1;
    std::cout << "process time: " << dur.count() << std::endl;
    return 0;
}

int depthToMesh::trianglelate(std::vector<int> grid_coor) {
    if (grid_coor[0] == grid_coor[2] || grid_coor[1] == grid_coor[3]) {
        return 0;
    }
    int idx[4] = {0};
    
    int r_[4] = {grid_coor[0], grid_coor[2], grid_coor[2], grid_coor[0]};
    int c_[4] = {grid_coor[1], grid_coor[1], grid_coor[3], grid_coor[3]};
    for (int i = 0; i < 4; ++i) {
        int r = r_[i];
        int c = c_[i];
        if (index_map.ptr<int32_t>(r)[c] == -1) {
            vertices.push_back(calcVertices(c,r));
            idx[i] = vertices.size()-1;
            index_map.ptr<int32_t>(r)[c] = idx[i];
        } else {
            idx[i] = index_map.ptr<int32_t>(r)[c];
        }
    }
    if ((grid_coor[2]-grid_coor[0] <= min_dis && grid_coor[3]-grid_coor[1] <= min_dis) || !needSplit(idx)) {
        generateTriangles(idx, grid_coor[3]-grid_coor[1], grid_coor[2]-grid_coor[0]);
    } else {
        int center_r = (grid_coor[0] + grid_coor[2]) / 2;
        int center_c = (grid_coor[1] + grid_coor[3]) / 2;
        trianglelate({grid_coor[0], grid_coor[1], center_r, center_c});
        trianglelate({grid_coor[0], center_c, center_r, grid_coor[3]});
        trianglelate({center_r, grid_coor[1], grid_coor[2], center_c});
        trianglelate({center_r, center_c, grid_coor[2], grid_coor[3]});
        // repairGap(grid_coor);
    }
    return 0;
}

bool depthToMesh::repairGap(std::vector<int> grid_coor) {
    int center_r = (grid_coor[0] + grid_coor[2]) / 2;
    int center_c = (grid_coor[1] + grid_coor[3]) / 2;
    std::vector<int> r_ = {center_r, center_r, grid_coor[0], grid_coor[2]}; // left , right , up, bottom
    std::vector<int> c_ = {grid_coor[1], grid_coor[3], center_c, center_c};
    int lt_id = index_map.ptr<int32_t>(grid_coor[0])[grid_coor[1]];
    int rb_id = index_map.ptr<int32_t>(grid_coor[2])[grid_coor[3]];
    int lb_id = index_map.ptr<int32_t>(grid_coor[2])[grid_coor[1]];
    int rt_id = index_map.ptr<int32_t>(grid_coor[0])[grid_coor[3]];
    // left
        int l_id = index_map.ptr<int32_t>(r_[0])[c_[0]];
        faces.push_back(lt_id);
        faces.push_back(lb_id);
        faces.push_back(l_id);
    
    // right
        int r_id = index_map.ptr<int32_t>(r_[1])[c_[1]];
        faces.push_back(rt_id);
        faces.push_back(r_id);
        faces.push_back(rb_id);
    // up
        int u_id = index_map.ptr<int32_t>(r_[2])[c_[2]];
        faces.push_back(lt_id);
        faces.push_back(u_id);
        faces.push_back(rt_id);
    // bottom
        int b_id = index_map.ptr<int32_t>(r_[3])[c_[3]];
        faces.push_back(lb_id);
        faces.push_back(rb_id);
        faces.push_back(b_id);
    
    return 0;
}

bool depthToMesh::needSplit(int idx[4]) {
    //just compare depth for four corner
    float min_d = vertices[idx[0]].z();
    float max_d = min_d;
    for (int i = 1; i < 4; ++i) {
        float depth = vertices[idx[i]].z();
        min_d = min_d > depth ? depth : min_d;
        max_d = max_d < depth ? depth : max_d;
    }
    if (max_d - min_d > diff_threshold) {
        return true;
    }
    return false;
}

std::vector<Eigen::Vector2f> depthToMesh::splitAngle(std::vector<Eigen::Vector2f> grid_coor, std::vector<int> idx) {
    //just compare depth
    float min_d = vertices[idx[0]].z();
    float max_d = min_d;
    for (int i = 1; i < idx.size(); ++i) {
        float depth = vertices[idx[i]].z();
        min_d = min_d > depth ? depth : min_d;
        max_d = max_d < depth ? depth : max_d;
    }
    if (max_d - min_d > diff_threshold) {
        float a_b = (grid_coor[0] - grid_coor[1]).norm();
        float a_c = (grid_coor[0] - grid_coor[2]).norm();
        float b_c = (grid_coor[1] - grid_coor[2]).norm();
        if (a_b >= a_c && a_b >= b_c) {
            Eigen::Vector2f center = (grid_coor[0] + grid_coor[1]) / 2;
            int32_t vertex_id = addVertices(center.x(), center.y());
            splitEdge(Edge(idx[0], idx[1]), vertex_id);
            return {grid_coor[0], center, grid_coor[2], center, grid_coor[1], grid_coor[2]};
        } else if (a_c >= a_b && a_c >= b_c) {
            Eigen::Vector2f center = (grid_coor[0] + grid_coor[2]) / 2;
            int32_t vertex_id = addVertices(center.x(), center.y());
            splitEdge(Edge(idx[0], idx[2]), vertex_id);
            return {grid_coor[0], grid_coor[1], center, center, grid_coor[1], grid_coor[2]};
        } else if (b_c >= a_b && b_c >= a_c) {
            Eigen::Vector2f center = (grid_coor[1] + grid_coor[2]) / 2;
            int32_t vertex_id = addVertices(center.x(), center.y());
            splitEdge(Edge(idx[1], idx[2]), vertex_id);
            return {grid_coor[0], grid_coor[1], center, grid_coor[0], center, grid_coor[2]};
        }
    }
    return {};
}

int depthToMesh::generateTriangles(int idx[4], float dx, float dy) {
    float dz_x = ((vertices[idx[3]].z() + vertices[idx[2]].z()) - (vertices[idx[0]].z() + vertices[idx[1]].z())) / dx;
    float dz_y = ((vertices[idx[1]].z() + vertices[idx[2]].z()) - (vertices[idx[0]].z() + vertices[idx[3]].z())) / dy;
    if (dz_x * dz_y <= 0) {
        faces.push_back(idx[0]);
        faces.push_back(idx[1]);
        faces.push_back(idx[2]);
        faces.push_back(idx[0]);
        faces.push_back(idx[2]);
        faces.push_back(idx[3]);
    } else {
        faces.push_back(idx[0]);
        faces.push_back(idx[1]);
        faces.push_back(idx[3]);
        faces.push_back(idx[1]);
        faces.push_back(idx[2]);
        faces.push_back(idx[3]);
    }
    return 0;

}

int depthToMesh::saveToObj(std::string obj_path) {
    std::ofstream obj_stream(obj_path);
    if (obj_stream.good()) {
        for (auto v : vertices) {
            obj_stream << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
        }
        for (auto vt : uv) {
            obj_stream << "vt " << vt.x() << " " << vt.y() << std::endl;
        }
        for (int i = 0; i < faces.size(); i+=3) {
            obj_stream << "f " << faces[i]+1 << " " << faces[i+1]+1 << " " << faces[i+2]+1 << std::endl;
        }
    }
    obj_stream.close();
    return 0;
}

float depthToMesh::depthDiff(int id_a, int id_b) {
    if (fabs(vertices[id_a].z() - vertices[id_b].z()) > diff_threshold) {
        return true;
    }
    return false;
}

Eigen::Vector3f depthToMesh::calcVertices(int u, int v) {
    float z = depth.ptr<uint16_t>(v)[u];
    if (z > 0) {
        std::cout << "u, v: " << u << " " << v << std::endl;
    }
    float vx = (u - cx) / focal * z;
    float vy = -(v - cy) / focal * z;
    float vz = -z;
    return Eigen::Vector3f(vx, vy, vz);
}

int32_t depthToMesh::addVertices(float u, float v) {
    int c = u + 0.5;
    int r = v + 0.5;
    int32_t vertex_id = index_map.ptr<int32_t>(r)[c];
    if (vertex_id == -1) {
        vertices.push_back(calcVertices(u, v));
        vertex_id = vertices.size()-1;
        index_map.ptr<int32_t>(r)[c] = vertex_id;
    } else {
        vertices[vertex_id] = calcVertices(u, v);
    }
    return vertex_id;
}

Eigen::Vector3f depthToMesh::calcVertices(float u, float v) {
    float z = depth.ptr<uint16_t>(v+0.5)[int(u+0.5)];
    float vx = (u - cx) / focal * z;
    float vy = -(v - cy) / focal * z;
    float vz = -z;
    return Eigen::Vector3f(vx, vy, vz);
}

int depthToMesh::trianglelation(std::vector<Eigen::Vector2f>grid_coor) {
    if (grid_coor.size()!=3) {return 0;}
    std::vector<int> idx(grid_coor.size(), 0);
    for (int i = 0; i < grid_coor.size(); ++i) {
        int r = grid_coor[i].y() + 0.5;
        int c = grid_coor[i].x() + 0.5;
        if (index_map.ptr<int32_t>(r)[c] == -1) {
            vertices.push_back(calcVertices(grid_coor[i].x(), grid_coor[i].y()));
            uv.push_back(Eigen::Vector2i(c, r));
            idx[i] = vertices.size()-1;
            index_map.ptr<int32_t>(r)[c] = idx[i];
        } else {
            idx[i] = index_map.ptr<int32_t>(r)[c];
            vertices[idx[i]] = calcVertices(grid_coor[i].x(), grid_coor[i].y());
        }
    }
    if (idx[0] == idx[1] || idx[0] == idx[2] || idx[1] == idx[2]) {
        return 0;
    }
    if (checkShape(grid_coor) < area_threshold) {
        // std::cout << "vertex: " << idx[0] << " " << idx[1] << " " << idx[2] << std::endl;
        faces.push_back(idx[0]); 
        faces.push_back(idx[1]);
        faces.push_back(idx[2]);
        addEdge(idx[0], idx[1], (faces.size()-1)/3);
        addEdge(idx[1], idx[2], (faces.size()-1)/3);
        addEdge(idx[0], idx[2], (faces.size()-1)/3);
        // drawFace(grid_coor);
        return 0;
    }
    // std::cout << "idx 0: " << idx[0] << " " << grid_coor[0].transpose() << std::endl;
    // std::cout << "idx 1: " << idx[1] << " " << grid_coor[1].transpose() << std::endl;
    // std::cout << "idx 2: " << idx[2] << " " << grid_coor[2].transpose() << std::endl;
    // std::cout << "shape: " << checkShape(grid_coor) << std::endl;
    // if (idx[0] == 2061) {
    //     std::cout << "xx" << std::endl;
    // }
    auto split_grid = splitAngle(grid_coor, idx);
    if (!split_grid.empty()) {
        trianglelation({split_grid[0], split_grid[1], split_grid[2]});
        trianglelation({split_grid[3], split_grid[4], split_grid[5]});
    } else {
        // std::cout << "vertex: " << idx[0] << " " << idx[1] << " " << idx[2] << std::endl;
        // faces.push_back(idx[0]);
        // faces.push_back(idx[1]);
        // faces.push_back(idx[2]);
        // addEdgeRecurive(idx[0], idx[1], (faces.size()-1)/3);
        // addEdgeRecurive(idx[1], idx[2], (faces.size()-1)/3);
        // addEdgeRecurive(idx[0], idx[2], (faces.size()-1)/3);
        addFace(idx);
        // drawFace(grid_coor);
    }
    return 0;
}

int depthToMesh::generateMesh_v2() {
    auto t1 = std::chrono::steady_clock::now();
    int height = depth.rows;
    int width = depth.cols;
    int step_h = 0.1 * height;
    int step_w = 0.1 * width;
    int grib_row = (height-1) / step_h;
    int grib_col = (width-1) / step_w;
    for (int r = 0; r < grib_row; r++) {
        for (int c = 0; c < grib_col; c++) {
            std::vector<int> grid_coor = { c*step_w,  r*step_h,      
                                          (c+1)*step_w, (r+1)*step_h };
            grid_coor[2] = grid_coor[2] >= width ? width-1 : grid_coor[2];
            grid_coor[3] = grid_coor[3] >= height ? height-1 : grid_coor[3];
            std::vector<Eigen::Vector2f> coord_lt{Eigen::Vector2f(grid_coor[0], grid_coor[1]), 
                                                  Eigen::Vector2f(grid_coor[0], grid_coor[3]), 
                                                  Eigen::Vector2f(grid_coor[2], grid_coor[3])};
            std::vector<Eigen::Vector2f> coord_rb{Eigen::Vector2f(grid_coor[0], grid_coor[1]), 
                                                  Eigen::Vector2f(grid_coor[2], grid_coor[3]), 
                                                  Eigen::Vector2f(grid_coor[2], grid_coor[1])};
            trianglelation(coord_lt);
            trianglelation(coord_rb);
        }
    }
    std::chrono::duration<double> dur = std::chrono::steady_clock::now() - t1;
    std::cout << "process time: " << dur.count() << std::endl;
    return 0;
}

float depthToMesh::checkShape(std::vector<Eigen::Vector2f>grid_coor) {
    Eigen::Vector2f a_b = grid_coor[1] - grid_coor[0];
    Eigen::Vector2f a_c = grid_coor[2] - grid_coor[0];
    return fabs(a_b.x() * a_c.y() - a_b.y() * a_c.x());
}

bool depthToMesh::edgeExist(int id_a, int id_b) {
    std::string key;
    if (id_a < id_b) {
        key = std::to_string(id_a) + "_" + std::to_string(id_b);
    } else {
        key = std::to_string(id_b) + "_" + std::to_string(id_a);
    }
    return edges.count(key) != 0;
}

int depthToMesh::edgeErase(int id_a, int id_b) {
    std::string key;
    if (id_a < id_b) {
        key = std::to_string(id_a) + "_" + std::to_string(id_b);
    } else {
        key = std::to_string(id_b) + "_" + std::to_string(id_a);
    }
    edges.erase(key);
    return 0;
}


int depthToMesh::splitEdge(Edge edge, uint16_t center_id) {
    if(!edgeExist(edge.id_a, edge.id_b)) {
        addEdge(edge.id_a, edge.id_b, -1, center_id);
        return 0;
    }
    if (edges[edge.getKey()].splited) {
        edgeErase(edge.id_a, edge.id_b);
        return 0;
    }
    int32_t face_id = edges[edge.getKey()].face_id;
    uint16_t v_a = faces[face_id * 3 + 0];
    uint16_t v_b = faces[face_id * 3 + 1];
    uint16_t v_c = faces[face_id * 3 + 2];
    for (int i = 1; i < 3; ++i) {
        if ((edge.id_a == v_a && edge.id_b == v_b) || (edge.id_b == v_a && edge.id_a == v_b)) {
            break;
        } else {
            v_a = faces[face_id * 3 + i];
            v_b = faces[face_id * 3 + (i + 1) % 3];
            v_c = faces[face_id * 3 + (i + 2) % 3];
        }
    }
    std::vector<uint16_t> face_a = {v_a, center_id, v_c};
    std::vector<uint16_t> face_b = {center_id, v_b, v_c};

    faces[face_id * 3 + 0] = face_a[0];
    faces[face_id * 3 + 1] = face_a[1];
    faces[face_id * 3 + 2] = face_a[2];
    // if (face_a[0] == 0)
    //     std::cout << "vertexx: " << face_a[0] << " " << face_a[1] << " " << face_a[2] << std::endl;
    faces.insert(faces.end(), face_b.begin(), face_b.end());
    // std::cout << "vertexx: " << face_b[0] << " " << face_b[1] << " " << face_b[2] << std::endl;
    addEdge(v_a, center_id, face_id);
    addEdge(center_id, v_b, (faces.size()-1)/3);
    edgeErase(edge.id_a, edge.id_b);
    return 0;
}

int depthToMesh::addEdge(int id_a, int id_b, int32_t face, int32_t center_id) {
    if (edgeExist(id_a, id_b)) { 
        edgeErase(id_a, id_b);
        return 0; 
    }
    std::string key;
    if (id_a < id_b) {
        key = std::to_string(id_a) + "_" + std::to_string(id_b);
    } else {
        key = std::to_string(id_b) + "_" + std::to_string(id_a);
    }
    edges[key] = Edge(id_a, id_b, face, center_id);
    return 0;
}

int depthToMesh::addFace(std::vector<int> idx) {
    if (idx[0] == 176 && idx[1]==250 && idx[2] == 328) {
        std::cout << "xxxx" << std::endl;
    }
    for (int i = 0; i < 3; ++i) {
        int id_a = idx[i];
        int id_b = idx[(i+1) % 3];
        if (edgeExist(id_a, id_b)) {
            if (edges[Edge(id_a, id_b).getKey()].splited) {
                int center_id = edges[Edge(id_a, id_b).getKey()].center_id;
                edgeErase(id_a, id_b);
                std::vector<int> new_idx = idx;
                for (int k = 0; k < 3; ++k) {
                    new_idx[k] = idx[k] == id_a ? center_id : idx[k];
                    if (idx[k] == id_b) {
                        idx[k] = center_id;
                    }
                }
                addFace(idx);
                addFace(new_idx);
                return 0;
            }
        }
    }
    faces.push_back(idx[0]);
    faces.push_back(idx[1]);
    faces.push_back(idx[2]);
    addEdge(idx[0], idx[1], (faces.size()-1)/3);
    addEdge(idx[1], idx[2], (faces.size()-1)/3);
    addEdge(idx[2], idx[0], (faces.size()-1)/3);
    return 0;
}


int depthToMesh::addEdgeRecurive(int id_a, int id_b,  int32_t face_id) {
    if (!edgeExist(id_a, id_b)) {
        addEdge(id_a, id_b, face_id);
        return 0;
    }
    if (!edges[Edge(id_a, id_b).getKey()].splited) {
        edgeErase(id_a, id_b);
        return 0;
    }
    int center_id = edges[Edge(id_a, id_b).getKey()].center_id;
    edgeErase(id_a, id_b);
    int new_face[3] = {0};
    for (int i = 0; i < 3; ++i) {
        new_face[i] = faces[face_id*3 + i] == id_a ? center_id : faces[face_id*3 + i];
        if (faces[face_id*3 + i] == id_b) {
            faces[face_id*3 + i] = center_id;
        }
    }
    addEdgeRecurive(id_a, center_id, face_id);
    faces.insert(faces.end(), new_face, new_face+3);
    addEdgeRecurive(center_id, id_b, (faces.size()-1)/3);
    return 0;
}

int depthToMesh::drawFace(std::vector<Eigen::Vector2f> grid_coor) {
    
    cv::Point2i p0(grid_coor[0].x(), grid_coor[0].y());
    cv::Point2i p1(grid_coor[1].x(), grid_coor[1].y());
    cv::Point2i p2(grid_coor[2].x(), grid_coor[2].y());
    cv::line(triangles_map, p0, p1, cv::Scalar(0,0.255), 1);
    cv::line(triangles_map, p1, p2, cv::Scalar(0,0.255), 1);
    cv::line(triangles_map, p0, p2, cv::Scalar(0,0.255), 1);
    cv::imshow("triangels", triangles_map);
    cv::waitKey(5);
    return 0;
}