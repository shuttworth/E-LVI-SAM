#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"
/**
 * @brief 特征点信息
 *
 */
class FeaturePerFrame
{
public:
    FeaturePerFrame(const Eigen::Matrix<double, 8, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        depth = _point(7);
        cur_td = td;
    }
    double cur_td;
    Vector3d point;//归一化坐标
    Vector2d uv; //像素坐标
    Vector2d velocity; //像素上的光流速度
    double z;//特征点的深度 三角化得来的
    bool is_used;//是否被用了
    double parallax;//视差
    //构造Ax = b
    MatrixXd A;//变换矩阵
    VectorXd b;
    double dep_gradient;//代码里没用到 意义不明
    double depth; // lidar depth, initialized with -1 from feature points in feature tracker node
};
/**
 * @brief 某个feature_id对应的所有FeaturePerFrame
 *
 */
class FeaturePerId
{
public:
    const int feature_id;//特征点的ID索引
    int start_frame;//首次被观测到时，该帧的索引
    vector<FeaturePerFrame> feature_per_frame;//观测到该特征点的所有帧

    int used_num;//该特征点出现的次数
    bool is_outlier;//是否为外点
    bool is_margin;//是否边缘化
    double estimated_depth;//估计的逆深度
    bool lidar_depth_flag;//是否有lidar提供的深度信息
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    Vector3d gt_p;//代码里面没用上 意义不明

    //以feature_id为索引，包含出现该特征点出现的第一帧
    FeaturePerId(int _feature_id, int _start_frame, double _measured_depth)
            : feature_id(_feature_id), start_frame(_start_frame),
              used_num(0), estimated_depth(-1.0), lidar_depth_flag(false), solve_flag(0)
    {
        //如果lidar测量的深度在图像前面 则为有效，用lidar的深度
        if (_measured_depth > 0)
        {
            estimated_depth = _measured_depth;
            lidar_depth_flag = true;
        }
            //否则没有关联上深度 设为-1
        else
        {
            estimated_depth = -1;
            lidar_depth_flag = false;
        }
    }

    int endFrame();// 返回最后一次观测到这个特征点的帧数ID
};
/**
 * @brief 管理所有特征点 通过list容器存储特征点属性
 *
 */
class FeatureManager
{
public:

    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();//返回最后一个观测到这个特征点的图像帧ID

    //检查特征点的视差 判断是否为关键帧
    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    //设置特征点逆深度
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    //特征点三角化求深度（SVD分解）
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    //边缘化最老帧时，处理特征点保存的帧号，将起始帧是最老帧的特征点的深度值进行转移
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    //边缘化最老帧，将特征点保存的帧号向前移动
    void removeBack();
    //边缘化上一帧（次新帧）移除特征点在该帧中的信息
    void removeFront(int frame_count);
    //移除外点
    void removeOutlier();

    // 特征点列表
    list<FeaturePerId> feature;
    int last_track_num;

private:
    //计算某个特征点it_per_id在上一帧和上上新帧的视差
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    //imu->world
    const Matrix3d *Rs;
    //相机到imu的旋转矩阵
    Matrix3d ric[NUM_OF_CAM];
};

#endif