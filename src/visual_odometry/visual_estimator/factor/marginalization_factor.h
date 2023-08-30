#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    //构造函数，包括cost_function（代价函数）、loss_function（核函数）、关联的参数块、待边缘化的参数块索引
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
            : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();
    //代价函数
    ceres::CostFunction *cost_function;
    //核函数
    ceres::LossFunction *loss_function;
    //优化变量内存地址
    std::vector<double *> parameter_blocks;
    //存储待边缘化的变量内存地址的id，即parameter_blocks的id
    std::vector<int> drop_set;

    //雅各比
    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    //残差项
    Eigen::VectorXd residuals;
    // 残差 IMU 15×1 视觉 2×1
    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};
// 多线程结构体
struct ThreadsStruct
{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
public:
    ~MarginalizationInfo();//析构函数
    int localSize(int size) const;
    int globalSize(int size) const;
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);//添加残差块相关变量（待边缘化）
    void preMarginalize();//计算每个残差对应的雅各比，并更新parameter_block_data
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    //所有观测项，包括视觉观测项，imu观测项以及上一次的边缘化项
    //从中去分离出需要边缘化的状态量和需要保留的状态量
    std::vector<ResidualBlockInfo *> factors;
    int m, n;//m：需要边缘化的变量个数，n:保留下来的变量个数

    //<优化变量内存地址，global size>
    std::unordered_map<long, int> parameter_block_size; //global size
    int sum_block_size;
    //<优化变量内存地址，在矩阵块中的id>
    std::unordered_map<long, int> parameter_block_idx; //local size
    //<优化变量内存地址,数据>
    std::unordered_map<long, double *> parameter_block_data;
    //按顺序存放上面的 parameter_block_size 中被保留的优化变量
    std::vector<int> keep_block_size; //global size
    //按顺序存放上面的 parameter_block_idx 中被保留的优化变量
    std::vector<int> keep_block_idx;  //local size
    //按顺序存放上面的 parameter_block_data 中被保留的优化变量
    std::vector<double *> keep_block_data;

    //边缘化后从信息矩阵恢复出来的雅各比矩阵
    Eigen::MatrixXd linearized_jacobians;
    //边缘化后从信息矩阵恢复出来的残差矩阵
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};
//直接继承最基本的类
class MarginalizationFactor : public ceres::CostFunction
{
public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
