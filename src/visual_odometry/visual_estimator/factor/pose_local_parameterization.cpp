#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    //初始值
    Eigen::Map<const Eigen::Vector3d> _p(x);//前三个为x y z
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);//后四个为w x y z
    //增量
    Eigen::Map<const Eigen::Vector3d> dp(delta);
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    //更新后的值
    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);//平移
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);//旋转，使用四元数表示

    p = _p + dp;
    q = (_q * dq).normalized();//对四元数进行归一化

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();//前六行设置为单位阵
    j.bottomRows<1>().setZero();//最后一行设置为0

    return true;
}
