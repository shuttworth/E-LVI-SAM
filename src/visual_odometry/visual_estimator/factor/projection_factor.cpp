#include "projection_factor.h"

Eigen::Matrix2d ProjectionFactor::sqrt_info;
double ProjectionFactor::sum_t;
//构造函数传入的是重投影误差函数中的常数，即一对匹配的特征点
ProjectionFactor::ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) : pts_i(_pts_i), pts_j(_pts_j)
{
//广角相机的球面模型
#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    // Pi Qi Pj Qj都是imu坐标系下的
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    // Eigen::Quaterniond q(w,x,y,z)
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    // pts_i 是i时刻归一化相机坐标系下特征点的三维坐标
    // 第i帧相机坐标系下特征点的逆深度
    double inv_dep_i = parameters[3][0];
    // 第i帧相机坐标系下特征点的三维坐标 --->除以逆深度 = 乘上深度
    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    // 第i帧IMU坐标系下特征点的三维坐标 ---> imu_i^t_feature = imu_i^q_cam_i * cam_i^P_feature + imu_i^t_cam_i
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    // 世界坐标系下特征点的三维坐标 ---> world^t_feature = world^Q_imu_i * imu_i^t_feature + world^t_imu_i
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    // 第j帧imu坐标系下特征点的三维坐标 ---> [  R1,   T1] * [  R2,   T2] = [ R1R2,   R1*T2+T1]
    //                                     [0, 0, 0, 1]   [0, 0, 0, 1]   [0, 0, 0,    1    ]
    // 即[imu_j^R_world, imu_j^T_world] * [world^R_feature, world^T_feature]
    // 对于平移部分：imu_j^R_world * world^T_feature + imu_j^T_world   ----(1)
    // 其中旋转： imu_j^R_world = world^R_imu_j.inverse()   平移：imu_j^T_world = - world^R_imu_j.inverse() * world^T_imu_j
    // 因此(1)式转换为：world^R_imu_j.inverse() * (world^T_feature -  world^T_imu_j)
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    // 第j帧相机坐标系下特征点的三维坐标
    // [cam_j^R_imu_j, cam_j^T_imu_j]*[imu_j^R_feature, imu_j^T_feature]
    // 对于平移部分：cam_j^R_imu_j * imu_j^T_feature + cam_j^T_imu_j   ----(2)
    // 其中旋转： cam_j^R_imu_j = imu_j^R_cam_j.inverse()     平移：cam_j^T_imu_j = - imu_j^R_cam_j.inverse()* imu_j^T_cam_j
    // 因此(2)式转换为：imu_j^R_cam_j.inverse() * (imu_j^T_feature - imu_j^T_cam_j)
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR //球面模型
    residual = tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
#else //针孔模型
    double dep_j = pts_camera_j.z(); //第j帧相机系下深度
    // 重投影和观测值在归一化坐标系上的差值
    // head<n>()函数：对变量Eigen::Vector4f x进行x.head<n>()操作表示提取前n个元素
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>(); // 重投影误差
    // F = G(f) f = V(u)
#endif

    residual = sqrt_info * residual; //误差乘上信息矩阵

    // reduce 表示残差residual对pts_camera_j的导数
    //  pts_camera_j = = qic.inverse() * (pts_imu_j - tic)
    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
        //投影到单位球面上
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), -x1 * x2 / pow(norm, 3), -x1 * x3 / pow(norm, 3),
            -x1 * x2 / pow(norm, 3), 1.0 / norm - x2 * x2 / pow(norm, 3), -x2 * x3 / pow(norm, 3),
            -x1 * x3 / pow(norm, 3), -x2 * x3 / pow(norm, 3), 1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        //投影到归一化平面上
        // 重投影误差对j帧相机坐标系下坐标求导
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
                0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        //雅各比部分
        reduce = sqrt_info * reduce;

        // 右扰动模型
        // 残差项的Jacobian
        // 先求residual对各项的Jacobian，然后用链式法则乘起来
        // 对第i帧的位姿 pbi,qbi      2X7的矩阵 最后一项是0
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            // c^R_b * w^R_bj
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            // skewSymmetric反对称阵
            // c^R_b * w^R_bj *(- imu_i^t_feature)
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();// todo
        }
        // 对第j帧的位姿 pbj,qbj
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        // 对相机到IMU的外参 pbc,qbc (qic,tic)
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                                     Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        // 对逆深度 \lambda (inv_dep_i)
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
#if 1
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
#else
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i;
#endif
        }
    }
    sum_t += tic_toc.toc();

    return true;
}