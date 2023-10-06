#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs)
{
    Matrix3d A;
    Vector3d b;
    Vector3d delta_bg;
    A.setZero();
    b.setZero();
    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        // 视觉匹配的相对旋转
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R);
        // 提取IMU旋转和bias的雅可比（等同于δR，δbias的系数矩阵，当然这是线性化的结果）
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        // 计算视觉匹配的相对旋转与预积分相对旋转的差值，乘2是因为bias对预积分R的影响为1/2（中值积分）
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    // 求解误差值
    delta_bg = A.ldlt().solve(b);
    ROS_INFO_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());
    // 更新bias
    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;
    // 重新传播预积分
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]);
    }
}

MatrixXd TangentBasis(Vector3d &g0) // 寻找单位正交基
{
    Vector3d b, c;
    Vector3d a = g0.normalized(); // a为当前g的方向向量
    Vector3d tmp(0, 0, 1);
    // b和c分别是垂直于g0的单位正交基
    // b为tmp向量减去a向量在tmp上的投影后的方向向量
    // c是a与b的叉乘，因此abc相互正交
    if (a == tmp)
        tmp << 1, 0, 0;
    b = (tmp - a * (a.transpose() * tmp)).normalized();
    c = a.cross(b);
    MatrixXd bc(3, 2);
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    // 采用上次计算的g的方向和G的模长，得到每次求解的初值
    Vector3d g0 = g.normalized() * G.norm();
    // 为了仍然保持g的模长9.8，并且保证仍然可以通过线性方程组的方式求解
    // 将g0的更新方式设置为g0=g0+w1lx+w2ly，原来的Ax=b改为求解w1和w2而不是直接求解g
    Vector3d lx, ly;
    // VectorXd x;
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 2 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    // 4轮的求解
    for (int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0);
        int i = 0;
        // 构建A和b与先前有所不同
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
        {
            frame_j = next(frame_i);

            MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt;

            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;

            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
            // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            // MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity();

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        VectorXd dg = x.segment<2>(n_state - 3);
        // 更新g0，并保持模长
        g0 = (g0 + lxly * dg).normalized() * G.norm();
        // double s = x(n_state - 1);
    }
    g = g0;
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    // 相机IMU松耦合初始化
    int all_frame_count = all_image_frame.size();
    int n_state = all_frame_count * 3 + 3 + 1;

    MatrixXd A{n_state, n_state};
    A.setZero();
    VectorXd b{n_state};
    b.setZero();

    map<double, ImageFrame>::iterator frame_i;
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    // 按照公式构建Ax=b
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++)
    {
        frame_j = next(frame_i);

        MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        // 特地把关于尺度的一行缩小为原来的0.01倍，从而将尺度结果放大100倍，均衡求解最小二乘解
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //// cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        // cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
        // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        // MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity();

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    // 同时放大A和b，可能是为了提高cholesky分解精度从而提高x的求解精度，提高数值稳定性。
    A = A * 1000.0;
    b = b * 1000.0;
    x = A.ldlt().solve(b);
    // 得到尺度，重力加速度
    double s = x(n_state - 1) / 100.0;
    ROS_INFO("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    ROS_INFO_STREAM(" result g     " << g.norm() << " " << g.transpose());
    // 验证结果
    if (fabs(g.norm() - G.norm()) > 1.0 || s < 0)
    {
        return false;
    }

    // 上面的计算得到的重力向量有可能得到的不是模长为9.8的重力向量，
    // 因为重力向量本身应当只有两个自由度（模长固定），但是上面的计算只是把g当做三个自由度普通向量计算
    // 因此需要重新计算重力向量和优化变量
    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    ROS_INFO_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if (s < 0.0)
        return false;
    else
        return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x)
{
    // 用相机位姿和IMU位姿建立约束标定陀螺仪bias
    solveGyroscopeBias(all_image_frame, Bgs);

    // 相机IMU松耦合初始化，求解线性solveGyroscopeBias方程
    if (LinearAlignment(all_image_frame, g, x))
        return true;
    else
        return false;
}
