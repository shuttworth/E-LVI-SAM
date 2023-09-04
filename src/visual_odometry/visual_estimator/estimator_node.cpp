#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;

// global variable saving the lidar odometry
deque<nav_msgs::Odometry> odomQueue;
odometryRegister *odomRegister;

std::mutex m_buf;
std::mutex m_state;
std::mutex m_estimator;
std::mutex m_odom;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    // 采用最新bias重新计算上一帧加速度
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    // 这里采用的是中值积分 求旋转增量
    // 计算角速度中值
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    // 更新imu旋转
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    // 计算当前帧加速度
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    // 计算加速度中值
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // 牛顿定律更新当前的位置和速度
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    // 更新上一帧的加速度和角速度
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    // 主要获得对齐的IMU和图片特征
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    while (ros::ok())
    {
        // 如果任意一个队列为空，则无法进一步获得对齐的测量，返回
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // 如果最晚的一帧imu测量仍然早于最早的图片特征，说明imu测量队列中的所有帧均早于图片特征队列中的所有帧
        // 那么直接返回当前结果，等待获取更晚的imu测量
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            return measurements;
        }

        // 如果最早一帧的imu测量仍然晚于最早的图片特征，说明imu测量队里中的所有帧均晚于最早的图片特征
        // 那么丢弃最早一帧的图片特征，并跳过本轮循环
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        // 排除上述情况后，现在的情况是：最早一帧的imu测量早于最早的图片特征，最晚一帧的imu测量晚于最早的图片特征
        // 提取早于最早图片特征的imu测量和第一个晚于图片特征的imu测量，组成数组，
        // imu测量数组与最早的图片特征一起放入measurements
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();
        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> IMUs;
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front()); // 第一个晚于图片特征的imu测量，也放入数组当中
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    // imu测量存入到队列
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    // 唤醒条件变量con，继续执行process
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    // lock_guard的功能在于，当lock_guard实例存在时，加锁，lock_guard实例死亡时，自动解锁
    // 花括号圈定了一个作用域，配合lock_guard，花括号以内时加锁，花括号以外解锁
    {
        std::lock_guard<std::mutex> lg(m_state);
        // 此处每收到一个imu数据，则通过中值积分对最新的状态[P,V,Q]进行预测
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        // 如果进入了非线性优化阶段，即稳定的VIO过程，会发布最新的里程计(和IMU同频率)
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header, estimator.failureCount,
                              estimator.tic[0], Eigen::Quaterniond(estimator.ric[0]));
    }
}

void odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg)
{
    m_odom.lock();
    // 保存lidar位姿到位姿队列
    odomQueue.push_back(*odom_msg);
    m_odom.unlock();
}

void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        // 跳过第一帧特征，因为它不包含速度信息
        // skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        // 如果重启标志位为真，则清空各队列，清空状态和参数，重置标志位
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while (!feature_buf.empty())
            feature_buf.pop();
        while (!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

// thread: visual-inertial odometry
void process()
{
    while (ros::ok())
    {
        // 数组<pair<imu测量的数组，图片特征>>
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        // 条件变量：若m_buf被锁定或没有对齐的imu测量和图片特征则等待
        // 若m_buf解锁并且有对齐的测量，则继续执行
        con.wait(lk, [&]
                 { return (measurements = getMeasurements()).size() != 0; });
        lk.unlock();

        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;

            // 1. IMU pre-integration
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();                    // 当前imu测量时间戳
                double img_t = img_msg->header.stamp.toSec() + estimator.td; // 图片特征时间戳（默认的时间偏移量为0）
                // 对于早于图片特征的imu测量
                if (t <= img_t)
                {
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    // imu预积分 计算状态[P,V,Q]
                    // 并利用imu对系统最新状态进行传播，为视觉三角化及重投影提供位姿初值；
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    ////printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);
                }
                else
                {
                    // 对于晚于图片特征的imu测量（只有一个）
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    // 断言：倒数第二个imu测量不晚于图片特征，倒数第一个imu测量不早于图片特征，并且它们是不相同的
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    // 计算权重值
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    // 按照权重计算加速度和角速度，插值计算图片特征时间戳对应的imu测量
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    // 预积分计算状态[P,V,Q]
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    // printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // 2. VINS Optimization
            // TicToc t_s;
            // 图像特征数据不是opencv，而是包含许多通道的msg
            // 每帧图像特征的特征信息，由一个map的数据结构组成；
            // 键为关键点ID，值为该特征每个相机中的x,y,z,u,v,velocity_x,velocity_y,depth 8个变量;
            // map< 关键点ID, vector< pair<相机ID, x,y,z,u,v,vel_x,vel_y,depth >>>
            map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5; // + 0.5 可能是四舍五入
                int feature_id = v / NUM_OF_CAM;              // 关键点id
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x; // 归一化平面坐标
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i]; // 像素坐标
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i]; // 速度
                double velocity_y = img_msg->channels[4].values[i];
                double depth = img_msg->channels[5].values[i]; // 关键点深度值

                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 8, 1> xyz_uv_velocity_depth;
                xyz_uv_velocity_depth << x, y, z, p_u, p_v, velocity_x, velocity_y, depth;
                image[feature_id].emplace_back(camera_id, xyz_uv_velocity_depth);
            }

            // Get initialization info from lidar odometry 初始信息的获得 通过LIS的预积分模块得来
            vector<float> initialization_info;
            m_odom.lock();
            //; 注意：这里lidar里程计只是为了给VINS做初始化使用的，只要初始化成功之后这个信息就没用了
            initialization_info = odomRegister->getOdometry(odomQueue, img_msg->header.stamp.toSec() + estimator.td);
            m_odom.unlock();

            // VIO处理图片获取位姿：初始化/非线性优化
            estimator.processImage(image, initialization_info, img_msg->header);
            //// double whole_t = t_s.toc();
            //// printStatistics(estimator, whole_t);

            // 3. Visualization
            std_msgs::Header header = img_msg->header;
            pubOdometry(estimator, header);   // pub 当前最新滑窗VIO位姿，并且写入到文件；
            pubKeyPoses(estimator, header);   // pub 滑动窗口内关键帧（10）位姿；
            pubCameraPose(estimator, header); // pub 相机相对于世界坐标系的位姿（与vio位姿只差个相机与imu之间的外参）；
            pubPointCloud(estimator, header); // pub 滑动窗口中世界坐标系下的点云信息；
            pubTF(estimator, header);         // pub 滑动窗口中相机与IMU的外参
            pubKeyframe(estimator);           // pub 最新关键帧位姿 及其 特征点各种坐标系的坐标 两个话题；（初始化及机器人静止时没有关键帧加入，则不会pub）
        }
        m_estimator.unlock();

        m_buf.lock();
        m_state.lock();
        // 如果优化器处在非线性优化阶段（不是初始化阶段）
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update(); // 采用优化后的状态和零偏等重新计算当前的IMU状态
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Odometry Estimator Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);

    // 读取参数并设置参数
    readParameters(n);
    estimator.setParameter();

    // 注册各种发布器
    registerPub(n);

    // 创建odom注册器(odomRegister)，用来获取LiDAR位姿并转换为camera位姿
#if IF_OFFICIAL
    odomRegister = new odometryRegister(n);
#else
    Eigen::Vector3d t_lidar_imu = -R_imu_lidar.transpose() * t_imu_lidar;
    odomRegister = new odometryRegister(n, R_imu_lidar.transpose(), t_lidar_imu);
#endif

    // 订阅话题：原始imu话题，（imu预积分的）lidar位姿话题，图片特征话题，系统重启话题
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 5000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_odom = n.subscribe("odometry/imu", 5000, odom_callback);
    ros::Subscriber sub_image = n.subscribe(PROJECT_NAME + "/vins/feature/feature", 1, feature_callback);
    ros::Subscriber sub_restart = n.subscribe(PROJECT_NAME + "/vins/feature/restart", 1, restart_callback);
    // 如果不适用LiDAR位姿，则停止订阅imu预积分位姿
    if (!USE_LIDAR)
        sub_odom.shutdown();

    // 单独的线程处理以上订阅并放入buf的消息
    std::thread measurement_process{process};

    // 四个线程共同处理
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}