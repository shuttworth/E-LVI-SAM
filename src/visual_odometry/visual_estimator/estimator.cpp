#include "estimator.h"

Estimator::Estimator() : f_manager{Rs}
{
    failureCount = -1;
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric);
    // sqrt_info在该处赋值，像素噪声的标准差为1.5个像素
    // 马氏范数 消除单位（长度、角度）的影响
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    ++failureCount;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
            sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        //预积分
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);

        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc; //根据运动学方程可有得到当前的P是从world指向当前的位置 Pos(now)-Pos(0) =Pos(o->now)
        Vs[j] += dt * un_acc;                         //
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

/**
 * @brief 处理图像特征数据
 * 
 * @param image 某帧所有的特征点[camera_id,[x,y,z,u,v,vx,vy]]构成的map,索引为feature_id
 * @param lidar_initialization_info 激光惯性子系统的相关状态量
 * @param header 某帧图像的头信息
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, 
                             const vector<float> &lidar_initialization_info,
                             const std_msgs::Header &header)
{
    // Add new image features
    // 返回值是当前帧是否是关键帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;  // 若次新帧为关键帧则边缘化掉滑窗中的最老帧
    else
        marginalization_flag = MARGIN_SECOND_NEW;   // 若次新帧不是关键帧，则边缘化掉滑窗中的次新帧

    // Marginalize old imgs if lidar odometry available for initialization
    // 注意这里： 更改了边缘化的条件，若激光惯性子系统已经传过来了有效值，则无论次新帧是否为关键帧
    // 都是只边缘化最老帧
    if (solver_flag == INITIAL && lidar_initialization_info[0] >= 0)
        marginalization_flag = MARGIN_OLD;

    Headers[frame_count] = header;

    ImageFrame imageframe(image, lidar_initialization_info, header.stamp.toSec());
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe));
    
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    // Calibrate rotational extrinsics
    if(ESTIMATE_EXTRINSIC == 2)
    {
        ROS_WARN("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        // 直到滑动窗口填满，才进行滑动窗口
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            // 若相机与IMU之间的外参没有标定成功，则一直进行标定，直到成功
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
                // 若相机与IMU之间已经获得外参，则进行系统初始化
               result = initialStructure();
               initial_timestamp = header.stamp.toSec();
            }
            if(result)
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                // ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        solveOdometry();

if (failureDetection())
{
ROS_ERROR("VINS failure detection!");
failure_occur = 1;
clearState();
setParameter();
ROS_ERROR("VINS system reboot!");
return;
}

slideWindow();
f_manager.removeFailures();

// prepare output of VINS
key_poses.clear();
for (int i = 0; i <= WINDOW_SIZE; i++)
key_poses.push_back(Ps[i]);

last_R = Rs[WINDOW_SIZE];
last_P = Ps[WINDOW_SIZE];
last_R0 = Rs[0];
last_P0 = Ps[0];
}
}
/**
 * @brief 视觉惯性子系统进行初始化
 * 
 * @return true 
 * @return false 
 */
bool Estimator::initialStructure()
{
    // Lidar initialization【】
    {
        bool lidar_info_available = true;

        // clear key frame in the container
        for (map<double, ImageFrame>::iterator frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
            frame_it->second.is_key_frame = false;

        // check if lidar info in the window is valid
        // reset_id 是激光雷达信息的标志位
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            if (all_image_frame[Headers[i].stamp.toSec()].reset_id < 0 ||   // 激光雷达信息无效
                all_image_frame[Headers[i].stamp.toSec()].reset_id != all_image_frame[Headers[0].stamp.toSec()].reset_id)
            {
                // 满足上面的信息，则激光雷达数据信息不可使用
                // lidar odometry not available (id=-1) or lidar odometry relocated due to pose correction
                lidar_info_available = false;
                ROS_INFO("Lidar initialization info not enough.");
                break;
            }
        }

        // 如果激光雷达信息可使用，则直接使用激光惯性子系统的数据
        if (lidar_info_available == true)
        {
            // Update state
            /**
             * @brief 大家可以考虑以下这里的的状态量所在的坐标系了，相机坐标系下还是其它坐标系下？
             * 
             */
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                Ps[i] = all_image_frame[Headers[i].stamp.toSec()].T;
                Rs[i] = all_image_frame[Headers[i].stamp.toSec()].R;
                Vs[i] = all_image_frame[Headers[i].stamp.toSec()].V;
                Bas[i] = all_image_frame[Headers[i].stamp.toSec()].Ba;
                Bgs[i] = all_image_frame[Headers[i].stamp.toSec()].Bg;

                // 使用激光惯性子系统的加速度零偏和角速度零偏，重新进行预积分
                pre_integrations[i]->repropagate(Bas[i], Bgs[i]);

                all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;  // 滑窗中的每一帧都有效
            }

            /**
             * @brief 疑问： 这里的重力向量？
             * 
             */
            g = Eigen::Vector3d(0, 0, all_image_frame[Headers[0].stamp.toSec()].gravity);

            // reset all features
            VectorXd dep = f_manager.getDepthVector();
            for (int i = 0; i < dep.size(); i++)
                dep[i] = -1;
            f_manager.clearDepth(dep);

            // triangulate all features
            Vector3d TIC_TMP[NUM_OF_CAM];
            for(int i = 0; i < NUM_OF_CAM; i++)
                TIC_TMP[i].setZero();   // 相机与IMU之间的平移设置为0，忽略不计
            ric[0] = RIC[0];
            f_manager.setRic(ric);
            // 三角化特征点
            f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

            return true;
        }
    }


    // 通过加速度标准差判断IMU是否有充分运动激励
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            //
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            // cout << "frame g " << tmp_g.transpose() << endl;
        }
        // 得到标准差
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        // 加速度的标椎差过小，反映了传感器运动不激烈，没有的进行激励
        if(var < 0.25)
        {
            ROS_INFO("Trying to initialize VINS, IMU excitation not enough!");
            // return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];  // l <<-- r
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;   // 存储滑窗中所有的特征点的归一化坐标
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        // 遍历同一个三维点的投影点
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;    // 归一化坐标
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;  // l <<-- r
    Vector3d relative_T;
    int l;  // 滑窗中第几帧拿到满足要求的位姿变换矩阵
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    /**
     * @brief 通过SFM之后，已经获得了滑窗中每帧的位姿，以及部分三维点，此时三维点还没有尺度信息
     *  第l帧设置为世界坐标系
     * 
     */
    GlobalSFM sfm;
    //
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    // solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i].stamp.toSec())
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();  //  camera <-- imu : l帧为世界坐标系的原点
            frame_it->second.T = T[i];   // 这里忽略了相机与IMU之间的平移向量
            i++;
            continue;
        }

        /**
         * @brief 下面程序未运行，应该是bug!!!
         * 
         */
        if((frame_it->first) > Headers[i].stamp.toSec())
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }

    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }
}
/**
 * @brief   视觉惯性联合初始化
 * @Description 陀螺仪的偏置校准(加速度偏置没有处理) 计算速度V[0:n] 重力g 尺度s
 *              更新了Bgs后，IMU测量量需要repropagate
 *              得到尺度s和重力g的方向后，需更新所有图像帧在世界坐标系下的Ps、Rs、Vs
 * @return  bool true：成功
 */
bool Estimator::visualInitialAlign()
{
    VectorXd x;
    // solve scale
    //计算陀螺仪偏置，尺度，重力加速度和速度
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        ROS_INFO("solve gravity failed, try again!");
        return false;
    }

    // change state
    // 得到所有图像帧的位姿Ps、Rs，并将其置为关键帧
    for (int i = 0; i <= frame_count; i++)
    {
        // R: l <-- imu   imu坐标系下旋转到世界坐标系下第l帧
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R;
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T;  // 这里忽略了相机与IMU之间的平移向量
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true;
    }

    // reset all depth to -1
    //将所有特征点的深度置为-1
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    f_manager.clearDepth(dep);

    // triangulat on cam pose , no tic
    //重新计算特征点的深度
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();


    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // 三角化所有的特征点， 这里的TIC设置为了0
    // 这里三角化得到的深度信息是没有尺度的！！！
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    //陀螺仪的偏置bgs改变，重新计算预积分
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }

    /**
     * @brief 添加尺度信息
     * 
     */
    //将Ps、Vs、depth尺度s缩放
    for (int i = frame_count; i >= 0; i--)
        // Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);

    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        // 获取每一帧的速度信息，并将速度信息转换到世界坐标系下（也就是第l帧）
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);  
        }
    }
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s; // 添加尺度信息
    }
    // 通过将重力旋转到z轴上，得到世界坐标系与摄像机坐标系c0之间的旋转矩阵rot_diff
    // R0 ---- Rw_g = Rw_c0
    Matrix3d R0 = Utility::g2R(g);
    // Rs ---- Rc0_imu  ，R0*Rs = Rw_c0 * Rc0_imu
    // 这里只是相乘取出yaw角，并没有改变R0的坐标系
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;

    // 使得，IMU坐标系下的重力向量与相机坐标系的第0帧的坐标系进行对齐
    // R0 --- Rw_c0
    // g_w = R0 * g_c0,这一步得到 g_w
    g = R0 * g;
    // Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    //所有变量从参考坐标系c0旋转到世界坐标系w
    // rot_diff = R0 --- R_w_c0
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

/**
 * @brief 在滑窗获取老帧和新帧的位姿变换： l <<-- r
 * 
 * @param relative_R 在滑窗中获得的老帧和新帧之间的旋转矩阵：  l <<-- r
 * @param relative_T 在滑窗中获得的老帧和新帧之间的平移向量 :  l <<-- r
 * @param l 滑窗中第几帧获得的位姿变换  
 * @return true 
 * @return false 
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        //寻找第i帧到窗口最后一帧的对应特征点
        vector<pair<Vector3d, Vector3d>> corres;

        /**
         * @brief 获取的匹配点是从滑窗中的最新的帧和滑窗中最后一帧的进行匹配，以便获得足够的视差！！！
         * 
         */
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;    // 匹配点对的视差和
            double average_parallax;    // 平均视差
            for (int j = 0; j < int(corres.size()); j++)
            {
                //第j个对应点在第i帧和最后一帧的(x,y)
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();   // 计算匹配点对的视差
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            //判断是否满足初始化条件：视差>30和内点数满足要求
            //同时返回窗口最后一帧（当前帧）到第l帧（参考帧）的Rt
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}
//三角化求解特征点的深度信息，然后进行非线性化
void Estimator::solveOdometry()
{
    //如果当前的滑动窗口内的图像数量小于滑动窗口的最大数量则退出
    //说明此时不满足批处理的要求
    if (frame_count < WINDOW_SIZE)
        return;
    //进行非线性优化
    if (solver_flag == NON_LINEAR)
    {
        //进行三角化
        f_manager.triangulate(Ps, tic, ric);
        //进入优化
        optimization();
    }
}
// vector转换成double数组，与ceres使用的容器类型相对应
// para_Pose存储Ps Rs，转变成para_SpeedBias存储Vs、Bas、Bgs
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}
//与上面函数相反
void Estimator::double2vector()
{
    //固定了先验信息
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                                 .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    // TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                .toRotationMatrix()
                .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5])
                .toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];
}
//系统故障检测
bool Estimator::failureDetection()
{
    //次新中的
    if (f_manager.last_track_num < 2)
    {
        ROS_ERROR("VINS little feature %d!", f_manager.last_track_num);
        // return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_ERROR("VINS big IMU acc bias estimation %f, restart estimator!", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_ERROR("VINS big IMU gyr bias estimation %f, restart estimator!", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    if (Vs[WINDOW_SIZE].norm() > 30.0)
    {
        ROS_ERROR("VINS big speed %f, restart estimator!", Vs[WINDOW_SIZE].norm());
        return true;
    }
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5.0)
    {
        ROS_ERROR("VINS big translation, restart estimator!");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        ROS_ERROR("VINS big z translation, restart estimator!");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / M_PI * 180.0;
    if (delta_angle > 50)
    {
        ROS_ERROR("VINS big delta_angle, moving too fast!");
        // return true;
    }
    return false;
}
/**
 * @brief 后端非线性优化 状态向量:(P Q V ba bg RIC inv_depth)
 * 𝑋=[𝑥0,𝑥1,⋯𝑥𝑛,𝑥𝑐𝑏,𝜆0,𝜆1,⋯𝜆𝑚]
 * 𝑥𝑘=[𝑤^𝑝_𝑏𝑘,𝑤^𝑣_𝑏𝑘,𝑤^𝑞_𝑏𝑘,𝑏𝑎,𝑏𝑔]
 * 𝑥𝑐𝑏=[𝑝𝑐𝑏,𝑞𝑐𝑏]
 */
void Estimator::optimization()
{
    //创建ceres problem实例
    ceres::Problem problem;
    //核函数设置为柯西核函数
    ceres::LossFunction *loss_function;
    // loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);
    //遍历滑动窗口
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        //位姿参数块
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    // 相机外参优化
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        //外参参数块
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC) //如果不需要标定cam和imu之间的外参
        {
            ROS_DEBUG("fix extinsic param");
            //则变量固定为constant
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    if (ESTIMATE_TD)
    {
        //标定同步时间优化
        problem.AddParameterBlock(para_Td[0], 1);
        // problem.SetParameterBlockConstant(para_Td[0]);
    }
    // Eigen ---> double
    vector2double();

    // marginalization residual
    // 先验信息残差
    if (last_marginalization_info)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        // last_marginalization_parameter_blocks保存的和margin掉变量有关系的变量
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }

    // IMU pre-integration residual
    // IMU残差
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0) //预积分时间间隔大于10则不作为观测
            continue;
        IMUFactor *imu_factor = new IMUFactor(pre_integrations[j]);
        //
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }

    // Image feature re-projection residual
    // 重投影误差
    int f_m_cnt = 0; //统计用于非线性化的特征点个数
    int feature_index = -1;
    //挨个遍历f_manager中feature容器中的特征点
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //特征至少被跟踪了两次并且最开始检测到的图像帧需在倒数第二帧之前
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //有效特征点数量++
        ++feature_index;

        //观测到该特征点的首帧
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        //获取首次观测到该特征点时其所在的归一化相机坐标系
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        //遍历该特征点在每一帧图像里的信息
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i == imu_j)
            {
                continue;
            }
            //得到第二个特征点
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD) //需要进行在线时间同步
            {
                ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                  it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                  it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);

                // depth is obtained from lidar, skip optimizing it
                //如果深度来自于lidar则不需要进行优化 设为constant
                if (it_per_id.lidar_depth_flag == true)
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
            }
            else //如果不需要在线时间同步
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]);

                // depth is obtained from lidar, skip optimizing it
                //如果深度来自于lidar则不需要进行优化 设为constant
                if (it_per_id.lidar_depth_flag == true)
                    problem.SetParameterBlockConstant(para_Feature[feature_index]);
            }
            f_m_cnt++; //可用于非线性化的特征点数量++
        }
    }
    //删除了原版vinsmono的回环检测残差
    // 求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    // options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    // options.use_explicit_schur_complement = true;
    // options.minimizer_progress_to_stdout = true;
    // options.use_nonmonotonic_steps = true;
    // 0.035ms
    //边缘化老的就减少处理时间???
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //优化后将double->Eigen
    double2vector();

    /**
     * @brief marginalization_flag 进行边缘化
     *
     */
    //丢弃老的一帧
    if (marginalization_flag == MARGIN_OLD)
    {
        //用来边缘化操作的对象
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();
        //如果上一帧边缘化的先验信息存在
        if (last_marginalization_info)
        {
            //存储上次边缘化后与Pose[0],SpeedBias[0]有关的参数块的序号
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                //如果待估计参数为首帧状态量，则将序号push进drop_set中
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            // 添加上一次边缘化的残差块
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        { // 加入需要边缘化的IMU信息
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor *imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                               vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                               vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            //加入视觉的先验
            int feature_index = -1;
            // 遍历滑窗内所有的特征
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                // 特征至少被跟踪了两次并且最开始检测到的图像帧需在倒数第二帧之前
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                    continue;

                ++feature_index;
                // 观测到该特征点的首帧
                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;
                //获取首次观测到该特征点时其所在的归一化相机坐标系
                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if (imu_i == imu_j)
                        continue;
                    //得到第二个特征点
                    Vector3d pts_j = it_per_frame.point;
                    if (ESTIMATE_TD)
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        TicToc t_pre_margin;
        // 计算每个残差对应的雅各比矩阵，并将各参数块拷贝统一内存中
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        // 多线程构造先验项舒尔补AX = b的结构，对X0处的值进行泰勒展开，保留一阶导，即雅各比矩阵，然后计算残差
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }
    else
    {
        //如果上上帧不是关键帧的话，则将该帧的视觉观测边缘化掉
        //步骤与之前的相同
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            //由于滑窗是会改变优化变量的地址的，因此对被保留的优化变量的地址进行更新
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
}
//滑窗
void Estimator::slideWindow()
{
    TicToc t_margin;
    //处理最老帧和次新帧的滑窗方式不同
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0].stamp.toSec();
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        //窗口必须是满的
        if (frame_count == WINDOW_SIZE)
        {
            //一帧一帧挨个交换
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            //最后一帧的状态量赋给当前值作为初始值
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];
            //预积分量置为零
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            //将buffer清空
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();
            //清空最老帧之间的状态
            if (true || solver_flag == INITIAL)
            {
                //释放空间
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }
                //释放完空间再erase
                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            // 将最后两个预积分观测合并成一个
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }
            //滑窗交换
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];
            //重置预积分量
            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};
            //清除相关buffer
            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
// 对被移除的倒数第二帧的地图点进行处理
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
//地图点与第一个观测到它的帧相关，因此需要对被移除的帧所看见的地图点进行
//每个地图点的首个观测帧id减1
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth) // 如果已经非线性化过了
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        // back_R0 back_P0是被移除的帧的位姿
        R0 = back_R0 * ric[0];           // 被移除的相机的姿态
        R1 = Rs[0] * ric[0];             // 当前最老的相机姿态（次老帧）
        P0 = back_P0 + back_R0 * tic[0]; // 被移除的相机的位置
        P1 = Ps[0] + Rs[0] * tic[0];     // 当前最老的相机位置（次老帧）
        //把被移除帧看见地图点的管理权交给当前的最老帧
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}