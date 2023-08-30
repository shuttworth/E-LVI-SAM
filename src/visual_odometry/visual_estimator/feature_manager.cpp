#include "feature_manager.h"
//获取最后一个观测到该特征的图像帧对应的索引
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}
//构造函数先将左右目相机到imu的旋转矩阵设为单位阵
FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}
//设置左右目相机到imu的旋转矩阵
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}
//清空容器内所有信息
void FeatureManager::clearState()
{
    feature.clear();
}

/**
 * @brief 获取特征点列表中的所有满足要求的特征点数目
 *
 * @return int
 */
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    // 遍历特征点列表，
    for (auto &it : feature)
    {
        // 获取对应特征点id已经跟踪到的特征点个数
        it.used_num = it.feature_per_frame.size();

        /**
         * @brief 对应的特征点已经跟踪到两次且首次观测是次次新帧以前的
         *
         */
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            //有效特征点数量++
            cnt++;
        }
    }
    return cnt;
}

/**
 * @brief 将特征点放入list容器，计算每一个点的跟踪次数和它在次新帧和次次新帧间的视差，返回是否是关键帧
 *
 * @param frame_count 窗口内帧的个数
 * @param image 某帧的所有特征点的[camera_id, [x, y, z, u, v, vx, vy]] 构成的map, 索引为feature_id
 * @param td IMU和cam同步的时间差
 * @return true 次新帧是关键帧
 * @return false 次新帧非关键帧
 */
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    // 用于记录所有特征点的视差和
    double parallax_sum = 0;
    // 计算视差的次数
    int parallax_num = 0;
    // 在此帧上被跟踪到的点的个数
    last_track_num = 0;
    // 遍历图像中所有的特征点
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        // find feature id in the feature bucket
        int feature_id = id_pts.first;
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          { return it.feature_id == feature_id; });

        // 没有找到，是新的特征点
        if (it == feature.end())
        {
            // this feature in the image is observed for the first time, create a new feature object
            feature.push_back(FeaturePerId(feature_id, frame_count, f_per_fra.depth));
            feature.back().feature_per_frame.push_back(f_per_fra);
        }
        else if (it->feature_id == feature_id) // 找到了，就是跟踪上的点
        {
            // this feature in the image has been observed before
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++; // 跟踪到的点的个数加一
            // sometimes the feature is first observed without depth
            // (initialize initial feature depth with current image depth is not exactly accurate if camera moves very fast, then lines bebow can be commented out)

            // 以下部分是新的内容
            if (f_per_fra.depth > 0 && it->lidar_depth_flag == false)
            {
                it->estimated_depth = f_per_fra.depth;
                it->lidar_depth_flag = true;

                it->feature_per_frame[0].depth = f_per_fra.depth;
            }
        }
    }

    /**
     * @brief 如果滑窗中的帧数少于2或者当前帧跟踪到特征点的数目过少，就将当前帧设置为关键帧
     *
     */
    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (auto &it_per_id : feature)
    {
        // 次次新帧
        if (it_per_id.start_frame <= frame_count - 2 &&
            // it_per_id.feature_per_frame: 可以表示一个特征点已经被跟踪到了多少次
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) // 次新帧及之后被观测到
        {
            // 上面的if语句能够确保产生视差
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
            parallax_num++;
        }
    }

    // 次次新帧和次新帧两帧没有同事观测到一个特征点
    if (parallax_num == 0)
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

//调试输出
void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ", j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

/**
 * @brief 获取匹配特征点
 *
 * @param frame_count_l 上一帧
 * @param frame_count_r 当前帧
 * @return vector<pair<Vector3d, Vector3d>> 匹配特征点 归一化坐标平面
 */
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres;
    for (auto &it : feature) //遍历feature中的所有特征点
    {
        // 要有足够的视差
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            //存储点在归一化坐标系下的位置
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame; //图像帧l减去观测到该特征点的起始帧
            int idx_r = frame_count_r - it.start_frame; //图像帧r减去观测到该特征点的起始帧

            a = it.feature_per_frame[idx_l].point;

            b = it.feature_per_frame[idx_r].point;

            corres.push_back(make_pair(a, b));
        }
    }
    return corres;
}
//设置特征点的逆深度估计值
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;         //赋值为-1 ++之后变为0
    for (auto &it_per_id : feature) //遍历feature容器中所有特征点
    {
        //至少两帧观测得到该特征点 且 首次观测到该特征点的图像帧在滑动窗口内
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        //求解逆深度
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //深度小于0
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2; //估计失败
        }
        else
            it_per_id.solve_flag = 1; //估计成功
    }
}
//如果估计失败则剔除
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        // 估计失败则剔除该特征点
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}
//清除存储的深度
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        it_per_id.lidar_depth_flag = false;
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount()); // 逆深度
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2                      // 滑窗中跟踪到某个特征点的数量过少
              && it_per_id.start_frame < WINDOW_SIZE - 2)) // 或者刚观测到特征点（次次新帧开始观测到新的特征点） 【感觉这两个条件一个意思！！！】
            continue;

        // optimized depth after ceres maybe negative, initialize them with default value for this optimization
        if (it_per_id.estimated_depth > 0)
            dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
        else
            dep_vec(++feature_index) = 1. / INIT_DEPTH; // 5.0
    }
    return dep_vec;
}

/**
 * @brief 三角化特征点
 *
 * @param Ps 激光惯性子系统获得的位置信息
 * @param tic 相机与IMU之间的平移向量
 * @param ric 相机与IMU之间的旋转矩阵
 */
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    //遍历feature容器中所有特征信息
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        //至少两帧观测得到这个特征点  且 不能是滑窗中的最后两帧
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // depth is available, skip triangulation (trust the first estimate)
        if (it_per_id.estimated_depth > 0)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;
        // R0 t0为第i帧cam--->world的变换矩阵
        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0]; // Rs应该是 Rwi   Ps  wi
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];             // Rwc
        // 投影矩阵
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            // R1 t1为第j帧cam--->world的变换矩阵
            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0); // 转换到相机坐标系下  T01
            Eigen::Matrix3d R = R0.transpose() * R1;        // Rc0c1

            // 若以上坐标系正确的话，就是将相机的世界坐标系的三维点投影到当前相机坐标系下
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            //获取归一化坐标系下的位置
            //只保留方向信息 去除尺度信息
            Eigen::Vector3d f = it_per_frame.point.normalized();
            // P = [P1 P2 P3]^T
            // AX=0      A = [A(2*i) A(2*i+1) A(2*i+2) A(2*i+3) ...]^T
            // A(2*i)   = x(i) * P3 - z(i) * P1
            // A(2*i+1) = y(i) * P3 - z(i) * P2
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            // 这里bug，应该提前
            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        //对A的SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)，深度为z/w
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];

        // update depth from triangulation
        // 得到的深度值实际上就是第一个观察到这个特征点的相机坐标系下的深度值
        it_per_id.estimated_depth = svd_method;
        // check if triangulation failed
        if (it_per_id.estimated_depth < 0) //太近 vinsmono--->0.1 ???
        {
            it_per_id.estimated_depth = INIT_DEPTH; // 5.0
        }
    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}
//边缘化最老帧时，处理特征点保存的帧号，将起始帧是最老帧的特征点的深度值进行转移
// marg_R、marg_P为被边缘化的位姿，new_R、new_P为在这下一帧的位姿
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        // 没有被移除的帧看到，则直接将帧号减一（滑窗滑掉了，往后移动一位）
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            // feature point and depth in old local camera frame
            //特征点起始帧为最老帧
            //取出归一化坐标系
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;
            double depth = -1;
            //如果lidar测量好使则取lidar的值
            if (it->feature_per_frame[0].depth > 0)
                // if lidar depth available at this frame for feature
                depth = it->feature_per_frame[0].depth;
            //如果三角化的好使则用三角化的值
            else if (it->estimated_depth > 0)
                // if estimated depth available
                depth = it->estimated_depth;

            // delete current feature in the old local camera frame
            // 该特征点被观测到的帧中减去一个
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            // 特征点在当前滑窗中只被观测到了一次或没有被观测则直接剔除
            if (it->feature_per_frame.size() < 2)
            {
                // delete feature from feature manager
                feature.erase(it);
                continue;
            }
            else //深度信息改变了（以最先观测到该特征点为起点计算的）
            {
                // 特征点在实际相机坐标系下的三维坐标
                Eigen::Vector3d pts_i = uv_i * depth; // feature in cartisian space in old local camera frame
                // 特征点在世界坐标系下的三维坐标
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // feautre in cartisian space in world frame
                // 转换到新的最老帧相机坐标系下的三维坐标
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // feature in cartisian space in shifted local camera frame
                double dep_j = pts_j(2);

                // after deletion, the feature has lidar depth in the first of the remaining frame
                if (it->feature_per_frame[0].depth > 0)
                {
                    it->estimated_depth = it->feature_per_frame[0].depth;
                    it->lidar_depth_flag = true;
                }
                // calculated depth in the current frame
                else if (dep_j > 0) //检查深度是否有效
                {
                    // 有效的话就得到在现在最老帧下的深度值
                    it->estimated_depth = dep_j;
                    it->lidar_depth_flag = false;
                }
                // non-positive depth, invalid
                else
                {
                    // 无效就设置默认值
                    it->estimated_depth = INIT_DEPTH; // 5.0
                    it->lidar_depth_flag = false;
                }
            }
        }
    }
}
//初始化还没结束，此时边缘化最老帧时，直接将特征点所保存的帧号向前滑动，
// 不进行地图点新的深度的换算
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        // 如果特征点起始帧号不为0则直接减一
        if (it->start_frame != 0)
            it->start_frame--;
        else //否则剔除feature_per_frame容器的头
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            //如果feature_per_frame为空则直接剔除该特征点
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}
//边缘化次新帧时，对特征点在次新帧的信息进行移除处理
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;
        // 如果起始帧为最新帧则将其滑动为次新帧
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            //
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

/**
 * @brief 计算某个特征点it_per_id在次新帧和次次新帧的视差
 *
 * @param it_per_id
 * @param frame_count
 * @return double
 */
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame];
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame];

    double ans = 0;
    Vector3d p_j = frame_j.point; //倒数第二帧j的3D路标点

    //因为特征点都是归一化之后的点，所以深度都为1，这里没有去除深度，下边去除深度，效果一样。
    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point; //倒数第三帧i的3D路标点
    Vector3d p_i_comp;

    // int r_i = frame_count - 2;
    // int r_j = frame_count - 1;
    // p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    //             cam^r_imu * imu^Rs_wolrd_j * wolrd^Rs_imu_i * imu^r_cam *  cam^P_point
    p_i_comp = p_i;
    double dep_i = p_i(2);
    // 归一化平面坐标
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    // 求得两个点的相对位移，即视差
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}