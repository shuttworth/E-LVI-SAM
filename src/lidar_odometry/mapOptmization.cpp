#include "utility.h"
#include "lvi_sam/cloud_info.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    ros::Publisher pubLaserCloudSurround;
    ros::Publisher pubOdomAftMappedROS;
    ros::Publisher pubKeyPoses;
    ros::Publisher pubPath;

    ros::Publisher pubHistoryKeyFrames;
    ros::Publisher pubIcpKeyFrames;
    ros::Publisher pubRecentKeyFrames;
    ros::Publisher pubRecentKeyFrame;
    ros::Publisher pubCloudRegisteredRaw;
    ros::Publisher pubLoopConstraintEdge;

    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subGPS;
    ros::Subscriber subLoopInfo;

    std::deque<nav_msgs::Odometry> gpsQueue;
    lvi_sam::cloud_info cloudInfo;

    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner featuer set from odoOptimization
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf featuer set from odoOptimization

    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    pcl::PointCloud<PointType>::Ptr latestKeyFrameCloud;
    pcl::PointCloud<PointType>::Ptr nearHistoryKeyFrameCloud;

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization
    
    ros::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;

    bool isDegenerate = false;
    cv::Mat matP;

    int laserCloudCornerLastDSNum = 0;
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    int imuPreintegrationResetId = 0;

    nav_msgs::Path globalPath;

    Eigen::Affine3f transPointAssociateToMap;

    map<int, int> loopIndexContainer; // from new to old
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;

    mapOptimization()
    {
        // 定义ISAM2参数类(ISAM2Params类)对象parameters
        ISAM2Params parameters;
        /** Only relinearize variables whose linear delta magnitude is greater than
         *  this threshold (default: 0.1).
        **/
        // 重线性化阈值relinearizeThreshold = 0.1   
        parameters.relinearizeThreshold = 0.1;
        ///< Only relinearize any variables every
        ///< relinearizeSkip calls to ISAM2::update (default:
        ///< 10)
        parameters.relinearizeSkip = 1;
        // gtsam::ISAM2 *isam;

        /* The typical cycle of using this class to create an instance :
           1.【by providing ISAM2Params to the constructor】, 即isam = new ISAM2(parameters);
           2.then add measurements and variables as they arrive using the update() method.
           3.At any time, calculateEstimate() may be
         * called to obtain the current estimate of all variables.
         */
        isam = new ISAM2(parameters);
        // 创建五个Publisher，并告诉将要发布的话题的名字
        // 1.轨迹
        pubKeyPoses           = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/trajectory", 1);
        // 2.全局地图
        pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_global", 1);
        // 3.里程计
        pubOdomAftMappedROS   = nh.advertise<nav_msgs::Odometry>      (PROJECT_NAME + "/lidar/mapping/odometry", 1);
        // 4.路径
        pubPath               = nh.advertise<nav_msgs::Path>          (PROJECT_NAME + "/lidar/mapping/path", 1);

        // 创建3个Subscriber，并告诉要订阅话题的名字
        // 1.featrueExtraction节点laserCloudInfoHandler发布的点云特征信息
        subLaserCloudInfo     = nh.subscribe<lvi_sam::cloud_info>     (PROJECT_NAME + "/lidar/feature/cloud_info", 5, &mapOptimization::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());
        // 2.gps信息 经纬度已经转化了
        subGPS                = nh.subscribe<nav_msgs::Odometry>      (gpsTopic,                                   50, &mapOptimization::gpsHandler, this, ros::TransportHints().tcpNoDelay());
        // 3.回环
        subLoopInfo           = nh.subscribe<std_msgs::Float64MultiArray>(PROJECT_NAME + "/vins/loop/match_frame", 5, &mapOptimization::loopHandler, this, ros::TransportHints().tcpNoDelay());

        // 再创建六个Publisher，并告诉将要发布的话题的名字
        // 5.历史关键帧
        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/loop_closure_history_cloud", 1);
        // 6.
        pubIcpKeyFrames       = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/loop_closure_corrected_cloud", 1);
        // 7.回环约束可视化信息
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>(PROJECT_NAME + "/lidar/mapping/loop_closure_constraints", 1);

        // 8.局部面地图
        pubRecentKeyFrames    = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/map_local", 1);
        // 9.当前帧点云
        pubRecentKeyFrame     = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered", 1);
        // 10.
        pubCloudRegisteredRaw = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/mapping/cloud_registered_raw", 1);

        // 设置降采样栅格大小
        // 角点云：0.2
        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        // 平面点云：0.4 
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        // 临近关键帧： 2.0
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        latestKeyFrameCloud.reset(new pcl::PointCloud<PointType>());
        nearHistoryKeyFrameCloud.reset(new pcl::PointCloud<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));
    }

    // 当接收到提取好特征的点云消息“PROJECT_NAME + "/lidar/feature/cloud_info" ” 就会调用该回调函数
    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr& msgIn)
    {
        // 整个函数围绕着处理点云信息消息msgIn
        
        // extract time stamp
        // 获取点云信息消息的时间戳(ros::Time类)timeLaserInfoStamp和时间点timeLaserInfoCur(double)
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = msgIn->header.stamp.toSec();

        // extract info ana feature cloud
        // 点云信息指针->点云信息cloudInfo
        cloudInfo = *msgIn;
        // 将sensor_msgs/PointCloud2类型的角点和面点转换为pcl格式的laserCloudCornerLast、laserCloudSurfLast
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        // 加线程锁
        std::lock_guard<std::mutex> lock(mtx);

        // 定义静态变量timeLastProcessing 初始化为-1，用来记录【上一次】处理点云信息消息的时间
        static double timeLastProcessing = -1;
        // 如果【当前】点云信息消息的时间点timeLaserInfoCur距离【上一次】处理点云信息消息的时间超过0.15s
        // 假设上一次点云时间timeLastProcessing 0.1s，则当前点云时间戳0.2s不满足，要下一帧点云时间戳0.3s才满足，
        // 才能进入if判断，然后将timeLastProcessing更新为0.3s
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval) {
            // 满足了时间条件才会更新【上一次】处理点云信息消息的时间timeLastProcessing
            timeLastProcessing = timeLaserInfoCur;

            /* 通过VINS IMU里程计的方式或者9轴IMU磁力计提供的姿态变换
               求解当前帧位姿的初值估计 存至transformTobeMapped
            */
            updateInitialGuess();

            /* 提取与【当前帧】相邻的关键帧surroundingKeyPosesDS，
             * 组成局部地图laserCloudCornerFromMap、laserCloudSurfFromMap
            */
            extractSurroundingKeyFrames();

            /* 对当前帧收到的角点云、面点云进行降采样，
               获取laserCloudCornerLastDS、laserCloudSurfLastDS
            */
            downsampleCurrentScan();

            // 求解当前帧位姿transformTobeMapped
            scan2MapOptimization();

            /* 对于关键帧，
             * 加入里程计因子、回环因子，进行全局位姿图优化，最新帧位姿保存至transformTobeMapped
             * 更新globalPath中的最新位姿
            */
            saveKeyFramesAndFactor();

            /* 遇到回环，就对所有关键帧位姿进行调整，
             * cloudKeyPoses3D、cloudKeyPoses6D、globalPath
             * 关键帧执行
            */
            correctPoses();

            // 相对高频的LO（5Hz）(transformTobeMapped)
            publishOdometry();

            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::Odometry::ConstPtr& gpsMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 affine3fTogtsamPose3(const Eigen::Affine3f& thisPose)
    {
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(thisPose, x, y, z, roll, pitch, yaw);
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(roll), double(pitch), double(yaw)),
                                  gtsam::Point3(double(x),    double(y),     double(z)));
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    { 
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    














    void visualizeGlobalMapThread()
    {
        ros::Rate rate(0.2);
        while (ros::ok()){
            rate.sleep();
            // 发布1km范围内的地图
            publishGlobalMap();
        }

        if (savePCD == false)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        // create directory and remove old files;
        // 在主机环境提供的环境列表中，搜索形参指向的C字符串，并返回指向与匹配的环境列表成员关联的C字符串指针
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str()); ++unused;
        // save key frame transformations
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        // extract global point cloud map        
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) 
        {
            // clip cloud
            // pcl::PointCloud<PointType>::Ptr cornerTemp(new pcl::PointCloud<PointType>());
            // pcl::PointCloud<PointType>::Ptr cornerTemp2(new pcl::PointCloud<PointType>());
            // *cornerTemp = *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            // for (int j = 0; j < (int)cornerTemp->size(); ++j)
            // {
            //     if (cornerTemp->points[j].z > cloudKeyPoses6D->points[i].z && cornerTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
            //         cornerTemp2->push_back(cornerTemp->points[j]);
            // }
            // pcl::PointCloud<PointType>::Ptr surfTemp(new pcl::PointCloud<PointType>());
            // pcl::PointCloud<PointType>::Ptr surfTemp2(new pcl::PointCloud<PointType>());
            // *surfTemp = *transformPointCloud(surfCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            // for (int j = 0; j < (int)surfTemp->size(); ++j)
            // {
            //     if (surfTemp->points[j].z > cloudKeyPoses6D->points[i].z && surfTemp->points[j].z < cloudKeyPoses6D->points[i].z + 5)
            //         surfTemp2->push_back(surfTemp->points[j]);
            // }
            // *globalCornerCloud += *cornerTemp2;
            // *globalSurfCloud   += *surfTemp2;

            // origin cloud
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            // \r：光标移到当前行的开头，不会换到下一行的开头，如果接着输出的话，本行以前的内容会被逐一覆盖
            // std::flush  把缓冲区的数据强行输出
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        // down-sample and save corner cloud
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloud);
        // down-sample and save surf cloud
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloud);
        // down-sample and save global point cloud map
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        downSizeFilterSurf.setInputCloud(globalMapCloud);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
        // "/lidar/mapping/map_global"
        if (pubLaserCloudSurround.getNumSubscribers() == 0)
            return;

        // 第一帧
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        // globalMapVisualizationSearchRadius 1km范围内
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            // 点云转到世界坐标系下拼接
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        // globalMapVisualizationLeafSize 1m
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        // "/lidar/mapping/map_global"
        publishCloud(&pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, "odom");    
    }



















    void loopHandler(const std_msgs::Float64MultiArray::ConstPtr& loopMsg)
    {
        // control loop closure frequency
        static double last_loop_closure_time = -1;
        {
            // std::lock_guard<std::mutex> lock(mtx);
            // 要保证两帧回环时间间隔超过5s
            if (timeLaserInfoCur - last_loop_closure_time < 5.0)
                return;
            else
                last_loop_closure_time = timeLaserInfoCur;
        }

        performLoopClosure(*loopMsg);
    }

    void performLoopClosure(const std_msgs::Float64MultiArray& loopMsg)
    {
        pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());
        {
            std::lock_guard<std::mutex> lock(mtx);
            // 将关键帧位姿集合拷贝出来
            *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        }

        // get lidar keyframe id
        int key_cur = -1; // latest lidar keyframe id
        int key_pre = -1; // previous lidar keyframe id
        {
            /* 根据回环消息loopMsg以及copy_cloudKeyPoses6D的时间戳
             * 在copy_cloudKeyPoses6D时间最相邻的关键帧索引key_cur key_pre
            */
            loopFindKey(loopMsg, copy_cloudKeyPoses6D, key_cur, key_pre);
            if (key_cur == -1 || key_pre == -1 || key_cur == key_pre)// || abs(key_cur - key_pre) < 25)
                return;
        }

        // check if loop added before
        {
            // if image loop closure comes at high frequency, many image loop may point to the same key_cur
            auto it = loopIndexContainer.find(key_cur);
            // 找到了
            if (it != loopIndexContainer.end())
                return;
        }
        
        // get lidar keyframe cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            loopFindNearKeyframes(copy_cloudKeyPoses6D, cureKeyframeCloud, key_cur, 0);
            // historyKeyframeSearchNum 25
            loopFindNearKeyframes(copy_cloudKeyPoses6D, prevKeyframeCloud, key_pre, historyKeyframeSearchNum);
            // 如果点云数目太少就算了
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            // "/lidar/mapping/loop_closure_history_cloud"
            if (pubHistoryKeyFrames.getNumSubscribers() != 0)
                // 把局部地图发布出去供rviz可视化使用
                publishCloud(&pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, "odom");
        }

        // get keyframe pose
        Eigen::Affine3f pose_cur;
        Eigen::Affine3f pose_pre;
        Eigen::Affine3f pose_diff_t; // serves as initial guess
        {
            // 取出新关键帧位姿
            pose_cur = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_cur]);
            // 旧关键帧位姿
            pose_pre = pclPointToAffine3f(copy_cloudKeyPoses6D->points[key_pre]);

            Eigen::Vector3f t_diff;
            // t_diff = pose_pre - pose_cur
            t_diff.x() = - (pose_cur.translation().x() - pose_pre.translation().x());
            t_diff.y() = - (pose_cur.translation().y() - pose_pre.translation().y());
            t_diff.z() = - (pose_cur.translation().z() - pose_pre.translation().z());
            // historyKeyframeSearchRadius 20m
            if (t_diff.norm() < historyKeyframeSearchRadius)
                t_diff.setZero();
            // 相当于将W'系的点云转到了一个W''系
            // pose_diff_t 可理解为tW''W'
            pose_diff_t = pcl::getTransformation(t_diff.x(), t_diff.y(), t_diff.z(), 0, 0, 0);
        }

        // transform and rotate cloud for matching
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        // pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
        // 设置对应点最大距离
        // 20m
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius * 2);
        // 设置最大迭代次数
        icp.setMaximumIterations(100);
        icp.setRANSACIterations(0);
        // 分别从长度和弧度定义了变换适量[x y z r p y]的最大容许差别(前后两次迭代)
        // 一旦变化减小到这个【临界值】以下，则配准算法会终止
        icp.setTransformationEpsilon(1e-3);
        // 设置前后两次迭代的点对集合的【欧氏距离】均值的最大容差
        // fabs (correspondences_cur_mse_ - correspondences_prev_mse_) / correspondences_prev_mse_ < mse_threshold_relative_
        // inline double
        // calculateMSE (const pcl::Correspondences &correspondences) const
        // {
        //   double mse = 0;
        //   for (size_t i = 0; i < correspondences.size (); ++i)
        //     mse += correspondences[i].distance;
        //   mse /= double (correspondences.size ());
        //   return (mse);
        // }
        icp.setEuclideanFitnessEpsilon(1e-3);

        // initial guess cloud
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud_new(new pcl::PointCloud<PointType>());
        // 相当于将W'系的点云转到了一个W''系
        // pose_diff_t 可理解为tW''W'
        pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud_new, pose_diff_t);

        // match using icp
        // W''系
        icp.setInputSource(cureKeyframeCloud_new);
        // W系
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);
        // "/lidar/mapping/loop_closure_corrected_cloud"
        if (pubIcpKeyFrames.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud_new, *closed_cloud, icp.getFinalTransformation());
            publishCloud(&pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, "odom");
        }

        // add graph factor
        // 0.3
        if (icp.getFitnessScore() < historyKeyframeFitnessScore && icp.hasConverged() == true)
        {
            // get gtsam pose
            // TWLk = TWW''*TW''W'*TW'Lk
            gtsam::Pose3 poseFrom = affine3fTogtsamPose3(Eigen::Affine3f(icp.getFinalTransformation()) * pose_diff_t * pose_cur);
            // 历史帧位姿TWLh
            gtsam::Pose3 poseTo   = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[key_pre]);
            // get noise
            float noise = icp.getFitnessScore();
            gtsam::Vector Vector6(6);
            Vector6 << noise, noise, noise, noise, noise, noise;
            noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);
            // save pose constraint
            mtx.lock();
            // 回环帧索引对
            loopIndexQueue.push_back(make_pair(key_cur, key_pre));
            // Twlk-1 * Twlh = Tlklh 历史帧到新帧
            loopPoseQueue.push_back(poseFrom.between(poseTo));
            loopNoiseQueue.push_back(constraintNoise);
            mtx.unlock();
            // add loop pair to container
            loopIndexContainer[key_cur] = key_pre;
        }

        // visualize loop constraints
        if (!loopIndexContainer.empty())
        {
            // 定义visualization_msgs::MarkerArray类对象markerArray
            visualization_msgs::MarkerArray markerArray;
            // loop nodes
            // 定义visualization_msgs::Marker类对象markerNode
            visualization_msgs::Marker markerNode;
            markerNode.header.frame_id = "odom";
            markerNode.header.stamp = timeLaserInfoStamp;
            markerNode.action = visualization_msgs::Marker::ADD;
            markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
            markerNode.ns = "loop_nodes";
            markerNode.id = 0;
            markerNode.pose.orientation.w = 1;
            markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
            markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
            markerNode.color.a = 1;
            // loop edges
            // 定义visualization_msgs::Marker类对象markerEdge
            visualization_msgs::Marker markerEdge;
            markerEdge.header.frame_id = "odom";
            markerEdge.header.stamp = timeLaserInfoStamp;
            markerEdge.action = visualization_msgs::Marker::ADD;
            markerEdge.type = visualization_msgs::Marker::LINE_LIST;
            markerEdge.ns = "loop_edges";
            markerEdge.id = 1;
            markerEdge.pose.orientation.w = 1;
            markerEdge.scale.x = 0.1;
            markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
            markerEdge.color.a = 1;

            for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
            {
                int key_cur = it->first;
                int key_pre = it->second;
                geometry_msgs::Point p;
                p.x = copy_cloudKeyPoses6D->points[key_cur].x;
                p.y = copy_cloudKeyPoses6D->points[key_cur].y;
                p.z = copy_cloudKeyPoses6D->points[key_cur].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
                p.x = copy_cloudKeyPoses6D->points[key_pre].x;
                p.y = copy_cloudKeyPoses6D->points[key_pre].y;
                p.z = copy_cloudKeyPoses6D->points[key_pre].z;
                markerNode.points.push_back(p);
                markerEdge.points.push_back(p);
            }

            markerArray.markers.push_back(markerNode);
            markerArray.markers.push_back(markerEdge);
            // "/lidar/mapping/loop_closure_constraints"
            pubLoopConstraintEdge.publish(markerArray);
        }
    }

    void loopFindNearKeyframes(const pcl::PointCloud<PointTypePose>::Ptr& copy_cloudKeyPoses6D,
                               pcl::PointCloud<PointType>::Ptr& nearKeyframes, 
                               const int& key, const int& searchNum)
    {
        // extract near keyframes
        // 移除局部地图形参nearKeyframes中所有角点
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            // 获取遍历到的关键帧索引key_near
            int key_near = key + i;
            /* 如果遍历到的关键帧索引key_near不属于[0,cloudSize)，
               则该关键帧索引无实际意义，则通过continue进入下次遍历
            */
            if (key_near < 0 || key_near >= cloudSize )
                continue;
            /* 将遍历到的关键帧索引key_near对应的角点云cornerCloudKeyFrames[key_near]
             *                               面点云surfCloudKeyFrames[key_near]
             * 按照其对应位姿copy_cloudKeyPoses6D->points[key_near]转至世界坐标系
             * 叠加到局部地图形参nearKeyframes上
             */
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[key_near], &copy_cloudKeyPoses6D->points[key_near]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[key_near],   &copy_cloudKeyPoses6D->points[key_near]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        // 0.4
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void loopFindKey(const std_msgs::Float64MultiArray& loopMsg, 
                     const pcl::PointCloud<PointTypePose>::Ptr& copy_cloudKeyPoses6D,
                     int& key_cur, int& key_pre)
    {
        if (loopMsg.data.size() != 2)
            return;

        double loop_time_cur = loopMsg.data[0];
        double loop_time_pre = loopMsg.data[1];

        // historyKeyframeSearchTimeDiff 30
        if (abs(loop_time_cur - loop_time_pre) < historyKeyframeSearchTimeDiff)
            return;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return;

        // latest key
        key_cur = cloudSize - 1;
        // 从后往前遍历关键帧位姿集合中的位姿
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            // 对于时间晚于loop_time_cur所有的关键帧位姿
            if (copy_cloudKeyPoses6D->points[i].time > loop_time_cur)
            // 记录下时间戳晚于loop_time_cur的最早关键帧索引
                key_cur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        key_pre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time < loop_time_pre)
                key_pre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }
    }

    void loopClosureThread()
    {
        if (loopClosureEnableFlag == false)
            return;

        ros::Rate rate(0.5);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosureDetection();
        }
    }

    void performLoopClosureDetection()
    {
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;

        int key_cur = -1;
        int key_pre = -1;

        double loop_time_cur = -1;
        double loop_time_pre = -1;

        // find latest key and time
        {
            std::lock_guard<std::mutex> lock(mtx);

            // 第一帧
            if (cloudKeyPoses3D->empty())
                return;
            
            kdtreeHistoryKeyPoses->setInputCloud(cloudKeyPoses3D);
            // historyKeyframeSearchRadius： 20.0m
            kdtreeHistoryKeyPoses->radiusSearch(cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);

            key_cur = cloudKeyPoses3D->size() - 1;
            loop_time_cur = cloudKeyPoses6D->points[key_cur].time;
        }

        // find previous key and time
        {
            for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
            {
                int id = pointSearchIndLoop[i];
                // historyKeyframeSearchTimeDiff 30s
                if (abs(cloudKeyPoses6D->points[id].time - loop_time_cur) > historyKeyframeSearchTimeDiff)
                {
                    key_pre = id;
                    loop_time_pre = cloudKeyPoses6D->points[key_pre].time;
                    break;
                }
            }
        }

        if (key_cur == -1 || key_pre == -1 || key_pre == key_cur ||
            loop_time_cur < 0 || loop_time_pre < 0)
            return;

        std_msgs::Float64MultiArray match_msg;
        match_msg.data.push_back(loop_time_cur);
        match_msg.data.push_back(loop_time_pre);
        performLoopClosure(match_msg);
    }






















    



    void updateInitialGuess()
    {        
        static Eigen::Affine3f lastImuTransformation;
        // system initialization
        // cloudKeyPoses3D是pcl点云格式的关键帧位置集合
        // 无关键帧时
        if (cloudKeyPoses3D->points.empty())
        {
            // 初始位姿由IMU磁力计提供（9轴IMU的用处）
            transformTobeMapped[0] = cloudInfo.imuRollInit;
            transformTobeMapped[1] = cloudInfo.imuPitchInit;
            transformTobeMapped[2] = cloudInfo.imuYawInit;

            // 可以选择不用9轴IMU磁力计的yaw角初始化姿态
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            // 将IMU磁力计提供的欧拉角＋平移置0转换为Eigen::Affine3f格式的静态变量lastImuTransformation
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            /* 第一帧的初始位姿估计执行至此就完成了【更新初值估计】的操作了，
               就可以退出updateInitialGuess()函数了
            */
            return;
        }
        // 【和LIO-SAM不同点：】
        // use VINS odometry estimation for pose guess
        static int odomResetId = 0;
        static bool lastVinsTransAvailable = false;
        static Eigen::Affine3f lastVinsTransformation;

        // 如果里程计可用
        if (cloudInfo.odomAvailable == true && cloudInfo.odomResetId == odomResetId)
        {
            // 如果里程计可用,去检查是否【第一次】收到VINS提供的里程计消息 分成了两个if else
            // ROS_INFO("Using VINS initial guess");
            // lastVinsTransAvailable == false表示【第一次】收到VINS提供的里程计消息
            if (lastVinsTransAvailable == false)
            {
                // ROS_INFO("Initializing VINS initial guess");
                // 如果【第一次】收到VINS提供的里程计消息
                // 将VINS IMU递推的里程计(xyz+欧拉角)转换为Eigen::Affine3f的lastVinsTransformation
                // 注意寻找cloudInfo.odom***数据来源的方式
                lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);
                // 已经收到过VINS IMU递推的里程计，则将lastVinsTransAvailable标志变量置为true
                lastVinsTransAvailable = true;
            } 
            else 
            {
                // ROS_INFO("Obtaining VINS incremental guess");
                // 程序执行到这里，说明已经不是【第一次】收到VINS提供的里程计消息
                // 则将【当前】VINS IMU递推的里程计转换为Eigen::Affine3f的transBack
                Eigen::Affine3f transBack = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                   cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);
                // 获取VINS里程计相对于上一次的里程计增量transIncre
                // Tvins_k-1 * Tvins_k = T_vins_k-1_k   
                Eigen::Affine3f transIncre = lastVinsTransformation.inverse() * transBack;

                // 将上一帧优化结束获得的关键帧估计结果transformTobeMapped转换为Eigen::Affine3f格式的transTobe
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                // Twlk^ = Twlk-1 * Tlk-1lk
                Eigen::Affine3f transFinal = transTobe * transIncre;
                // 将Eigen::Affine3f格式的transFinal转换为数组形式
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
                // 【当前】VINS IMU递推的里程计 -> lastVinsTransformation（新旧交替）
                lastVinsTransformation = pcl::getTransformation(cloudInfo.odomX,    cloudInfo.odomY,     cloudInfo.odomZ, 
                                                                cloudInfo.odomRoll, cloudInfo.odomPitch, cloudInfo.odomYaw);

                // 虽然有vins里程计信息，但仍然需要把IMU磁力计提供的姿态转换为Eigen::Affine3f格式的lastImuTransformation
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
                // 非第一帧的VINS里程计 初始位姿估计完成，可以退出了
                return;
            }
        } 
        // VINS IMU里程计不可用的情况
        else
        {
            // ROS_WARN("VINS failure detected.");
            lastVinsTransAvailable = false;
            odomResetId = cloudInfo.odomResetId;
        }

        // use imu incremental estimation for pose guess (only rotation)
        // 如果没有里程记信息，就是用imu的旋转信息来更新，因为单纯使用imu无法得到靠谱的平移信息，因此，平移直接置0
        if (cloudInfo.imuAvailable == true)
        {
            // ROS_INFO("Using IMU initial guess");
            // 获取【最新的】IMU磁力计提供的位姿transBack
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit);
            // 获取IMU旋转变化量
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            // 将上一帧优化结束获取的最优位姿估计值transformTobeMapped转换为Eigen::Affine3f的transTobe
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            // Twlk^ = Twlk-1 * Tlk-1lk
            Eigen::Affine3f transFinal = transTobe * transIncre;
            // 当前帧初始估计结果transFinal再转换赋值给transFinal
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            // 仍然记录本次IMU磁力计提供的欧拉角存至lastImuTransformation
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imuRollInit, cloudInfo.imuPitchInit, cloudInfo.imuYawInit); // save imu before return;
            // 当前帧初始位姿估计结果已定，return退出updateInitialGuess()函数
            return;
        }
    }

    void extractNearby()
    {
        // 定义存储【当前帧】附近(指定半径范围内)的【关键帧位置信息】的点云指针(集合)
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        // 定义存储临近关键帧在cloudKeyPoses3D中的索引的vector
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        // 用存储关键帧位置信息的点云/集合，用以构建KD-Tree
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        /* 根据最新关键帧位置cloudKeyPoses3D->back()提取关键帧位置集合cloudKeyPoses3D中
           距离最后一个关键帧指定半径范围(50.0m)内的所有关键帧
           找到的关键帧
           在cloudKeyPoses3D中的索引存至pointSearchInd
           距离最新关键帧距离的平方存至pointSearchSqDis
        */
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        // 遍历存储索引的vector pointSearchInd
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            // 获取当前遍历到的关键帧在cloudKeyPoses3D中的索引id
            int id = pointSearchInd[i];
            // 将该关键帧存至surroundingKeyPoses
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }
        
        /* 避免50m范围内关键帧过密,对提取出的临近关键帧降采样，每2*2*2m^3范围选出一个关键帧代表，
            得到surroundingKeyPosesDS
            ToDO
        */
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);

        /*
         * 【和LIO-SAM不同点：】
         * 少了这段：
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }
        */   

        // also extract some latest key frames in case the robot rotates in one position
        // 上面的surroundingKeyPosesDS只能针对【位置】上选出邻近范围内的关键帧
        // 如果出现原地旋转的情况 则这些姿态不同的关键帧只能根据位置选一个
        // 为了尽量选出姿态不同的关键帧进入surroundingKeyPosesDS，
        int numPoses = cloudKeyPoses3D->size();
        // 从新往旧遍历关键帧集合
        for (int i = numPoses-1; i >= 0; --i)
        {
            // 选出距离当前帧点云时间timeLaserInfoCur小于10s的关键帧也塞入surroundingKeyPosesDS
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            // 超过10s后直接break结束遍历循环
            else
                break;
        }

        // 根据筛选出来的近邻关键帧surroundingKeyPosesDS进行局部地图构建
        /* surroundingKeyPosesDS只是一个关键帧位置集合，
           要得到局部地图还要整合其对应点云
           得到laserCloudCornerFromMap、laserCloudSurfFromMap
        */
        extractCloud(surroundingKeyPosesDS);
    }

    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        std::vector<pcl::PointCloud<PointType>> laserCloudCornerSurroundingVec;
        std::vector<pcl::PointCloud<PointType>> laserCloudSurfSurroundingVec;

        // 将这个vector大小设置为局部地图包含的点云帧数
        laserCloudCornerSurroundingVec.resize(cloudToExtract->size());
        laserCloudSurfSurroundingVec.resize(cloudToExtract->size());

        // extract surrounding map
        #pragma omp parallel for num_threads(numberOfCores)
        // 遍历【筛选出来的关键帧集合】cloudToExtract
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 获取遍历到的关键帧索引thisKeyInd(intensity)
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            // 距离当前帧位置超过50m的关键帧不予考虑
            if (pointDistance(cloudKeyPoses3D->points[thisKeyInd], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            // cornerCloudKeyFrames是在 saveKeyFramesAndFactor中被赋值
            // 将遍历到的关键帧按照其【位姿】转到世界坐标系，加入到点云vector中
            laserCloudCornerSurroundingVec[i]  = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            laserCloudSurfSurroundingVec[i]    = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }

        // fuse the map
        // 首先移除角点云、面点云局部地图中的所有点
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        // 遍历【筛选出来的关键帧集合】cloudToExtract
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 将转到世界坐标系的角点云、面点云 加入到各自局部地图
            *laserCloudCornerFromMap += laserCloudCornerSurroundingVec[i];
            *laserCloudSurfFromMap   += laserCloudSurfSurroundingVec[i];
        }

        // 对局部地图进行降采样
        // 角点云：0.2
        // 平面点云：0.4 
        // Downsample the surrounding corner key frames (or map)
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        // Downsample the surrounding surf key frames (or map)
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
    }

    void extractSurroundingKeyFrames()
    {
        /* 如果当前帧是第一帧，则直接return退出extractSurroundingKeyFrames()
           因为第一帧无需提取周围的局部地图
        */
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        /* 将【最新关键帧】周围50m内的关键帧提取出来，
          并将这些关键帧对应点云整合成局部地图laserCloudCornerFromMap、laserCloudSurfFromMap
        */
        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        // laserCloudCornerLast、laserCloudSurfLast是在处理点云回调函数一开始就从形参转化而来

        // 对当前帧角面点云进行降采样
        // 角点云：0.2
        // 平面点云：0.4 
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }

    void cornerOptimization()
    {
        /* 前面在updateInitialGuess()函数中已经得到当前帧Twl的预测值transformTobeMapped了，
           现在需要将其转换为Eigen::Affine3f格式的transPointAssociateToMap
        */
        updatePointAssociateToMap();
        // 接下来的这个for循环将被多个线程同时运行的，也就是多个线程同时运行一个for循环
        // 并行执行for循环
        #pragma omp parallel for num_threads(numberOfCores)
        // 遍历当前帧降采样后的角点云
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 取出降采样后的当前帧角点云中 遍历到的第i个点pointOri
            pointOri = laserCloudCornerLastDS->points[i];
            // 按照当前帧预测的位姿transPointAssociateToMap将pointOri转换到W'系，得到pointSel
            pointAssociateToMap(&pointOri, &pointSel);
            // 在构建的角点局部地图laserCloudCornerFromMapDS中找到距离pointSel最近的5个点
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
            
            // 如果找到的5个点中距离pointSel的最远点也<1m     
            if (pointSearchSqDis[4] < 1.0)
            {
                // 首先计算角点局部地图laserCloudCornerFromMapDS中这5个最近点的均值
                float cx = 0, cy = 0, cz = 0;
                // 首先遍历五个最近点，求出5个最近点x、y、z坐标的均值
                for (int j = 0; j < 5; j++)
                {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                // 计算这5个点的协方差矩阵matA1
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 求解这5个点协方差矩阵的特征值matD1和特征向量matV1
                cv::eigen(matA1, matD1, matV1);

                // 如果最大特征值 > 3*次大特征值，则认为这5个点呈线性分布
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
                {
                    // 转到世界坐标系的遍历到的点O(x0,y0,z0)
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 获取这五个点所在的线的两个端点的坐标 A(x1,y1,z1)  B(x2,y2,z2)
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // 计算|OM->|=|AO×BO| 
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
                    
                    // 计算|BA->|
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // 计算BA×OM/|BA×OM| ，即O到AB的垂线的反方向的单位向量
                    // 其x、y、z坐标分别为la、lb、lc
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    // |AO×BO|/|BA->| 即O到AB距离
                    float ld2 = a012 / l12;

                    // 可以理解为残差的权重
                    float s = 1 - 0.9 * fabs(ld2);

                    // coeff为pcl::PointXYZI类型
                    // la、lb、lc为O到AB距离的反方向的单位向量
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    // |AO×BO|/|BA->| 即O到AB距离
                    coeff.intensity = s * ld2;

                    // 如果残差小于1m，则将
                    if (s > 0.1)
                    {
                        // 将当前点LiDAR坐标系的坐标pointOri保存至laserCloudOriCornerVec
                        laserCloudOriCornerVec[i] = pointOri;
                        // 将带有残差信息的coeff存至coeffSelCornerVec
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }

            // A-LOAM的做法：
            /************************************************
            for (int i = 0; i < laserCloudCornerStackNum; i++)
			{
                pointOri = laserCloudCornerStack->points[i];
                //double sqrtDis = pointOri.x * pointOri.x + pointOri.y * pointOri.y + pointOri.z * pointOri.z;
                pointAssociateToMap(&pointOri, &pointSel);
                kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis); 

                if (pointSearchSqDis[4] < 1.0)
                { 
                    std::vector<Eigen::Vector3d> nearCorners;
                    Eigen::Vector3d center(0, 0, 0);
                    for (int j = 0; j < 5; j++)
                    {
                        Eigen::Vector3d tmp(laserCloudCornerFromMap->points[pointSearchInd[j]].x,
                                            laserCloudCornerFromMap->points[pointSearchInd[j]].y,
                                            laserCloudCornerFromMap->points[pointSearchInd[j]].z);
                        center = center + tmp;
                        nearCorners.push_back(tmp);
                    }
                    center = center / 5.0;

                    Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
                    for (int j = 0; j < 5; j++)
                    {
                        Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                        covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                    }

                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                    // if is indeed line feature
                    // note Eigen library sort eigenvalues in increasing order
                    Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
                    Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                    if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                    { 
                        Eigen::Vector3d point_on_line = center;
                        Eigen::Vector3d point_a, point_b;
                        point_a = 0.1 * unit_direction + point_on_line;
                        point_b = -0.1 * unit_direction + point_on_line;

                        ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, point_a, point_b, 1.0);
                        problem.AddResidualBlock(cost_function, loss_function, parameters, parameters + 4);
                        corner_num++;	
                    }							
                } 
            */
        }
    }

    void surfOptimization()
    {
        /* 前面在updateInitialGuess()函数中已经得到当前帧Twl的预测值transformTobeMapped了，
           现在需要将其转换为Eigen::Affine3f格式的transPointAssociateToMap
        */
        updatePointAssociateToMap();

        // 并行执行for循环
        #pragma omp parallel for num_threads(numberOfCores)
        // 遍历所有面点
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 取出降采样后的当前帧面点云中 遍历到的第i个点pointOri
            pointOri = laserCloudSurfLastDS->points[i];
            // 按照当前帧预测的位姿transPointAssociateToMap将pointOri转换到W'系，得到pointSel
            pointAssociateToMap(&pointOri, &pointSel); 
            // 在构建的角点局部地图laserCloudSurfFromMapDS中找到距离pointSel最近的5个点
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            // 求解平面Ax+By+Cz+1 = 0中的ABC
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            if (pointSearchSqDis[4] < 1.0)
            {
                for (int j = 0; j < 5; j++)
                {
                    /****************
                     *matA0 5*3
                        x1 y1 z1
                        x2 y2 z2
                        x3 y3 z3
                        x4 y4 z4
                        x5 y5 z5
                    */
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }
                
                // Ax=B 求解x ABC
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                // A、B、C
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;

                // √(A2+B2+C2)
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 默认该次平面拟合有效
                bool planeValid = true;
                for (int j = 0; j < 5; j++)
                {
                    // 如果有一个点到该拟合平面超过0.2m,就认为该次平面拟合失败
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2)
                    {
                        planeValid = false;
                        // 退出五个点的判断
                        break;
                    }
                }
                
                // 如果该拟合平面成功
                if (planeValid)
                {
                    // 计算当前帧遍历到的w系点到拟合平面的距离（不带绝对值）
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    // 残差越大，权重越小
                    // 距离LiDAR越远，权重越大
                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));

                    // 计算带有权重的平面法向量
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    // 带有权重的残差 （不是绝对值）
                    coeff.intensity = s * pd2;

                    if (s > 0.1) 
                    {
                        // 将当前点LiDAR坐标系的坐标pointOri保存至laserCloudOriSurfVec
                        laserCloudOriSurfVec[i] = pointOri;
                        // 将带有残差信息的coeff存至coeffSelCornerVec
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        // 遍历局部地图所有角点
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i)
        {
            // 只有呈线性分布的角点
            if (laserCloudOriCornerFlag[i] == true)
            {
                // 将当前帧 遍历到的 角点 在LiDAR坐标系的坐标push进laserCloudOri
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                // 将带有权重信息的残差push入coeffSel
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        // 遍历局部地图所有面点
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i)
        {
            // 只有拟合结果合理的平面
            if (laserCloudOriSurfFlag[i] == true)
            {
                // 将当前帧 遍历到的 面点 在LiDAR坐标系的坐标push进laserCloudOri
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                // 将带有权重信息的残差push入coeffSel
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        // 将判断线、面特征分布是否合理的标志变量vector全部置为false
        /* 注：
            目前在执行的函数是combineOptimizationCoeffs()，
            在执行完combineOptimizationCoeffs()之后，会执行LMOptimization()。
            而LMOptimization()构建增量方程用到的约束个数就是由laserCloudOri的数量决定的。
            而只有在laserCloudOriCornerFlag[i]、laserCloudOriSurfFlag[i]为true才会将点塞入laserCloudOri。
            而laserCloudOriCornerFlag[i]、laserCloudOriSurfFlag[i]
               是在cornerOptimization()和surfOptimization()中判断s > 0.1时才置为true。

            即 
            1.cornerOptimization()和surfOptimization() （都是在scan2MapOptimization()函数中）
              s > 0.1 （并不是所有角点和面点都满足这个条件）
              laserCloudOriCornerFlag[i]、laserCloudOriSurfFlag[i]置为true
            2.combineOptimizationCoeffs()
              分别遍历角点和面点，
              只记录laserCloudOriCornerFlag[i]、laserCloudOriSurfFlag[i]置为true、即满足s > 0.1的角点和面点
               对应的特征点和残差分别塞入laserCloudOri、coeffSel
              即执行完该函数laserCloudOriCornerFlag[i]、laserCloudOriSurfFlag[i]作用就失效了，
              一朝 当前帧点云 一朝 laserCloudOriCornerFlag[i]、laserCloudOriSurfFlag[i]
              因此，为了下帧点云使用，需要提前将laserCloudOriCornerFlag[i]、laserCloudOriSurfFlag[i]置为false

            3.LMOptimization()
              求雅克比和残差用到的都在laserCloudOri和coeffSel中取
        */
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }

    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne, need to cope with coordinate transformation
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        // 计算这些欧拉角在camera坐标系的正弦、余弦值
        // 【画图示意】
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        int laserCloudSelNum = laserCloudOri->size();
        // 确保可以构建增量方程的有效观测超过50个
        // 不满足的话直接退出LMOptimization
        if (laserCloudSelNum < 50) {
            return false;
        }

        // √[Σ^(-1)]J
        // 【公式推导】
        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        // 构造H△x=b
        // 遍历存储【当前帧】角/面特征的点云laserCloudOri中laserCloudSelNum个点
        for (int i = 0; i < laserCloudSelNum; i++)
        {
            // 求解遍历到的LiDAR系下的点laserCloudOri、残差coeffSel
            // 在camera系下的点坐标pointOri以及残差coeff
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            // arx = ∂e/∂rxc = (∂e/∂pw)*(∂pw/∂rxc)
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            // arz = ∂e/∂rzc = (∂e/∂pw)*(∂pw/∂rzc)
            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            // √[Σ^(-1)]J
            // 1/σi JiL = 1/σi ∂e/∂rxL = 1/σi ∂e/∂rzc
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            // -1/σi ei L = -coeff.intensity
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        // JT[Σ^(-1)]J
        matAtA = matAt * matA;
        // -JT[Σ^(-1)]e
        matAtB = matAt * matB;
        // JT[Σ^(-1)]J△x = -JT[Σ^(-1)]e
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // 如果是第一次迭代，判断JT[Σ^(-1)]J的最小特征值
        if (iterCount == 0) 
        {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            // 求解JT[Σ^(-1)]J的特征值填入matE，特征值填入matV
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--)
            {
                if (matE.at<float>(0, i) < eignThre[i])
                {
                    for (int j = 0; j < 6; j++)
                    {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                }
                else
                {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        // 如果发生退化，就把求解出来的解投影到未退化的方向上再进行线性组合，即舍弃掉退化的方向
        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        // 将求解出来的 rpyxyz的增量加到transformTobeMapped
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        // 本次迭代旋转变化量
        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        // 位移变化量(cm)
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }

    void scan2MapOptimization()
    {
        // 如果当前帧是第一帧，自然没法scan-to-map，退出
        if (cloudKeyPoses3D->points.empty())
            return;

        // 只有当前帧角点云、面点云特征分别大于10、100,才能进行scan-to-map优化
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // 用刚构建的角、面局部地图分别构建KD-Tree
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // 以下操作迭代计算30次
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                laserCloudOri->clear();
                coeffSel->clear();

                // 求解遍历到的角点对应的残差的反方向向量与残差大小
                cornerOptimization();

                // 得到了平面法向量以及残差(无绝对值)
                surfOptimization();

                /* 将符合条件的角点以及面点
                   在LiDAR系坐标push入laserCloudOri
                   带有权重的残差push入coeffSel
                */
                combineOptimizationCoeffs();

                // 构建H△x=b方程，求解△x，并更新到待优化变量transformTobeMapped上
                if (LMOptimization(iterCount) == true)
                    break;              
            }

            // 迭代优化30次后，再对优化结果transformTobeMapped做一个限制
            transformUpdate();
        } 
        // 对于某一种特征不足的情况，相当于并没有执行scan2MapOptimization()，直接报警告
        else
        {
            ROS_WARN("Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }

    void transformUpdate()
    {
        // 如果磁力计可用
        if (cloudInfo.imuAvailable == true)
        {
            // 如果磁力计pitch角 < 1.4°
            if (std::abs(cloudInfo.imuPitchInit) < 1.4)
            {
                // IMU权重
                double imuWeight = 0.01;
                // IMU磁力计提供的位姿
                tf::Quaternion imuQuaternion;
                // LO计算估计得到的位姿
                tf::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imuRollInit, 0, 0);
                // tf::Quaternion::slerp
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imuPitchInit, 0);
                tf::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }
        // 对roll pitch z做一个限制
        // 1000 rad 57.3
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        // z_tollerance 1000m
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }

    bool saveFrame()
    {
        // 如果当前是第一帧，自然是关键帧（返回true）
        if (cloudKeyPoses3D->points.empty())
            return true;

        // 计算最新关键帧的位姿
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 计算目前帧优化后的位姿
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 两帧之间的位姿变换 Eigen::Affine3f格式
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        // 两帧之间的位姿变换 转换为x, y, z, roll, pitch, yaw
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // r p y 或者距离中任何一个超过阈值 就是关键帧 返回true
        // surroundingkeyframeAddingAngleThreshold： 0.2rad 0.2*57.3 = 11.5°
        // surroundingkeyframeAddingDistThreshold：1m
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        // 针对第一帧的情况，添加的是先验约束
        if (cloudKeyPoses3D->points.empty())
        {
            // 设置先验噪声 yaw和位置方差设置较大
            // Diagonal类继承自Gaussian类
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            // gtsam::NonlinearFactorGraph
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            // gtsam::Values
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }

        // 如果不是第一帧，则添加帧间约束
        else
        {
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            // 上一帧关键帧位姿
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            // 当前最新关键帧位姿
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            // 这里不确定写的对不对，初值给的是最新帧位姿？而不应该是帧间位姿？
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
            // if (isDegenerate)
            // {
                // adding VINS constraints is deleted as benefits are not obvious, disable for now
                // gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), vinsPoseFrom.between(vinsPoseTo), odometryNoise));
            // }
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (gpsQueue.front().header.stamp.toSec() < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (gpsQueue.front().header.stamp.toSec() > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;

                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;

                break;
            }
        }
    }

    void addLoopFactor()
    {
        // vector<pair<int, int>> loopIndexQueue;
        // 如果存储【回环帧索引对】的容器为空，直接退出，不添加因子
        if (loopIndexQueue.empty())
            return;

        // 遍历 存储【回环帧索引对】的容器
        for (size_t i = 0; i < loopIndexQueue.size(); ++i)
        {
            // 取出当前帧的索引
            int indexFrom = loopIndexQueue[i].first;
            // 取出历史帧的索引
            int indexTo = loopIndexQueue[i].second;
            // 获取【历史帧】到【新帧】的位姿约束
            // vector<gtsam::Pose3> loopPoseQueue;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            // 获取位姿约束的噪声
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }

    void saveKeyFramesAndFactor()
    {
        // 如果当前帧不是关键帧，则直接return 退出
        if (saveFrame() == false)
            return;

        // odom factor
        addOdomFactor();

        // gps factor
        // addGPSFactor();

        // loop factor
        addLoopFactor();

        // update iSAM
        // add measurements and variables
        // add measurements and variables as they arrive using the update() method.
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();
        
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // obtain the current estimate of all variables
        // gtsam::ISAM2 isam
        // gtsam::Values
        isamCurrentEstimate = isam->calculateEstimate();
        // gtsam::Values isamCurrentEstimate;
        // 从优化结果isamCurrentEstimate取出最新的位姿估计结果latestEstimate
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        // 给关键帧位置集合添值
        cloudKeyPoses3D->push_back(thisPose3D);

        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        // 获取最新位姿估计的协方差
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        // 获取最新位姿估计结果存至transformTobeMapped
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        // 将当前帧降采样后的角点和面点存至thisCornerKeyFrame、thisSurfKeyFrame
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        // vector<pcl::PointCloud<PointType>::Ptr>
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        // 更新了globalPath
        // 遇到关键帧后进行全局位姿图优化，之后对globalPath进行更新
        updatePath(thisPose6D);
    }

    void correctPoses()
    {
        // 第一帧不执行这个函数
        if (cloudKeyPoses3D->points.empty())
            return;

        // 如果检测到回环
        if (aLoopIsClosed == true)
        {
            // clear path
            globalPath.poses.clear();

            // update key poses
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                // 更新所有历史【关键帧】位姿至cloudKeyPoses3D
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                // 顺便也更新globalPath
                updatePath(cloudKeyPoses6D->points[i]);
            }

            // 标志位复位
            aLoopIsClosed = false;
            // ID for reseting IMU pre-integration
            // 暂时不知道这个标志变量的作用
            ++imuPreintegrationResetId;
        }
    }

    void publishOdometry()
    {
        // Publish odometry for ROS
        nav_msgs::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = "odom";
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        laserOdometryROS.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 协方差设置成imuPreintegrationResetId
        // 检测到回环/全局位姿修正的次数
        laserOdometryROS.pose.covariance[0] = double(imuPreintegrationResetId);
        // /lidar/mapping/odometry"
        pubOdomAftMappedROS.publish(laserOdometryROS);
        // Publish TF
        static tf::TransformBroadcaster br;
        tf::Transform t_odom_to_lidar = tf::Transform(tf::createQuaternionFromRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]),
                                                      tf::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf::StampedTransform trans_odom_to_lidar = tf::StampedTransform(t_odom_to_lidar, timeLaserInfoStamp, "odom", "lidar_link");
        br.sendTransform(trans_odom_to_lidar);
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time().fromSec(pose_in.time);
        pose_stamped.header.frame_id = "odom";
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        // "/lidar/mapping/trajectory"
        // 经过修正的位姿(点云形式)
        publishCloud(&pubKeyPoses, cloudKeyPoses6D, timeLaserInfoStamp, "odom");
        // Publish surrounding key frames
        // "/lidar/mapping/map_local"
        // 局部面地图
        publishCloud(&pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, "odom");
        // publish registered key frame
        // "/lidar/mapping/cloud_registered"
        if (pubRecentKeyFrame.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            // 将当前帧角点云、面点云转至W系发布出去
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            // "/lidar/mapping/cloud_registered"
            publishCloud(&pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish registered high-res raw cloud
        // "/lidar/mapping/cloud_registered_raw"
        if (pubCloudRegisteredRaw.getNumSubscribers() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            // 将当前帧去畸变原始点云转至W系发布出去
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(&pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, "odom");
        }
        // publish path
        // "/lidar/mapping/path"
        if (pubPath.getNumSubscribers() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = "odom";
            pubPath.publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidar");

    mapOptimization MO;

    ROS_INFO("\033[1;32m----> Lidar Map Optimization Started.\033[0m");
    
    std::thread loopDetectionthread(&mapOptimization::loopClosureThread, &MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, &MO);

    ros::spin();

    loopDetectionthread.join();
    visualizeMapThread.join();

    return 0;
}