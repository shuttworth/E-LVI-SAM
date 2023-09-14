#include "utility.h"
#include "lvi_sam/cloud_info.h"

struct smoothness_t
{
    float value; // 曲率值
    size_t ind;  // 点序值
};

struct by_value
{
    // 仿函数用来做比较器，用来做
    bool operator()(smoothness_t const &left, smoothness_t const &right)
    {
        return left.value < right.value;
    }
};

class FeatureExtraction : public ParamServer
{

public:
    ros::Subscriber subLaserCloudInfo;

    ros::Publisher pubLaserCloudInfo;
    ros::Publisher pubCornerPoints;
    ros::Publisher pubSurfacePoints;

    pcl::PointCloud<PointType>::Ptr extractedCloud;
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    pcl::PointCloud<PointType>::Ptr surfaceCloud;

    pcl::VoxelGrid<PointType> downSizeFilter;

    lvi_sam::cloud_info cloudInfo;
    std_msgs::Header cloudHeader;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    FeatureExtraction() // 构造函数
    {
        // 订阅去畸变、有序化的自定义格式点云cloudInfo，回调函数负责去除干扰点，根据曲率提取角点和平面点
        subLaserCloudInfo = nh.subscribe<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/deskew/cloud_info", 5, &FeatureExtraction::laserCloudInfoHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布注册了角点点云和面点点云，并且“瘦身”之后的cloudInfo
        pubLaserCloudInfo = nh.advertise<lvi_sam::cloud_info>(PROJECT_NAME + "/lidar/feature/cloud_info", 5);

        // 发布角点点云和面点点云
        pubCornerPoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_corner", 5);
        pubSurfacePoints = nh.advertise<sensor_msgs::PointCloud2>(PROJECT_NAME + "/lidar/feature/cloud_surface", 5);

        // 重置所有参数，设置体素化滤波器分辨率
        initializationValue();
    }

    void initializationValue()
    {
        cloudSmoothness.resize(N_SCAN * Horizon_SCAN);

        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        cloudCurvature = new float[N_SCAN * Horizon_SCAN];
        cloudNeighborPicked = new int[N_SCAN * Horizon_SCAN];
        cloudLabel = new int[N_SCAN * Horizon_SCAN];
    }

    void laserCloudInfoHandler(const lvi_sam::cloud_infoConstPtr &msgIn)
    {
        // 1、预处理：点云格
        cloudInfo = *msgIn;                                      // new cloud info
        cloudHeader = msgIn->header;                             // new cloud header
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction

        // 2、计算点云中各点曲率，为提取特征点
        calculateSmoothness();

        // 3、屏蔽点云中被遮挡点或平行
        markOccludedPoints();

        // 4、依据曲率，提取角特征和面特征
        extractFeatures();

        // 5、向后续节点发布提取到的角特征和面特征
        publishFeatureCloud();
    }

    void calculateSmoothness()
    {
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            // 采用距离来计算曲率，这是对之前imageProjection节点计算过的距离信息再次利用，提高信息利用率和计算效率
            float diffRange = cloudInfo.pointRange[i - 5] + cloudInfo.pointRange[i - 4] + cloudInfo.pointRange[i - 3] + cloudInfo.pointRange[i - 2] + cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i] * 10 + cloudInfo.pointRange[i + 1] + cloudInfo.pointRange[i + 2] + cloudInfo.pointRange[i + 3] + cloudInfo.pointRange[i + 4] + cloudInfo.pointRange[i + 5];

            // 曲率采用距离的平方的
            cloudCurvature[i] = diffRange * diffRange; // diffX * diffX + diffY * diffY + diffZ * diffZ;

            //  计算特征点的标记位
            // 1:表示不参与提取特征点，可能受遮挡等
            // 0:表示参与提取特征点
            // 默认为参与提取特征点
            cloudNeighborPicked[i] = 0;

            // 特征点的分类标记
            // 1：曲率比较大
            // 0:默认情况下，比较平坦点，现实世界也是以平面点为主；
            // 不满足1、-1的情况下，都是0比较平坦点
            // -1:平坦的点
            cloudLabel[i] = 0;

            // cloudSmoothness for sorting
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            float depth1 = cloudInfo.pointRange[i];
            float depth2 = cloudInfo.pointRange[i + 1];
            int columnDiff = std::abs(int(cloudInfo.pointColInd[i + 1] - cloudInfo.pointColInd[i]));

            // 相近点：在有序点云中顺序相邻，并且在距离图像上的列序号之差小于10
            if (columnDiff < 10)
            {
                // 10 pixel diff in range image
                // 两个相近的点与LiDAR的距离之差大于0.3m，则认为它们是互相遮挡的点
                // 标记后景点，防止把遮挡点当做角点
                if (depth1 - depth2 > 0.3)
                {
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }
                else if (depth2 - depth1 > 0.3)
                {
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            // 理论上物体距离越远时，点云才应该越稀疏，相邻点距离较远
            // 但是对于平行于激光光束的表面来说，即使距离激光雷达很近，依然会导致两个相邻点很远
            // 所以当相邻点距离之差大于0.02倍距离时，认为它是平行于激光光束的平面，排除掉该点
            float diff1 = std::abs(float(cloudInfo.pointRange[i - 1] - cloudInfo.pointRange[i]));
            float diff2 = std::abs(float(cloudInfo.pointRange[i + 1] - cloudInfo.pointRange[i]));

            if (diff1 > 0.02 * cloudInfo.pointRange[i] && diff2 > 0.02 * cloudInfo.pointRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }

    void extractFeatures()
    {
        // 存放角点特征的点云和面点特征的点云；
        // 计算特征前，容器先清空；
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        // 保证提取的特征整体分布较为均匀
        // 遍历每个scan
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            //每个scan6等分
            for (int j = 0; j < 6; j++)
            {

                int sp = (cloudInfo.startRingIndex[i] * (6 - j) + cloudInfo.endRingIndex[i] * j) / 6;
                int ep = (cloudInfo.startRingIndex[i] * (5 - j) + cloudInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                // 对当前提取段的曲率进行排序
                // 按照曲率从小到大：by_value();
                // cloudSmoothness: 点的index和曲率，排序按照曲率排序，尽管cloudSmoothness被打乱了，可以用index与曲率的关联找到,以及该点状态；
                std::sort(cloudSmoothness.begin() + sp, cloudSmoothness.begin() + ep, by_value());

                // 记录挑到的
                int largestPickedNum = 0;
                // 选取曲率大的点，所以从后往前遍历
                for (int k = ep; k >= sp; k--)
                {
                    // 利用cloudSmoothness中index可以关联到该点是否参与提
                    int ind = cloudSmoothness[k].ind;
                    // 参与提取特征点，曲率大于阈值，则算1个
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        largestPickedNum++;
                        // 每个scan分成6段，每段提取满足条件的曲率前20大的点
                        if (largestPickedNum <= 20)
                        {
                            cloudLabel[ind] = 1;
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        }
                        else
                        {
                            break;
                        }

                        // 避免特征点过于集中，将当前点周围5个点状态置为1，不参
                        cloudNeighborPicked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 面点特征提取，相同道理，每段scan分
                for (int k = sp; k <= ep; k++)
                {
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {

                        cloudLabel[ind] = -1;
                        //避免面点过于集中，当前点周围5各点被屏蔽掉；
                        cloudNeighborPicked[ind] = 1;

                        for (int l = 1; l <= 5; l++)
                        {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {

                            int columnDiff = std::abs(int(cloudInfo.pointColInd[ind + l] - cloudInfo.pointColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0)
                    {
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            //避免特征过大，造成计算时间过长，进行降采样；
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    // cloud_info是LVI-SAM自定义格式的点云，包含了大量的数据内容，需要占用大量的内存空间(16G不够)。

    // imageProjection节点把原始的无序点云，进行了去畸变和有序化，
    // 并且设置了大量的索引用来方便遍历查找特征点，还保留了距离信息用来筛除一些特殊点云点。

    // 这些数据对于featureExtraction节点是必要的，但是对于后续的过程没有意义，而且重新创建一个cloud_info实例是不方便的，
    // 因此在发布话题之前，lvi-sam对cloudInfo进行了一次“瘦身”，以提高后面的运行效率。
    void freeCloudInfoMemory()
    {
        cloudInfo.startRingIndex.clear();
        cloudInfo.endRingIndex.clear();
        cloudInfo.pointColInd.clear();
        cloudInfo.pointRange.clear();
    }

    void publishFeatureCloud()
    {
        // free cloud info memory
        freeCloudInfoMemory();
        // save newly extracted features
        cloudInfo.cloud_corner = publishCloud(pubCornerPoints, cornerCloud, cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lvi_sam");

    // 与imageProjection相似，核心工作都由FeatureExtraction类的构造函数完成
    FeatureExtraction FE;

    ROS_INFO("\033[1;32m----> Lidar Feature Extraction Started.\033[0m");

    ros::spin();

    return 0;
}