# Cloud Info
Header header ## 标头

int32[] startRingIndex  ## 点云的第i条扫描线（ring）上的第一个可计算曲率的点
int32[] endRingIndex  ## 点云的第i条扫描线（ring）上的最后一个可计算曲率的点

int32[]  pointColInd # point column index in range image ## 点云中每个点在投影图片中的列序号
float32[] pointRange # point range ## 点云中每个点与LiDAR的距离，即投影图片的像素值

int64 imuAvailable        #; 原始9轴IMU输出的位姿是否可用
int64 vinsOdomAvailable   #; vins发来的里程计是否可用
int64 imuOdomAvailable    #; LIO-SAM的imu预积分节点发来的里程计是否可用

# Attitude for lidar odometry initialization
## 可用的imu旋转测量，作为LIS帧间位姿变换的预测值
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Odometry 
## 可用的相邻帧位姿变换，同样作为LIS帧间位姿变换的预测值
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Odometry reset ID
## 从里程计获得的位姿变换协方差的取整（四舍五入），可以用于计算该位姿变换值的可信度
int64 vinsOdomResetId   #; vins里程计重启的id，在vins失效时会重启，id+1
int64 imuOdomResetId    #; LIO-SAM的imu预积分里程计重启的id，在imu预积分节点估计的bias过大时会重启，id+1

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature

# 3rd party messages
sensor_msgs/PointCloud2 key_frame_cloud
sensor_msgs/PointCloud2 key_frame_color
sensor_msgs/PointCloud2 key_frame_poses
sensor_msgs/PointCloud2 key_frame_map