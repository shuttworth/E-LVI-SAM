#include "parameters.h"
#include "keyframe.h"
#include "loop_detection.h"

#define SKIP_FIRST_CNT 10

queue<sensor_msgs::ImageConstPtr>      image_buf;
queue<sensor_msgs::PointCloudConstPtr> point_buf;
queue<nav_msgs::Odometry::ConstPtr>    pose_buf;

std::mutex m_buf;
std::mutex m_process;

LoopDetector loopDetector;

double SKIP_TIME = 0;
double SKIP_DIST = 0;

camodocal::CameraPtr m_camera;

Eigen::Vector3d tic;
Eigen::Matrix3d qic;

std::string PROJECT_NAME;
std::string IMAGE_TOPIC;

int DEBUG_IMAGE;
int LOOP_CLOSURE;
double MATCH_IMAGE_SCALE;


ros::Publisher pub_match_img;
ros::Publisher pub_match_msg;
ros::Publisher pub_key_pose;



BriefExtractor briefExtractor;

void new_sequence()
{
    m_buf.lock();
    while(!image_buf.empty())
        image_buf.pop();
    while(!point_buf.empty())
        point_buf.pop();
    while(!pose_buf.empty())
        pose_buf.pop();
    m_buf.unlock();
}

void image_callback(const sensor_msgs::ImageConstPtr &image_msg)
{
    if(!LOOP_CLOSURE)
        return;

    // 保存到图片队列
    m_buf.lock();
    image_buf.push(image_msg);
    m_buf.unlock();

    // detect unstable camera stream
    // 检查图片连续性，如果不连续，则清空所有队列
    static double last_image_time = -1;
    if (last_image_time == -1)
        last_image_time = image_msg->header.stamp.toSec();
    else if (image_msg->header.stamp.toSec() - last_image_time > 1.0 || image_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! detect a new sequence!");
        new_sequence(); // 清空
    }
    last_image_time = image_msg->header.stamp.toSec();
}

void point_callback(const sensor_msgs::PointCloudConstPtr &point_msg)
{
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    point_buf.push(point_msg);
    m_buf.unlock();
}

void pose_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    if(!LOOP_CLOSURE)
        return;

    m_buf.lock();
    pose_buf.push(pose_msg);
    m_buf.unlock();
}

void extrinsic_callback(const nav_msgs::Odometry::ConstPtr &pose_msg)
{
    m_process.lock();
    tic = Vector3d(pose_msg->pose.pose.position.x,
                   pose_msg->pose.pose.position.y,
                   pose_msg->pose.pose.position.z);
    qic = Quaterniond(pose_msg->pose.pose.orientation.w,
                      pose_msg->pose.pose.orientation.x,
                      pose_msg->pose.pose.orientation.y,
                      pose_msg->pose.pose.orientation.z).toRotationMatrix();
    m_process.unlock();
}

void process()
{
    if (!LOOP_CLOSURE)
        return;

    while (ros::ok())
    {
        sensor_msgs::ImageConstPtr image_msg = NULL;
        sensor_msgs::PointCloudConstPtr point_msg = NULL;
        nav_msgs::Odometry::ConstPtr pose_msg = NULL;

        // Step 1.1
        // find out the messages with same time stamp
        m_buf.lock();
        if(!image_buf.empty() && !point_buf.empty() && !pose_buf.empty())
        {
            // 时间戳对齐
            // 若最旧位姿时间戳旧于最旧图片时间戳，则抛弃旧的位姿数据
            if (image_buf.front()->header.stamp.toSec() > pose_buf.front()->header.stamp.toSec())
            {
                pose_buf.pop();
                printf("throw pose at beginning\n");
            }
            // 若最旧特征点时间戳旧于最旧图片时间戳，则抛弃旧的特征点数据
            else if (image_buf.front()->header.stamp.toSec() > point_buf.front()->header.stamp.toSec())
            {
                point_buf.pop();
                printf("throw point at beginning\n");
            }
            else if (image_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec() 
                && point_buf.back()->header.stamp.toSec() >= pose_buf.front()->header.stamp.toSec())
            {
                pose_msg = pose_buf.front();
                pose_buf.pop();
                while (!pose_buf.empty())
                    pose_buf.pop();
                while (image_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    image_buf.pop();
                image_msg = image_buf.front();
                image_buf.pop();

                while (point_buf.front()->header.stamp.toSec() < pose_msg->header.stamp.toSec())
                    point_buf.pop();
                point_msg = point_buf.front();
                point_buf.pop();
            }
        }
        m_buf.unlock();

        if (pose_msg != NULL)
        {
            // skip fisrt few 前几帧不易产生回环
            static int skip_first_cnt = 0;
            if (skip_first_cnt < SKIP_FIRST_CNT)
            {
                skip_first_cnt++;
                continue;
            }

            // limit frequency
            static double last_skip_time = -1;
            if (pose_msg->header.stamp.toSec() - last_skip_time < SKIP_TIME)
                continue;
            else
                last_skip_time = pose_msg->header.stamp.toSec();

            // get keyframe pose
            static Eigen::Vector3d last_t(-1e6, -1e6, -1e6);
            Vector3d T = Vector3d(pose_msg->pose.pose.position.x,
                                  pose_msg->pose.pose.position.y,
                                  pose_msg->pose.pose.position.z);
            Matrix3d R = Quaterniond(pose_msg->pose.pose.orientation.w,
                                     pose_msg->pose.pose.orientation.x,
                                     pose_msg->pose.pose.orientation.y,
                                     pose_msg->pose.pose.orientation.z).toRotationMatrix();

            // Step 1.2
            // add keyframe  如果平移距离足够大才认定为回环检测模块的关键帧
            if((T - last_t).norm() > SKIP_DIST)
            {
                // convert image
                cv_bridge::CvImageConstPtr ptr;
                if (image_msg->encoding == "8UC1")
                {
                    sensor_msgs::Image img;
                    img.header = image_msg->header;
                    img.height = image_msg->height;
                    img.width = image_msg->width;
                    img.is_bigendian = image_msg->is_bigendian;
                    img.step = image_msg->step;
                    img.data = image_msg->data;
                    img.encoding = "mono8";
                    ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
                }
                else
                    ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::MONO8);
                
                cv::Mat image = ptr->image;

                // 构建新的关键帧
                vector<cv::Point3f> point_3d; 
                vector<cv::Point2f> point_2d_uv; 
                vector<cv::Point2f> point_2d_normal;
                vector<double> point_id;

                for (unsigned int i = 0; i < point_msg->points.size(); i++)
                {
                    cv::Point3f p_3d;
                    p_3d.x = point_msg->points[i].x;
                    p_3d.y = point_msg->points[i].y;
                    p_3d.z = point_msg->points[i].z;
                    point_3d.push_back(p_3d);

                    cv::Point2f p_2d_uv, p_2d_normal;
                    double p_id;
                    p_2d_normal.x = point_msg->channels[i].values[0];
                    p_2d_normal.y = point_msg->channels[i].values[1];
                    p_2d_uv.x = point_msg->channels[i].values[2];
                    p_2d_uv.y = point_msg->channels[i].values[3];
                    p_id = point_msg->channels[i].values[4];
                    point_2d_normal.push_back(p_2d_normal);
                    point_2d_uv.push_back(p_2d_uv);
                    point_id.push_back(p_id);
                }

                // Step 1.2.1 
                // construct new keyframe
                static int global_frame_index = 0;
                KeyFrame* keyframe = new KeyFrame(pose_msg->header.stamp.toSec(), global_frame_index, 
                                                  T, R, 
                                                  image,
                                                  point_3d, point_2d_uv, point_2d_normal, point_id);   

                // Step 1.2.2 
                // detect loop  添加新的关键帧并做回环检测
                m_process.lock();
                loopDetector.addKeyFrame(keyframe, 1);
                m_process.unlock();

                loopDetector.visualizeKeyPoses(pose_msg->header.stamp.toSec());

                global_frame_index++;
                last_t = T;
            }
        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
} 


int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins");
    ros::NodeHandle n;
    ROS_INFO("\033[1;32m----> Visual Loop Detection Started.\033[0m");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Warn);

    // Load params  载入参数文件，并等待100微秒，载入参数
    std::string config_file;
    n.getParam("vins_config_file", config_file);
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    usleep(100);

    // Initialize global params
    fsSettings["project_name"] >> PROJECT_NAME;  
    fsSettings["image_topic"]  >> IMAGE_TOPIC;  
    fsSettings["loop_closure"] >> LOOP_CLOSURE;
    fsSettings["skip_time"]    >> SKIP_TIME;
    fsSettings["skip_dist"]    >> SKIP_DIST;
    fsSettings["debug_image"]  >> DEBUG_IMAGE;
    fsSettings["match_image_scale"] >> MATCH_IMAGE_SCALE;
    
    if (LOOP_CLOSURE)
    {
        // 读取字典路径并载入字典
        string pkg_path = ros::package::getPath(PROJECT_NAME);

        // initialize vocabulary
        // 加载词典：我们可以去yaml文件中看看vocabulary_file的位置和文件是什么，比如config/M2DGR_camera.yaml
        string vocabulary_file;
        fsSettings["vocabulary_file"] >> vocabulary_file;  
        vocabulary_file = pkg_path + vocabulary_file;
        loopDetector.loadVocabulary(vocabulary_file);

        // initialize brief extractor
        // 加载描述子：看yaml文件里的brief_pattern_file，比如config/M2DGR_camera.yaml
        string brief_pattern_file;
        fsSettings["brief_pattern_file"] >> brief_pattern_file;  
        brief_pattern_file = pkg_path + brief_pattern_file;
        briefExtractor = BriefExtractor(brief_pattern_file);

        // initialize camera model
        // 载入相机参数
        m_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(config_file.c_str());
    }

    // 订阅VIS窗口关键帧位姿keyframe_pose，相机IMU外参tf话题odometry/extrinsic，最新关键帧位姿话题keyframe_point，原始图片话题image_raw。
    ros::Subscriber sub_image     = n.subscribe(IMAGE_TOPIC, 30, image_callback);
    ros::Subscriber sub_pose      = n.subscribe(PROJECT_NAME + "/vins/odometry/keyframe_pose",  3, pose_callback);
    ros::Subscriber sub_point     = n.subscribe(PROJECT_NAME + "/vins/odometry/keyframe_point", 3, point_callback);
    ros::Subscriber sub_extrinsic = n.subscribe(PROJECT_NAME + "/vins/odometry/extrinsic",      3, extrinsic_callback);

    // 发布回环帧图片match_image和位姿keyframe_pose用于可视化
    // 发布视觉回环帧match_frame
    pub_match_img = n.advertise<sensor_msgs::Image>             (PROJECT_NAME + "/vins/loop/match_image", 3);
    pub_match_msg = n.advertise<std_msgs::Float64MultiArray>    (PROJECT_NAME + "/vins/loop/match_frame", 3);
    pub_key_pose  = n.advertise<visualization_msgs::MarkerArray>(PROJECT_NAME + "/vins/loop/keyframe_pose", 3);

    // 关闭回环检测
    if (!LOOP_CLOSURE)
    {
        sub_image.shutdown();
        sub_pose.shutdown();
        sub_point.shutdown();
        sub_extrinsic.shutdown();

        pub_match_img.shutdown();
        pub_match_msg.shutdown();
        pub_key_pose.shutdown();
    }

    std::thread measurement_process;
    // 回环检测线程,process()里是主要的处理所在
    measurement_process = std::thread(process);

    ros::spin();

    return 0;
}