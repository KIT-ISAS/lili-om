#include "livox_ros_driver/CustomMsg.h"
#include "utils/common.h"

#define  PI  3.1415926535

using namespace std;

ros::Publisher pub_ros_points;
string frame_id = "lili_om";

void livoxLidarHandler(const livox_ros_driver::CustomMsgConstPtr& livox_msg_in) {
    pcl::PointCloud<PointXYZINormal> pcl_in;
    auto time_end = livox_msg_in->points.back().offset_time;
    for (unsigned int i = 0; i < livox_msg_in->point_num; ++i) {
        PointXYZINormal pt;
        pt.x = livox_msg_in->points[i].x;
        pt.y = livox_msg_in->points[i].y;
        pt.z = livox_msg_in->points[i].z;
        float s = float(livox_msg_in->points[i].offset_time / (float)time_end);
        pt.intensity = livox_msg_in->points[i].line + s*0.1; // integer part: line number; decimal part: timestamp
        pt.curvature = 0.1 * livox_msg_in->points[i].reflectivity;
        pcl_in.push_back(pt);
    }

    /// timebase 5ms ~ 50000000, so 10 ~ 1ns

    ros::Time timestamp(livox_msg_in->header.stamp.toSec());

    sensor_msgs::PointCloud2 pcl_ros_msg;
    pcl::toROSMsg(pcl_in, pcl_ros_msg);
    pcl_ros_msg.header.stamp = timestamp;
    pcl_ros_msg.header.frame_id = frame_id;

    pub_ros_points.publish(pcl_ros_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "lili_om");
    ros::NodeHandle nh;

    ROS_INFO("\033[1;32m---->\033[0m Format Convert Started.");

    if (!getParameter("/common/frame_id", frame_id))
    {
        ROS_WARN("frame_id not set, use default value: lili_om");
        frame_id = "lili_om";
    }

    ros::Subscriber sub_livox_lidar = nh.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", 100, livoxLidarHandler);
    pub_ros_points = nh.advertise<sensor_msgs::PointCloud2>("/livox_ros_points", 100);

    ros::spin();
}
