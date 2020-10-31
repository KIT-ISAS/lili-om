#include "utils/common.h"
#include "utils/math_tools.h"
#include "utils/timer.h"
#include "factors/LidarKeyframeFactor.h"
#include "factors/LidarPoseFactor.h"
#include "factors/ImuFactor.h"
#include "factors/PriorFactor.h"
#include "factors/Preintegration.h"
#include "factors/MarginalizationFactor.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

class BackendFusion
{
private:
    int odom_sub_cnt = 0;
    int map_pub_cnt = 0;
    ros::NodeHandle nh;

    ros::Subscriber sub_edge;
    ros::Subscriber sub_surf;
    ros::Subscriber sub_odom;
    ros::Subscriber sub_each_odom;
    ros::Subscriber sub_full_cloud;
    ros::Subscriber sub_imu;

    ros::Publisher pub_map;
    ros::Publisher pub_odom;
    ros::Publisher pub_poses;
    ros::Publisher pub_edge;
    ros::Publisher pub_surf;
    ros::Publisher pub_full;
    ros::Publisher pub_local_surfs;
    ros::Publisher pub_local_edges;

    nav_msgs::Odometry odom_mapping;

    bool new_edge = false;
    bool new_surf = false;
    bool new_odom = false;
    bool new_each_odom = false;
    bool new_full_cloud = false;

    double time_new_edge;
    double time_new_surf;
    double time_new_odom;
    double time_new_each_odom = 0;

    pcl::PointCloud<PointType>::Ptr edge_last;
    pcl::PointCloud<PointType>::Ptr surf_last;
    pcl::PointCloud<PointType>::Ptr full_cloud;
    vector<pcl::PointCloud<PointType>::Ptr> full_clouds_ds;
    vector<pcl::PointCloud<PointType>::Ptr> full_clouds;

    pcl::PointCloud<PointType>::Ptr edge_last_ds;
    pcl::PointCloud<PointType>::Ptr surf_last_ds;

    vector<pcl::PointCloud<PointType>::Ptr> edge_lasts_ds;
    vector<pcl::PointCloud<PointType>::Ptr> surf_lasts_ds;

    pcl::PointCloud<PointType>::Ptr edge_local_map;
    pcl::PointCloud<PointType>::Ptr surf_local_map;
    pcl::PointCloud<PointType>::Ptr edge_local_map_ds;
    pcl::PointCloud<PointType>::Ptr surf_local_map_ds;

    vector<pcl::PointCloud<PointType>::Ptr> vec_edge_cur_pts;
    vector<pcl::PointCloud<PointType>::Ptr> vec_edge_match_j;
    vector<pcl::PointCloud<PointType>::Ptr> vec_edge_match_l;

    vector<pcl::PointCloud<PointType>::Ptr> vec_surf_cur_pts;
    vector<pcl::PointCloud<PointType>::Ptr> vec_surf_normal;
    vector<vector<double>> vec_surf_scores;

    pcl::PointCloud<PointType>::Ptr latest_key_frames;
    pcl::PointCloud<PointType>::Ptr latest_key_frames_ds;
    pcl::PointCloud<PointType>::Ptr his_key_frames;
    pcl::PointCloud<PointType>::Ptr his_key_frames_ds;

    pcl::PointCloud<PointXYZI>::Ptr pose_cloud_frame; //position of keyframe
    // Usage for PointPoseInfo
    // position: x, y, z
    // orientation: qw - w, qx - x, qy - y, qz - z
    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_cloud_frame; //pose of keyframe

    pcl::PointCloud<PointXYZI>::Ptr pose_each_frame; //position of each frame
    pcl::PointCloud<PointPoseInfo>::Ptr pose_info_each_frame; //pose of each frame

    PointXYZI select_pose;
    PointType pt_in_local, pt_in_map;

    pcl::PointCloud<PointType>::Ptr global_map;
    pcl::PointCloud<PointType>::Ptr global_map_ds;

    vector<pcl::PointCloud<PointType>::Ptr> edge_frames;
    vector<pcl::PointCloud<PointType>::Ptr> surf_frames;

    deque<pcl::PointCloud<PointType>::Ptr> recent_edge_keyframes;
    deque<pcl::PointCloud<PointType>::Ptr> recent_surf_keyframes;
    int latest_frame_idx;

    pcl::KdTreeFLANN<PointType>::Ptr kd_tree_edge_local_map;
    pcl::KdTreeFLANN<PointType>::Ptr kd_tree_surf_local_map;
    pcl::KdTreeFLANN<PointXYZI>::Ptr kd_tree_his_key_poses;

    std::vector<int> pt_search_idx;
    std::vector<float> pt_search_sq_dists;

    pcl::VoxelGrid<PointType> ds_filter_edge;
    pcl::VoxelGrid<PointType> ds_filter_surf;
    pcl::VoxelGrid<PointType> ds_filter_edge_map;
    pcl::VoxelGrid<PointType> ds_filter_surf_map;
    pcl::VoxelGrid<PointType> ds_filter_his_frames;
    pcl::VoxelGrid<PointType> ds_filter_global_map;
    pcl::VoxelGrid<PointType> ds_filter_full_cloud;

    vector<int> vec_edge_res_cnt;
    vector<int> vec_surf_res_cnt;

    // Form of the transformation
    vector<double> abs_pose;
    vector<double> last_pose;

    std::mutex mutual_exclusion;

    int max_num_iter;

    // Boolean for functions
    bool loop_closure_on;

    gtsam::NonlinearFactorGraph glocal_pose_graph;
    gtsam::Values glocal_init_estimate;
    gtsam::ISAM2 *isam;
    gtsam::Values glocal_estimated;

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise;
    gtsam::noiseModel::Diagonal::shared_ptr constraint_noise;

    // Loop closure detection related
    bool loop_to_close;
    int closest_his_idx;
    int latest_frame_idx_loop;
    bool loop_closed;

    int local_map_width;

    double lc_search_radius;
    int lc_map_width;
    float lc_icp_thres;

    int slide_window_width = 3;

    //index of keyframe
    vector<int> keyframe_idx;
    vector<int> keyframe_id_in_frame;

    vector<vector<double>> abs_poses;

    int num_kf_sliding;

    vector<sensor_msgs::ImuConstPtr> imu_buf;
    nav_msgs::Odometry::ConstPtr odom_cur;
    vector<nav_msgs::Odometry::ConstPtr> each_odom_buf;
    double time_last_imu;
    double cur_time_imu;
    bool first_imu;
    vector<Preintegration*> pre_integrations;
    Eigen::Vector3d acc_0, gyr_0, g, tmp_acc_0, tmp_gyr_0;

    Eigen::Vector3d tmp_P, tmp_V;
    Eigen::Quaterniond tmp_Q;

    vector<Eigen::Vector3d> Ps;
    vector<Eigen::Vector3d> Vs;
    vector<Eigen::Matrix3d> Rs;
    vector<Eigen::Vector3d> Bas;
    vector<Eigen::Vector3d> Bgs;
    vector<vector<double>> para_speed_bias;

    //extrinsic imu to lidar
    Eigen::Quaterniond q_lb;
    Eigen::Vector3d t_lb;

    Eigen::Quaterniond q_bl;
    Eigen::Vector3d t_bl;

    double qlb0, qlb1, qlb2, qlb3, tlb0, tlb1, tlb2;

    int idx_imu;

    //first sliding window optimazition
    bool first_opt;

    // for marginalization
    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;

    double **tmpQuat;
    double **tmpTrans;
    double **tmpSpeedBias;

    bool marg = true;

    vector<int> imu_idx_in_kf;

    bool quat_ini = false;

    string imu_topic;

    double surf_dist_thres;
    double kd_max_radius;
    bool save_pcd = false;

    double lidar_const = 0;
    double reflect_thres = 0;
    int mapping_interval = 1;
    double global_lc_time_thres = 30.0;
    double local_lc_time_thres = 20.0;

    string frame_id = "lili_om";
    double surf_ds = 0.6;
    double edge_ds = 0.3;
    double mapping_ds = 0.02;

    double runtime = 0;

public:
    BackendFusion():
        nh("~")
    {
        initializeParameters();
        allocateMemory();

        sub_full_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/full_point_cloud", 100, &BackendFusion::full_cloudHandler, this);
        sub_edge = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100, &BackendFusion::edge_lastHandler, this);
        sub_surf = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100, &BackendFusion::surfaceLastHandler, this);
        sub_odom = nh.subscribe<nav_msgs::Odometry>("/odom", 5, &BackendFusion::odomHandler, this);
        sub_each_odom = nh.subscribe<nav_msgs::Odometry>("/each_odom", 5, &BackendFusion::eachOdomHandler, this);

        sub_imu = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200, &BackendFusion::imuHandler, this);

        pub_map = nh.advertise<sensor_msgs::PointCloud2>("/global_map", 2);
        pub_odom = nh.advertise<nav_msgs::Odometry>("/odom_mapped", 2);
        pub_poses = nh.advertise<sensor_msgs::PointCloud2>("/trajectory", 2);
        pub_edge = nh.advertise<sensor_msgs::PointCloud2>("/extracted_edges", 2);
        pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/extracted_planes", 2);
        pub_full = nh.advertise<sensor_msgs::PointCloud2>("/raw_scan", 2);
        pub_local_surfs = nh.advertise<sensor_msgs::PointCloud2>("/lm_planes", 2);
        pub_local_edges = nh.advertise<sensor_msgs::PointCloud2>("/lm_edges", 2);
    }

    ~BackendFusion() {}

    void allocateMemory()
    {
        tmpQuat = new double *[slide_window_width];
        tmpTrans = new double *[slide_window_width];
        tmpSpeedBias = new double *[slide_window_width];
        for (int i = 0; i < slide_window_width; ++i) {
            tmpQuat[i] = new double[4];
            tmpTrans[i] = new double[3];
            tmpSpeedBias[i] = new double[9];
        }

        edge_last.reset(new pcl::PointCloud<PointType>());
        surf_last.reset(new pcl::PointCloud<PointType>());
        edge_local_map.reset(new pcl::PointCloud<PointType>());
        surf_local_map.reset(new pcl::PointCloud<PointType>());
        edge_last_ds.reset(new pcl::PointCloud<PointType>());
        surf_last_ds.reset(new pcl::PointCloud<PointType>());
        edge_local_map_ds.reset(new pcl::PointCloud<PointType>());
        surf_local_map_ds.reset(new pcl::PointCloud<PointType>());
        full_cloud.reset(new pcl::PointCloud<PointType>());

        for(int i = 0; i < slide_window_width; i++) {
            pcl::PointCloud<PointType>::Ptr tmpSurfCurrent;
            tmpSurfCurrent.reset(new pcl::PointCloud<PointType>());
            vec_surf_cur_pts.push_back(tmpSurfCurrent);

            vector<double> tmpD;
            vec_surf_scores.push_back(tmpD);

            pcl::PointCloud<PointType>::Ptr tmpSurfNorm;
            tmpSurfNorm.reset(new pcl::PointCloud<PointType>());
            vec_surf_normal.push_back(tmpSurfNorm);

            vec_surf_res_cnt.push_back(0);

            pcl::PointCloud<PointType>::Ptr tmpCornerCurrent;
            tmpCornerCurrent.reset(new pcl::PointCloud<PointType>());
            vec_edge_cur_pts.push_back(tmpCornerCurrent);

            pcl::PointCloud<PointType>::Ptr tmpCornerL;
            tmpCornerL.reset(new pcl::PointCloud<PointType>());
            vec_edge_match_l.push_back(tmpCornerL);

            pcl::PointCloud<PointType>::Ptr tmpCornerJ;
            tmpCornerJ.reset(new pcl::PointCloud<PointType>());
            vec_edge_match_j.push_back(tmpCornerJ);

            vec_edge_res_cnt.push_back(0);
        }

        pose_cloud_frame.reset(new pcl::PointCloud<PointXYZI>());
        pose_info_cloud_frame.reset(new pcl::PointCloud<PointPoseInfo>());

        pose_each_frame.reset(new pcl::PointCloud<PointXYZI>());
        pose_info_each_frame.reset(new pcl::PointCloud<PointPoseInfo>());

        global_map.reset(new pcl::PointCloud<PointType>());
        global_map_ds.reset(new pcl::PointCloud<PointType>());

        latest_key_frames.reset(new pcl::PointCloud<PointType>());
        latest_key_frames_ds.reset(new pcl::PointCloud<PointType>());
        his_key_frames.reset(new pcl::PointCloud<PointType>());
        his_key_frames_ds.reset(new pcl::PointCloud<PointType>());

        kd_tree_edge_local_map.reset(new pcl::KdTreeFLANN<PointType>());
        kd_tree_surf_local_map.reset(new pcl::KdTreeFLANN<PointType>());
        kd_tree_his_key_poses.reset(new pcl::KdTreeFLANN<PointXYZI>());
    }

    void initializeParameters()
    {
        gtsam::ISAM2Params isamPara;
        isamPara.relinearizeThreshold = 0.1;
        isamPara.relinearizeSkip = 1;
        isam = new gtsam::ISAM2(isamPara);

        // Load parameters from yaml
        if (!getParameter("/backend_fusion/edge_ds", edge_ds))
        {
            ROS_WARN("edge_ds not set, use default value: 0.3");
            edge_ds = 0.3;
        }

        if (!getParameter("/backend_fusion/surf_ds", surf_ds))
        {
            ROS_WARN("surf_ds not set, use default value: 0.6");
            surf_ds = 0.6;
        }

        if (!getParameter("/backend_fusion/mapping_ds", mapping_ds))
        {
            ROS_WARN("mapping_ds not set, use default value: 0.02");
            mapping_ds = 0.02;
        }

        if (!getParameter("/backend_fusion/surf_dist_thres", surf_dist_thres))
        {
            ROS_WARN("surf_dist_thres not set, use default value: 0.1");
            surf_dist_thres = 0.1;
        }

        if (!getParameter("/backend_fusion/kd_max_radius", kd_max_radius))
        {
            ROS_WARN("kd_max_radius not set, use default value: 1.0");
            kd_max_radius = 1.0;
        }

        if (!getParameter("/backend_fusion/save_pcd", save_pcd))
        {
            ROS_WARN("save_pcd not set, use default value: false");
            save_pcd = false;
        }

        if (!getParameter("/backend_fusion/mapping_interval", mapping_interval))
        {
            ROS_WARN("mapping_interval not set, use default value: 1");
            mapping_interval = 1;
        }

        if (!getParameter("/backend_fusion/global_lc_time_thres", global_lc_time_thres))
        {
            ROS_WARN("global_lc_time_thres not set, use default value: 30.0");
            global_lc_time_thres = 30.0;
        }

        if (!getParameter("/backend_fusion/local_lc_time_thres", local_lc_time_thres))
        {
            ROS_WARN("local_lc_time_thres not set, use default value: 20.0");
            local_lc_time_thres = 20.0;
        }

        if (!getParameter("/backend_fusion/lidar_const", lidar_const))
        {
            ROS_WARN("lidar_const not set, use default value: 1.0");
            lidar_const = 1.0;
        }

        if (!getParameter("/backend_fusion/reflect_thres", reflect_thres))
        {
            ROS_WARN("reflect_thres not set, use default value: 15.0");
            reflect_thres = 15.0;
        }

        if (!getParameter("/backend_fusion/imu_topic", imu_topic))
        {
            ROS_WARN("imu_topic not set, use default value: /imu/data");
            imu_topic = "/imu/data";
        }

        if (!getParameter("/backend_fusion/max_num_iter", max_num_iter))
        {
            ROS_WARN("maximal iteration number of mapping optimization not set, use default value: 50");
            max_num_iter = 50;
        }

        if (!getParameter("/backend_fusion/loop_closure_on", loop_closure_on))
        {
            ROS_WARN("loop closure detection set to false");
            loop_closure_on = false;
        }

        if (!getParameter("/backend_fusion/local_map_width", local_map_width))
        {
            ROS_WARN("local_map_width not set, use default value: 5");
            local_map_width = 5;
        }

        if (!getParameter("/backend_fusion/lc_search_radius", lc_search_radius))
        {
            ROS_WARN("lc_search_radius not set, use default value: 7.0");
            lc_search_radius = 7.0;
        }

        if (!getParameter("/backend_fusion/lc_map_width", lc_map_width))
        {
            ROS_WARN("lc_map_width not set, use default value: 25");
            lc_map_width = 25;
        }

        if (!getParameter("/backend_fusion/lc_icp_thres", lc_icp_thres))
        {
            ROS_WARN("lc_icp_thres not set, use default value: 0.3");
            lc_icp_thres = 0.3;
        }

        if (!getParameter("/backend_fusion/slide_window_width", slide_window_width))
        {
            ROS_WARN("slide_window_width not set, use default value: 4");
            slide_window_width = 4;
        }

        //extrinsic parameters
        if (!getParameter("/backend_fusion/ql2b_w", qlb0))
        {
            ROS_WARN("qlb0 not set, use default value: 1");
            qlb0 = 1;
        }

        if (!getParameter("/backend_fusion/ql2b_x", qlb1))
        {
            ROS_WARN("qlb1 not set, use default value: 0");
            qlb1 = 0;
        }

        if (!getParameter("/backend_fusion/ql2b_y", qlb2))
        {
            ROS_WARN("qlb2 not set, use default value: 0");
            qlb2 = 0;
        }

        if (!getParameter("/backend_fusion/ql2b_z", qlb3))
        {
            ROS_WARN("qlb3 not set, use default value: 0");
            qlb3 = 0;
        }

        if (!getParameter("/backend_fusion/tl2b_x", tlb0))
        {
            ROS_WARN("tlb0 not set, use default value: 0");
            tlb0 = 0;
        }

        if (!getParameter("/backend_fusion/tl2b_y", tlb1))
        {
            ROS_WARN("tlb1 not set, use default value: 0");
            tlb1 = 0;
        }

        if (!getParameter("/backend_fusion/tl2b_z", tlb2))
        {
            ROS_WARN("tlb2 not set, use default value: 0");
            tlb2 = 0;
        }

        last_marginalization_info = nullptr;
        tmp_P = tmp_V = Eigen::Vector3d(0, 0, 0);
        tmp_Q = Eigen::Quaterniond::Identity();
        idx_imu = 0;
        first_opt = false;
        cur_time_imu = -1;

        Rs.push_back(Eigen::Matrix3d::Identity());
        Ps.push_back(Eigen::Vector3d::Zero());
        Vs.push_back(Eigen::Vector3d(0, 0, 0));

        Bas.push_back(Eigen::Vector3d::Zero());
        Bgs.push_back(Eigen::Vector3d(0, 0, 0));
        vector<double> tmpSpeedBias;
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);
        tmpSpeedBias.push_back(0.0);

        para_speed_bias.push_back(tmpSpeedBias);

        num_kf_sliding = 0;
        time_last_imu = 0;
        first_imu = false;

        g = Eigen::Vector3d(0, 0, 9.805);

        time_new_edge = 0;
        time_new_surf = 0;
        time_new_odom = 0;

        abs_pose.push_back(1);
        last_pose.push_back(1);

        latest_frame_idx = 0;

        vector<double> tmpOdom;
        tmpOdom.push_back(1);

        for (int i = 1; i < 7; ++i)
        {
            abs_pose.push_back(0);
            last_pose.push_back(0);
            tmpOdom.push_back(0);
        }
        abs_poses.push_back(tmpOdom);

        abs_pose = tmpOdom;

        odom_mapping.header.frame_id = frame_id;

        gtsam::Vector vector6p(6);
        gtsam::Vector vector6o(6);
        vector6p << 1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8;
        vector6o << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
        prior_noise = gtsam::noiseModel::Diagonal::Variances(vector6p);
        odom_noise = gtsam::noiseModel::Diagonal::Variances(vector6o);

        loop_to_close = false;
        loop_closed = false;


        ds_filter_edge.setLeafSize(edge_ds, edge_ds, edge_ds);
        ds_filter_surf.setLeafSize(surf_ds, surf_ds, surf_ds);
        ds_filter_edge_map.setLeafSize(edge_ds, edge_ds, edge_ds);
        ds_filter_surf_map.setLeafSize(surf_ds, surf_ds, surf_ds);
        ds_filter_his_frames.setLeafSize(surf_ds, surf_ds, surf_ds);
        ds_filter_global_map.setLeafSize(mapping_ds, mapping_ds, mapping_ds);
        ds_filter_full_cloud.setLeafSize(mapping_ds, mapping_ds, mapping_ds);

        q_lb = Eigen::Quaterniond(qlb0, qlb1, qlb2, qlb3);
        t_lb = Eigen::Vector3d(tlb0, tlb1, tlb2);

        q_bl = q_lb.inverse();
        t_bl = - (q_bl * t_lb);
    }

    void full_cloudHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn)
    {
        full_cloud->clear();
        pcl::fromROSMsg(*pointCloudIn, *full_cloud);

        new_full_cloud = true;
    }

    void edge_lastHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn)
    {
        edge_last->clear();
        time_new_edge = pointCloudIn->header.stamp.toSec();
        pcl::fromROSMsg(*pointCloudIn, *edge_last);

        new_edge = true;
    }

    void surfaceLastHandler(const sensor_msgs::PointCloud2ConstPtr& pointCloudIn)
    {
        surf_last->clear();
        time_new_surf = pointCloudIn->header.stamp.toSec();
        pcl::fromROSMsg(*pointCloudIn, *surf_last);
        new_surf = true;
    }

    void odomHandler(const nav_msgs::Odometry::ConstPtr& odomIn)
    {
        //cout<<"odom_sub_cnt: "<<++odom_sub_cnt<<endl;

        time_new_odom = odomIn->header.stamp.toSec();
        odom_cur = odomIn;

        new_odom = true;
    }

    void eachOdomHandler(const nav_msgs::Odometry::ConstPtr& odomIn)
    {
        time_new_each_odom = odomIn->header.stamp.toSec();
        each_odom_buf.push_back(odomIn);

        if(each_odom_buf.size() > 50)
            each_odom_buf[each_odom_buf.size() - 51] = nullptr;

        new_each_odom = true;
    }

    void imuHandler(const sensor_msgs::ImuConstPtr& ImuIn)
    {
        time_last_imu = ImuIn->header.stamp.toSec();

        imu_buf.push_back(ImuIn);

        if(imu_buf.size() > 600)
            imu_buf[imu_buf.size() - 601] = nullptr;

        if (cur_time_imu < 0)
            cur_time_imu = time_last_imu;

        if (!first_imu)
        {
            Eigen::Quaterniond quat(ImuIn->orientation.w,
                                    ImuIn->orientation.x,
                                    ImuIn->orientation.y,
                                    ImuIn->orientation.z);

            Rs[0] = quat.toRotationMatrix();
            abs_poses[0][0] = ImuIn->orientation.w;
            abs_poses[0][1] = ImuIn->orientation.x;
            abs_poses[0][2] = ImuIn->orientation.y;
            abs_poses[0][3] = ImuIn->orientation.z;


            first_imu = true;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            dx = ImuIn->linear_acceleration.x;
            dy = ImuIn->linear_acceleration.y;
            dz = ImuIn->linear_acceleration.z;
            rx = ImuIn->angular_velocity.x;
            ry = ImuIn->angular_velocity.y;
            rz = ImuIn->angular_velocity.z;

            Eigen::Vector3d linear_acceleration(dx, dy, dz);
            Eigen::Vector3d angular_velocity(rx, ry, rz);
            acc_0 = linear_acceleration;
            gyr_0 = angular_velocity;
            pre_integrations.push_back(new Preintegration(acc_0, gyr_0, Bas[0], Bgs[0]));
            pre_integrations.back()->g_vec_ = -g;
        }
    }


    void transformPoint(PointType const *const pi, PointType *const po)
    {
        Eigen::Quaterniond quaternion(abs_pose[0],
                abs_pose[1],
                abs_pose[2],
                abs_pose[3]);
        Eigen::Vector3d transition(abs_pose[4],
                abs_pose[5],
                abs_pose[6]);

        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;

        Eigen::Vector3d normIn(pi->normal_x, pi->normal_y, pi->normal_z);
        Eigen::Vector3d normOut = quaternion * normIn;

        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
        po->curvature = pi->curvature;
        po->normal_x = normOut.x();
        po->normal_y = normOut.y();
        po->normal_z = normOut.z();
    }

    void transformPoint(PointType const *const pi, PointType *const po, Eigen::Quaterniond quaternion, Eigen::Vector3d transition)
    {
        Eigen::Vector3d ptIn(pi->x, pi->y, pi->z);
        Eigen::Vector3d ptOut = quaternion * ptIn + transition;

        Eigen::Vector3d normIn(pi->normal_x, pi->normal_y, pi->normal_z);
        Eigen::Vector3d normOut = quaternion * normIn;

        po->x = ptOut.x();
        po->y = ptOut.y();
        po->z = ptOut.z();
        po->intensity = pi->intensity;
        po->curvature = pi->curvature;
        po->normal_x = normOut.x();
        po->normal_y = normOut.y();
        po->normal_z = normOut.z();
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i)
        {
            PointType ptIn = cloudIn->points[i];
            PointType ptOut;
            transformPoint(&ptIn, &ptOut);
            cloudOut->points[i] = ptOut;
        }
        return cloudOut;
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, PointPoseInfo * PointInfoIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        Eigen::Quaterniond quaternion(PointInfoIn->qw,
                                      PointInfoIn->qx,
                                      PointInfoIn->qy,
                                      PointInfoIn->qz);
        Eigen::Vector3d transition(PointInfoIn->x,
                                   PointInfoIn->y,
                                   PointInfoIn->z);

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i)
        {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;

            Eigen::Vector3d normIn(cloudIn->points[i].normal_x, cloudIn->points[i].normal_y, cloudIn->points[i].normal_z);
            Eigen::Vector3d normOut = quaternion * normIn;

            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;
            pt.curvature = cloudIn->points[i].curvature;
            pt.normal_x = normOut.x();
            pt.normal_y = normOut.y();
            pt.normal_z = normOut.z();

            cloudOut->points[i] = pt;
        }

        return cloudOut;
    }


    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr &cloudIn, Eigen::Quaterniond quaternion, Eigen::Vector3d transition)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int numPts = cloudIn->points.size();
        cloudOut->resize(numPts);

        for (int i = 0; i < numPts; ++i)
        {
            Eigen::Vector3d ptIn(cloudIn->points[i].x, cloudIn->points[i].y, cloudIn->points[i].z);
            Eigen::Vector3d ptOut = quaternion * ptIn + transition;

            Eigen::Vector3d normIn(cloudIn->points[i].normal_x, cloudIn->points[i].normal_y, cloudIn->points[i].normal_z);
            Eigen::Vector3d normOut = quaternion * normIn;

            PointType pt;
            pt.x = ptOut.x();
            pt.y = ptOut.y();
            pt.z = ptOut.z();
            pt.intensity = cloudIn->points[i].intensity;
            pt.curvature = cloudIn->points[i].curvature;
            pt.normal_x = normOut.x();
            pt.normal_y = normOut.y();
            pt.normal_z = normOut.z();

            cloudOut->points[i] = pt;
        }

        return cloudOut;
    }

    void processIMU(double dt, const Eigen::Vector3d &linear_acceleration, const Eigen::Vector3d &angular_velocity)
    {
        //      if(pre_integrations.size() == 0)
        //          pre_integrations.push_back(new IntegrationBase(acc_0, gyr_0, Bas[0], Bgs[0]));
        if(pre_integrations.size() < abs_poses.size()) {
            pre_integrations.push_back(new Preintegration(acc_0, gyr_0, Bas.back(), Bgs.back()));
            pre_integrations.back()->g_vec_ = -g;
            Bas.push_back(Bas.back());
            Bgs.push_back(Bgs.back());
            Rs.push_back(Rs.back());
            Ps.push_back(Ps.back());
            Vs.push_back(Vs.back());
        }

        Eigen::Vector3d un_acc_0 = Rs.back() * (acc_0 - Bas.back()) - g;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs.back();
        Rs.back() *= deltaQ(un_gyr * dt).toRotationMatrix();
        Eigen::Vector3d un_acc_1 = Rs.back() * (linear_acceleration - Bas.back()) - g;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps.back() += dt * Vs.back() + 0.5 * dt * dt * un_acc;
        Vs.back() += dt * un_acc;

        pre_integrations.back()->push_back(dt, linear_acceleration, angular_velocity);

        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }


    void optimizeSlidingWindowWithLandMark()
    {
        if(slide_window_width < 1) return;
        if(keyframe_idx.size() < slide_window_width) return;

        first_opt = true;

        int windowSize = keyframe_idx[keyframe_idx.size()-1] - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;

        kd_tree_surf_local_map->setInputCloud(surf_local_map_ds);
        kd_tree_edge_local_map->setInputCloud(edge_local_map_ds);


        for (int iterCount = 0; iterCount < 1; ++iterCount)
        {
            ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
            ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
            ceres::Problem problem;

            //eigen to double
            for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); i++){

                Eigen::Quaterniond tmpQ(Rs[i]);
                tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0] = tmpQ.w();
                tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1] = tmpQ.x();
                tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2] = tmpQ.y();
                tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3] = tmpQ.z();
                tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0] = Ps[i][0];
                tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1] = Ps[i][1];
                tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2] = Ps[i][2];

                abs_poses[i][0] = tmpQ.w();
                abs_poses[i][1] = tmpQ.x();
                abs_poses[i][2] = tmpQ.y();
                abs_poses[i][3] = tmpQ.z();
                abs_poses[i][4] = Ps[i][0];
                abs_poses[i][5] = Ps[i][1];
                abs_poses[i][6] = Ps[i][2];

                for(int j = 0; j < 9; j++) {
                    tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][j] = para_speed_bias[i][j];
                }

                //add lidar parameters
                problem.AddParameterBlock(tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 3);
                problem.AddParameterBlock(tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 4, quatParameterization);

                //add IMU parameters
                problem.AddParameterBlock(tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]], 9);
            }

            abs_pose = abs_poses.back();

            if(true) {
                if (last_marginalization_info) {
                    // construct new marginlization_factor
                    MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                    problem.AddResidualBlock(marginalization_factor, NULL,
                                             last_marginalization_parameter_blocks);
                }
            }

            if(!marg) {
                //add prior factor
                for(int i = 0; i < slide_window_width - 1; i++) {

                    vector<double> tmps;
                    for(int j = 0; j < 9; j++) {
                        tmps.push_back(tmpSpeedBias[i][j]);
                    }
                    ceres::CostFunction *speedBiasPriorFactor = SpeedBiasPriorFactorAutoDiff::Create(tmps);
                    problem.AddResidualBlock(speedBiasPriorFactor, NULL, tmpSpeedBias[i]);
                }

            }

            for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx < keyframe_idx.back(); ++idx) {
                //add imu factor
                ImuFactor *imuFactor = new ImuFactor(pre_integrations[idx+1]);
                problem.AddResidualBlock(imuFactor, NULL, tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpSpeedBias[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpTrans[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpQuat[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                        tmpSpeedBias[idx+1-keyframe_idx[keyframe_idx.size()-slide_window_width]]);


            }

            for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx <= keyframe_idx.back(); idx++)
            {
                Eigen::Quaterniond Q2 = Eigen::Quaterniond(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]);
                Eigen::Vector3d T2 = Eigen::Vector3d(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]][2]);

                Q2 = Q2 * q_lb.inverse();
                T2 = T2 - Q2 * t_lb;

                int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width];
                if (surf_local_map_ds->points.size() > 50 && edge_local_map_ds->points.size() > 0)
                {
                    findCorrespondingSurfFeatures(idx-1, Q2, T2);
                    findCorrespondingCornerFeatures(idx-1, Q2, T2);

                    for (int i = 0; i < vec_edge_res_cnt[idVec]; ++i)
                    {
                        Eigen::Vector3d currentPt(vec_edge_cur_pts[idVec]->points[i].x,
                                                  vec_edge_cur_pts[idVec]->points[i].y,
                                                  vec_edge_cur_pts[idVec]->points[i].z);
                        Eigen::Vector3d lastPtJ(vec_edge_match_j[idVec]->points[i].x,
                                                vec_edge_match_j[idVec]->points[i].y,
                                                vec_edge_match_j[idVec]->points[i].z);
                        Eigen::Vector3d lastPtL(vec_edge_match_l[idVec]->points[i].x,
                                                vec_edge_match_l[idVec]->points[i].y,
                                                vec_edge_match_l[idVec]->points[i].z);

                        ceres::CostFunction *costFunction = LidarEdgeFactor::Create(currentPt, lastPtJ, lastPtL, q_lb, t_lb, vec_edge_cur_pts[idVec]->points[i].intensity);



                        problem.AddResidualBlock(costFunction, lossFunction, tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                                tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
                    }

                    for (int i = 0; i < vec_surf_res_cnt[idVec]; ++i)
                    {
                        Eigen::Vector3d currentPt(vec_surf_cur_pts[idVec]->points[i].x,
                                                  vec_surf_cur_pts[idVec]->points[i].y,
                                                  vec_surf_cur_pts[idVec]->points[i].z);
                        Eigen::Vector3d norm(vec_surf_normal[idVec]->points[i].x,
                                             vec_surf_normal[idVec]->points[i].y,
                                             vec_surf_normal[idVec]->points[i].z);
                        double normInverse = vec_surf_normal[idVec]->points[i].intensity;

                        ceres::CostFunction *costFunction = LidarPlaneNormFactor::Create(currentPt, norm, q_lb, t_lb, normInverse, vec_surf_scores[idVec][i]);


                        problem.AddResidualBlock(costFunction, lossFunction, tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]],
                                tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);

                    }
                }
                else
                {
                    ROS_WARN("Not enough feature points from the map");
                }
            }



            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = max_num_iter;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 0.5;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            for(int i = 0; i < windowSize; i++) {
                if(tmpQuat[i][0] < 0) {
                    Eigen::Quaterniond tmp(tmpQuat[i][0],
                            tmpQuat[i][1],
                            tmpQuat[i][2],
                            tmpQuat[i][3]);
                    tmp = unifyQuaternion(tmp);
                    tmpQuat[i][0] = tmp.w();
                    tmpQuat[i][1] = tmp.x();
                    tmpQuat[i][2] = tmp.y();
                    tmpQuat[i][3] = tmp.z();
                }
            }
        }

        MarginalizationInfo *marginalization_info = new MarginalizationInfo();

        if (last_marginalization_info) {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) {
                if (last_marginalization_parameter_blocks[i] == tmpTrans[0] ||
                        last_marginalization_parameter_blocks[i] == tmpQuat[0] ||
                        last_marginalization_parameter_blocks[i] == tmpSpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor

            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);

            marginalization_info->AddResidualBlockInfo(residual_block_info);
        }


        if(!marg) {
            //add prior factor
            for(int i = 0; i < slide_window_width - 1; i++) {

                vector<double*> tmp;
                tmp.push_back(tmpTrans[i]);
                tmp.push_back(tmpQuat[i]);

                vector<int> drop_set;
                if(i == 0) {
                    drop_set.push_back(0);
                    drop_set.push_back(1);
                }

                vector<double> tmps;
                for(int j = 0; j < 9; j++) {
                    tmps.push_back(tmpSpeedBias[i][j]);
                }

                vector<double*> tmp1;
                tmp1.push_back(tmpSpeedBias[i]);

                vector<int> drop_set1;
                if(i == 0) {
                    drop_set1.push_back(0);
                }
                ceres::CostFunction *speedBiasPriorFactor = SpeedBiasPriorFactorAutoDiff::Create(tmps);
                ResidualBlockInfo *residual_block_info1 = new ResidualBlockInfo(speedBiasPriorFactor, NULL,
                                                                                tmp1,
                                                                                drop_set1);

                marginalization_info->AddResidualBlockInfo(residual_block_info1);
            }

            marg = true;
        }


        //imu
        ImuFactor *imuFactor = new ImuFactor(pre_integrations[keyframe_idx[keyframe_idx.size()-slide_window_width]+1]);

        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imuFactor, NULL,
                                                                       vector<double *>{
                                                                           tmpTrans[0],
                                                                           tmpQuat[0],
                                                                           tmpSpeedBias[0],
                                                                           tmpTrans[1],
                                                                           tmpQuat[1],
                                                                           tmpSpeedBias[1]
                                                                       },
                                                                       vector<int>{0, 1, 2});

        marginalization_info->AddResidualBlockInfo(residual_block_info);


        //lidar
        for (int idx = keyframe_idx[keyframe_idx.size()-slide_window_width]; idx <= keyframe_idx.back(); idx++)
        {
            ceres::LossFunction *lossFunction = new ceres::CauchyLoss(1.0);
            int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width];
            if (surf_local_map_ds->points.size() > 50 && edge_local_map_ds->points.size() > 0)
            {
                vector<double*> tmp;
                tmp.push_back(tmpTrans[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);
                tmp.push_back(tmpQuat[idx-keyframe_idx[keyframe_idx.size()-slide_window_width]]);

                for (int i = 0; i < vec_surf_res_cnt[idVec]; ++i)
                {
                    Eigen::Vector3d currentPt(vec_surf_cur_pts[idVec]->points[i].x,
                                              vec_surf_cur_pts[idVec]->points[i].y,
                                              vec_surf_cur_pts[idVec]->points[i].z);
                    Eigen::Vector3d norm(vec_surf_normal[idVec]->points[i].x,
                                         vec_surf_normal[idVec]->points[i].y,
                                         vec_surf_normal[idVec]->points[i].z);
                    double normInverse = vec_surf_normal[idVec]->points[i].intensity;

                    ceres::CostFunction *costFunction = LidarPlaneNormFactor::Create(currentPt, norm, q_lb, t_lb, normInverse, vec_surf_scores[idVec][i]);

                    vector<int> drop_set;
                    if(idx == keyframe_idx[keyframe_idx.size()-slide_window_width]) {
                        drop_set.push_back(0);
                        drop_set.push_back(1);
                    }
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(costFunction, lossFunction,
                                                                                   tmp,
                                                                                   drop_set);
                    marginalization_info->AddResidualBlockInfo(residual_block_info);
                }


                for (int i = 0; i < vec_edge_res_cnt[idVec]; ++i)
                {
                    Eigen::Vector3d currentPt(vec_edge_cur_pts[idVec]->points[i].x,
                                              vec_edge_cur_pts[idVec]->points[i].y,
                                              vec_edge_cur_pts[idVec]->points[i].z);
                    Eigen::Vector3d lastPtJ(vec_edge_match_j[idVec]->points[i].x,
                                            vec_edge_match_j[idVec]->points[i].y,
                                            vec_edge_match_j[idVec]->points[i].z);
                    Eigen::Vector3d lastPtL(vec_edge_match_l[idVec]->points[i].x,
                                            vec_edge_match_l[idVec]->points[i].y,
                                            vec_edge_match_l[idVec]->points[i].z);


                    ceres::CostFunction *costFunction = LidarEdgeFactor::Create(currentPt, lastPtJ, lastPtL, q_lb, t_lb, vec_edge_cur_pts[idVec]->points[i].intensity);



                    vector<int> drop_set;
                    if(idx == keyframe_idx[keyframe_idx.size()-slide_window_width]) {
                        drop_set.push_back(0);
                        drop_set.push_back(1);
                    }
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(costFunction, lossFunction,
                                                                                   tmp,
                                                                                   drop_set);
                    marginalization_info->AddResidualBlockInfo(residual_block_info);
                }

            }
            else
            {
                ROS_WARN("Not enough feature points from the map");
            }


            vec_edge_cur_pts[idVec]->clear();
            vec_edge_match_j[idVec]->clear();
            vec_edge_match_l[idVec]->clear();
            vec_edge_res_cnt[idVec] = 0;

            vec_surf_cur_pts[idVec]->clear();
            vec_surf_normal[idVec]->clear();
            vec_surf_res_cnt[idVec] = 0;
            vec_surf_scores[idVec].clear();
        }

        marginalization_info->PreMarginalize();
        marginalization_info->Marginalize();

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i < windowSize; ++i) {
            addr_shift[reinterpret_cast<long>(tmpTrans[i])] = tmpTrans[i-1];
            addr_shift[reinterpret_cast<long>(tmpQuat[i])] = tmpQuat[i-1];
            addr_shift[reinterpret_cast<long>(tmpSpeedBias[i])] = tmpSpeedBias[i-1];
        }

        vector<double *> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);


        if (last_marginalization_info) {
            delete last_marginalization_info;
        }
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;


        //double to eigen
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx.back(); ++i){

            double dp0 = Ps[i][0] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            double dp1 = Ps[i][1] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            double dp2 = Ps[i][2] - tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            double pnorm = sqrt(dp0*dp0+dp1*dp1+dp2*dp2);

            double dv0 = Vs[i][0] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
            double dv1 = Vs[i][1] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
            double dv2 = Vs[i][2] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            double vnorm = sqrt(dv0*dv0+dv1*dv1+dv2*dv2);

            double dba1 = para_speed_bias[i][3] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];
            double dba2 = para_speed_bias[i][4] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4];
            double dba3 = para_speed_bias[i][5] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5];
            double dbg1 = para_speed_bias[i][6] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6];
            double dbg2 = para_speed_bias[i][7] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7];
            double dbg3 = para_speed_bias[i][8] - tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8];

            Eigen::Quaterniond dq = Eigen::Quaterniond (tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                    tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]).normalized().inverse() *
                    Eigen::Quaterniond(Rs[i]);
            double qnorm = dq.vec().norm();


            if(pnorm < 10) {
                abs_poses[i][4] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                abs_poses[i][5] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                abs_poses[i][6] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];

                Ps[i][0] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                Ps[i][1] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                Ps[i][2] = tmpTrans[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            } else
                ROS_WARN("bad optimization result of p!!!!!!!!!!!!!");

            if(qnorm < 10) {
                abs_poses[i][0] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                abs_poses[i][1] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                abs_poses[i][2] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
                abs_poses[i][3] = tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];

                Rs[i] = Eigen::Quaterniond (tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2],
                        tmpQuat[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3]).normalized().toRotationMatrix();
            } else
                ROS_WARN("bad optimization result of q!!!!!!!!!!!!!");

            if(vnorm < 10) {
                for(int j = 0; j < 3; j++) {
                    para_speed_bias[i][j] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][j];
                }
                Vs[i][0] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][0];
                Vs[i][1] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][1];
                Vs[i][2] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][2];
            } else
                ROS_WARN("bad optimization result of v!!!!!!!!!!!!!");

            if(abs(dba1) < 22) {
                para_speed_bias[i][3] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][3];
                Bas[i][0] = para_speed_bias[i][3];
            } else
                ROS_WARN("bad ba1!!!!!!!!!!");

            if(abs(dba2) < 22) {
                para_speed_bias[i][4] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][4];
                Bas[i][1] = para_speed_bias[i][4];
            } else
                ROS_WARN("bad ba2!!!!!!!!!!");

            if(abs(dba3) < 22) {
                para_speed_bias[i][5] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][5];
                Bas[i][2] = para_speed_bias[i][5];
            } else
                ROS_WARN("bad ba3!!!!!!!!!!");

            if(abs(dbg1) < 22) {
                para_speed_bias[i][6] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][6];
                Bgs[i][0] = para_speed_bias[i][6];
            } else
                ROS_WARN("bad bg1!!!!!!!!!!");

            if(abs(dbg2) < 22) {
                para_speed_bias[i][7] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][7];
                Bgs[i][1] = para_speed_bias[i][7];
            } else
                ROS_WARN("bad bg2!!!!!!!!!!");

            if(abs(dbg3) < 22) {
                para_speed_bias[i][8] = tmpSpeedBias[i-keyframe_idx[keyframe_idx.size()-slide_window_width]][8];
                Bgs[i][2] = para_speed_bias[i][8];
            } else
                ROS_WARN("bad bg3!!!!!!!!!!");
        }

        updatePose();
    }


    void updatePose()
    {
        abs_pose = abs_poses.back();
        for (int i = keyframe_idx[keyframe_idx.size()-slide_window_width]; i <= keyframe_idx[keyframe_idx.size()-1]; ++i){
            pose_cloud_frame->points[i-1].x = abs_poses[i][4];
            pose_cloud_frame->points[i-1].y = abs_poses[i][5];
            pose_cloud_frame->points[i-1].z = abs_poses[i][6];

            pose_info_cloud_frame->points[i-1].x = abs_poses[i][4];
            pose_info_cloud_frame->points[i-1].y = abs_poses[i][5];
            pose_info_cloud_frame->points[i-1].z = abs_poses[i][6];
            pose_info_cloud_frame->points[i-1].qw = abs_poses[i][0];
            pose_info_cloud_frame->points[i-1].qx = abs_poses[i][1];
            pose_info_cloud_frame->points[i-1].qy = abs_poses[i][2];
            pose_info_cloud_frame->points[i-1].qz = abs_poses[i][3];
        }
    }


    void optimizeLocalGraph(vector<double*> paraEach) {
        ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
        ceres::Problem problem;

        int numPara = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width] - keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1;
        double dQuat[numPara][4];
        double dTrans[numPara][3];

        for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
            i < keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {
            dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0] = pose_each_frame->points[i].x;
            dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1] = pose_each_frame->points[i].y;
            dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2] = pose_each_frame->points[i].z;

            dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0] = pose_info_each_frame->points[i].qw;
            dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1] = pose_info_each_frame->points[i].qx;
            dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2] = pose_info_each_frame->points[i].qy;
            dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][3] = pose_info_each_frame->points[i].qz;

            problem.AddParameterBlock(dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1], 3);
            problem.AddParameterBlock(dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1], 4, quatParameterization);
        }



        ceres::CostFunction *LeftFactor = LidarPoseLeftFactorAutoDiff::Create(Eigen::Quaterniond(paraEach[1][0], paraEach[1][1], paraEach[1][2], paraEach[1][3]),
                Eigen::Vector3d(paraEach[0][0], paraEach[0][1], paraEach[0][2]),
                Eigen::Quaterniond(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].qw,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].qx,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].qy,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].qz),
                Eigen::Vector3d(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].x,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].y,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width - 1].z));
        problem.AddResidualBlock(LeftFactor, NULL, dTrans[0], dQuat[0]);
        for(int i = 0; i < numPara - 1; i++) {
            ceres::CostFunction *Factor = LidarPoseFactorAutoDiff::Create(Eigen::Quaterniond(paraEach[2*i+1][0], paraEach[2*i+1][1], paraEach[2*i+1][2], paraEach[2*i+1][3]),
                    Eigen::Vector3d(paraEach[2*i][0], paraEach[2*i][1], paraEach[2*i][2]));
            problem.AddResidualBlock(Factor, NULL, dTrans[i], dQuat[i], dTrans[i+1], dQuat[i+1]);

        }

        ceres::CostFunction *RightFactor = LidarPoseRightFactorAutoDiff::Create(Eigen::Quaterniond(paraEach.back()[0], paraEach.back()[1], paraEach.back()[2], paraEach.back()[3]),
                Eigen::Vector3d(paraEach[paraEach.size()-2][0], paraEach[paraEach.size()-2][1], paraEach[paraEach.size()-2][2]),
                Eigen::Quaterniond(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].qw,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].qx,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].qy,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].qz),
                Eigen::Vector3d(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].x,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].y,
                pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width].z));
        problem.AddResidualBlock(RightFactor, NULL, dTrans[numPara-1], dQuat[numPara-1]);


        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 15;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
            i < keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {
            pose_each_frame->points[i].x = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0];
            pose_each_frame->points[i].y = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1];
            pose_each_frame->points[i].z = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2];

            pose_info_each_frame->points[i].x = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0];
            pose_info_each_frame->points[i].y = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1];
            pose_info_each_frame->points[i].z = dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2];
            pose_info_each_frame->points[i].qw = dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0];
            pose_info_each_frame->points[i].qx = dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1];
            pose_info_each_frame->points[i].qy = dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2];
            pose_info_each_frame->points[i].qz = dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][3];
        }
    }


    void buildLocalMapWithLandMark()
    {
        // Initialization
        if (pose_cloud_frame->points.size() < 1)
        {
            PointPoseInfo Tbl;
            Tbl.qw = q_bl.w();
            Tbl.qx = q_bl.x();
            Tbl.qy = q_bl.y();
            Tbl.qz = q_bl.z();
            Tbl.x = t_bl.x();
            Tbl.y = t_bl.y();
            Tbl.z = t_bl.z();
            // ROS_INFO("Initialization for local map building");
            *edge_local_map += *transformCloud(edge_last, &Tbl);
            *surf_local_map += *transformCloud(surf_last, &Tbl);
            return;
        }


        if (recent_surf_keyframes.size() < local_map_width)
        {
            recent_edge_keyframes.clear();
            recent_surf_keyframes.clear();

            for (int i = pose_cloud_frame->points.size() - 1; i >= 0; --i)
            {
                int idx = (int)pose_cloud_frame->points[i].intensity;

                Eigen::Quaterniond q_po(pose_info_cloud_frame->points[idx].qw,
                                        pose_info_cloud_frame->points[idx].qx,
                                        pose_info_cloud_frame->points[idx].qy,
                                        pose_info_cloud_frame->points[idx].qz);

                Eigen::Vector3d t_po(pose_info_cloud_frame->points[idx].x,
                                     pose_info_cloud_frame->points[idx].y,
                                     pose_info_cloud_frame->points[idx].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                recent_edge_keyframes.push_front(transformCloud(edge_frames[idx], &Ttmp));
                recent_surf_keyframes.push_front(transformCloud(surf_frames[idx], &Ttmp));

                if (recent_surf_keyframes.size() >= local_map_width)
                    break;
            }
        }
        // If already more then 50 frames, pop the frames at the beginning
        else
        {
            if (latest_frame_idx != pose_cloud_frame->points.size() - 1)
            {
                recent_edge_keyframes.pop_front();
                recent_surf_keyframes.pop_front();
                latest_frame_idx = pose_cloud_frame->points.size() - 1;

                Eigen::Quaterniond q_po(pose_info_cloud_frame->points[latest_frame_idx].qw,
                                        pose_info_cloud_frame->points[latest_frame_idx].qx,
                                        pose_info_cloud_frame->points[latest_frame_idx].qy,
                                        pose_info_cloud_frame->points[latest_frame_idx].qz);

                Eigen::Vector3d t_po(pose_info_cloud_frame->points[latest_frame_idx].x,
                                     pose_info_cloud_frame->points[latest_frame_idx].y,
                                     pose_info_cloud_frame->points[latest_frame_idx].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                recent_edge_keyframes.push_back(transformCloud(edge_frames[latest_frame_idx], &Ttmp));
                recent_surf_keyframes.push_back(transformCloud(surf_frames[latest_frame_idx], &Ttmp));
            }
        }

        for (int i = 0; i < recent_surf_keyframes.size(); ++i)
        {
            *edge_local_map += *recent_edge_keyframes[i];
            *surf_local_map += *recent_surf_keyframes[i];
        }
    }

    void downSampleCloud()
    {
        ds_filter_surf_map.setInputCloud(surf_local_map);
        ds_filter_surf_map.filter(*surf_local_map_ds);

        ds_filter_edge_map.setInputCloud(edge_local_map);
        ds_filter_edge_map.filter(*edge_local_map_ds);

        pcl::PointCloud<PointType>::Ptr fullDS(new pcl::PointCloud<PointType>());
        ds_filter_surf_map.setInputCloud(full_cloud);
        ds_filter_surf_map.filter(*fullDS);
        full_clouds_ds.push_back(fullDS);
        pcl::PointCloud<PointType>::Ptr full(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*full_cloud, *full);
        full_clouds.push_back(full);

        surf_last_ds->clear();
        ds_filter_surf.setInputCloud(surf_last);
        ds_filter_surf.filter(*surf_last_ds);
        pcl::PointCloud<PointType>::Ptr surf(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*surf_last_ds, *surf);
        surf_lasts_ds.push_back(surf);

        edge_last_ds->clear();
        ds_filter_edge.setInputCloud(edge_last);
        ds_filter_edge.filter(*edge_last_ds);
        pcl::PointCloud<PointType>::Ptr corner(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*edge_last_ds, *corner);
        edge_lasts_ds.push_back(corner);

        sensor_msgs::PointCloud2 lm_msgs;

        pcl::toROSMsg(*surf_local_map_ds, lm_msgs);
        lm_msgs.header.stamp = ros::Time().fromSec(time_new_odom);
        lm_msgs.header.frame_id = frame_id;
        pub_local_surfs.publish(lm_msgs);

        pcl::toROSMsg(*edge_local_map_ds, lm_msgs);
        lm_msgs.header.stamp = ros::Time().fromSec(time_new_odom);
        lm_msgs.header.frame_id = frame_id;
        pub_local_edges.publish(lm_msgs);

    }


    void findCorrespondingCornerFeatures(int idx, Eigen::Quaterniond q, Eigen::Vector3d t)
    {
        int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;
        vec_edge_res_cnt[idVec] = 0;

        for (int i = 0; i < edge_lasts_ds[idx]->points.size(); ++i)
        {
            pt_in_local = edge_lasts_ds[idx]->points[i];

            transformPoint(&pt_in_local, &pt_in_map, q, t);
            kd_tree_edge_local_map->nearestKSearch(pt_in_map, 5, pt_search_idx, pt_search_sq_dists);

            if (pt_search_sq_dists[4] < 1.0)
            {
                std::vector<Eigen::Vector3d> nearCorners;
                Eigen::Vector3d center(0, 0, 0);
                for (int j = 0; j < 5; ++j)
                {
                    Eigen::Vector3d pt(edge_local_map_ds->points[pt_search_idx[j]].x,
                            edge_local_map_ds->points[pt_search_idx[j]].y,
                            edge_local_map_ds->points[pt_search_idx[j]].z);
                    center = center + pt;
                    nearCorners.push_back(pt);
                }
                center /= 5.0;

                // Covariance matrix of distance error
                Eigen::Matrix3d matA1 = Eigen::Matrix3d::Zero();

                for (int j = 0; j < 5; ++j)
                {
                    Eigen::Vector3d zeroMean = nearCorners[j] - center;
                    matA1 = matA1 + zeroMean * zeroMean.transpose();
                }

                // Computes eigenvalues and eigenvectors of selfadjoint matrices
                // The eigenvalues re sorted in increasing order
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSolver(matA1);

                Eigen::Vector3d unitDirection = eigenSolver.eigenvectors().col(2);

                double d_norm = unitDirection.x() * pt_in_local.normal_x + unitDirection.y() * pt_in_local.normal_y + unitDirection.z() * pt_in_local.normal_z;

                // if one eigenvalue is significantly larger than the other two
                if (eigenSolver.eigenvalues()[2] > 3 * eigenSolver.eigenvalues()[1])
                {
                    Eigen::Vector3d ptOnLine = center;
                    Eigen::Vector3d ptA, ptB;
                    ptA = ptOnLine + 0.1 * unitDirection;
                    ptB = ptOnLine - 0.1 * unitDirection;
                    pt_in_local.intensity = lidar_const;

                    PointType pointA, pointB;
                    pointA.x = ptA.x();
                    pointA.y = ptA.y();
                    pointA.z = ptA.z();
                    pointB.x = ptB.x();
                    pointB.y = ptB.y();
                    pointB.z = ptB.z();
                    vec_edge_cur_pts[idVec]->push_back(pt_in_local);
                    vec_edge_match_j[idVec]->push_back(pointA);
                    vec_edge_match_l[idVec]->push_back(pointB);

                    ++vec_edge_res_cnt[idVec];
                }
            }

        }
    }

    void findCorrespondingSurfFeatures(int idx, Eigen::Quaterniond q, Eigen::Vector3d t)
    {
        int idVec = idx - keyframe_idx[keyframe_idx.size()-slide_window_width] + 1;
        vec_surf_res_cnt[idVec] = 0;

        for (int i = 0; i < surf_lasts_ds[idx]->points.size(); ++i)
        {
            pt_in_local = surf_lasts_ds[idx]->points[i];

            transformPoint(&pt_in_local, &pt_in_map, q, t);
            kd_tree_surf_local_map->nearestKSearch(pt_in_map, 5, pt_search_idx, pt_search_sq_dists);

            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = - Eigen::Matrix<double, 5, 1>::Ones();
            if (pt_search_sq_dists[4] < kd_max_radius)
            {
                Eigen::Matrix<double, 5, 1> vec_w;
                double sum_w = 0;
                double sum_w_inv = 0;
                for (int j = 0; j < 5; ++j)
                {
                    double tmp_w = fabs(pt_in_map.curvature - surf_local_map_ds->points[pt_search_idx[j]].curvature);
                    sum_w += tmp_w;
                    sum_w_inv += 1.0 / tmp_w;
                    vec_w[j] = 1.0 / tmp_w;
                }
                vec_w /= sum_w;
                if(sum_w > reflect_thres) {
                    continue;
                }

                for (int j = 0; j < 5; ++j)
                {
                    matA0(j, 0) = vec_w(j, 0) * surf_local_map_ds->points[pt_search_idx[j]].x;
                    matA0(j, 1) = vec_w(j, 0) * surf_local_map_ds->points[pt_search_idx[j]].y;
                    matA0(j, 2) = vec_w(j, 0) * surf_local_map_ds->points[pt_search_idx[j]].z;
                    matB0(j, 0) *= vec_w(j, 0);
                }

                // Get the norm of the plane using linear solver based on QR composition
                Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
                double normInverse = 1 / norm.norm();
                norm.normalize(); // get the unit norm

                // Make sure that the plan is fit
                bool planeValid = true;
                for (int j = 0; j < 5; ++j)
                {
                    if (fabs(norm.x() * surf_local_map_ds->points[pt_search_idx[j]].x +
                             norm.y() * surf_local_map_ds->points[pt_search_idx[j]].y +
                             norm.z() * surf_local_map_ds->points[pt_search_idx[j]].z + normInverse) > surf_dist_thres)
                    {
                        planeValid = false;
                        break;
                    }
                }

                // if one eigenvalue is significantly larger than the other two
                if (planeValid)
                {
                    float pd = norm.x() * pt_in_map.x + norm.y() * pt_in_map.y + norm.z() *pt_in_map.z + normInverse;
                    float weight = 1 - 0.9 * fabs(pd) / sqrt(sqrt(pt_in_map.x * pt_in_map.x + pt_in_map.y * pt_in_map.y + pt_in_map.z * pt_in_map.z));
                    double d_norm = norm.x() * pt_in_local.normal_x + norm.y() * pt_in_local.normal_y + norm.z() * pt_in_local.normal_z;

                    if(weight > 0.2) {
                        PointType normal;
                        normal.x = weight * norm.x();
                        normal.y = weight * norm.y();
                        normal.z = weight * norm.z();
                        normal.intensity = weight * normInverse;

                        vec_surf_cur_pts[idVec]->push_back(pt_in_local);
                        vec_surf_normal[idVec]->push_back(normal);

                        ++vec_surf_res_cnt[idVec];
                        vec_surf_scores[idVec].push_back(lidar_const*(weight + exp(-sum_w)));
                    }
                }
            }
        }
    }

    void saveKeyFramesAndFactors()
    {
        abs_poses.push_back(abs_pose);
        keyframe_id_in_frame.push_back(each_odom_buf.size()-1);

        pcl::PointCloud<PointType>::Ptr cornerEachFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfEachFrame(new pcl::PointCloud<PointType>());

        pcl::copyPointCloud(*edge_last_ds, *cornerEachFrame);
        pcl::copyPointCloud(*surf_last_ds, *surfEachFrame);

        edge_frames.push_back(cornerEachFrame);
        surf_frames.push_back(surfEachFrame);

        //record index of kayframe on imu preintegration poses
        keyframe_idx.push_back(abs_poses.size()-1);

        double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

        int i = idx_imu;
        Eigen::Quaterniond tmpOrient;
        double timeodom_cur = odom_cur->header.stamp.toSec();
        if(imu_buf[i]->header.stamp.toSec() > timeodom_cur)
            ROS_WARN("Timestamp not synchronized, please check your hardware!");
        while(imu_buf[i]->header.stamp.toSec() < timeodom_cur) {

            double t = imu_buf[i]->header.stamp.toSec();
            if (cur_time_imu < 0)
                cur_time_imu = t;
            double dt = t - cur_time_imu;
            cur_time_imu = imu_buf[i]->header.stamp.toSec();
            dx = imu_buf[i]->linear_acceleration.x;
            dy = imu_buf[i]->linear_acceleration.y;
            dz = imu_buf[i]->linear_acceleration.z;
            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            rx = imu_buf[i]->angular_velocity.x;
            ry = imu_buf[i]->angular_velocity.y;
            rz = imu_buf[i]->angular_velocity.z;

            tmpOrient = Eigen::Quaterniond(imu_buf[i]->orientation.w,
                                           imu_buf[i]->orientation.x,
                                           imu_buf[i]->orientation.y,
                                           imu_buf[i]->orientation.z);
            processIMU(dt, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
            i++;
            if(i >= imu_buf.size())
                break;
        }

        imu_idx_in_kf.push_back(i - 1);

        if(i < imu_buf.size()) {
            double dt1 = timeodom_cur - cur_time_imu;
            double dt2 = imu_buf[i]->header.stamp.toSec() - timeodom_cur;

            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);

            Eigen::Quaterniond orient1 = Eigen::Quaterniond(imu_buf[i]->orientation.w,
                                                            imu_buf[i]->orientation.x,
                                                            imu_buf[i]->orientation.y,
                                                            imu_buf[i]->orientation.z);
            tmpOrient = tmpOrient.slerp(w2, orient1);

            dx = w1 * dx + w2 * imu_buf[i]->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_buf[i]->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_buf[i]->linear_acceleration.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            rx = w1 * rx + w2 * imu_buf[i]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[i]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[i]->angular_velocity.z;
            processIMU(dt1, Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
        }
        cur_time_imu = timeodom_cur;


        vector<double> tmpSpeedBias;
        tmpSpeedBias.push_back(Vs.back().x());
        tmpSpeedBias.push_back(Vs.back().y());
        tmpSpeedBias.push_back(Vs.back().z());
        tmpSpeedBias.push_back(Bas.back().x());
        tmpSpeedBias.push_back(Bas.back().y());
        tmpSpeedBias.push_back(Bas.back().z());
        tmpSpeedBias.push_back(Bgs.back().x());
        tmpSpeedBias.push_back(Bgs.back().y());
        tmpSpeedBias.push_back(Bgs.back().z());
        para_speed_bias.push_back(tmpSpeedBias);
        idx_imu = i;

        PointXYZI latestPose;
        PointPoseInfo latestPoseInfo;
        latestPose.x = Ps.back().x();
        latestPose.y = Ps.back().y();
        latestPose.z = Ps.back().z();
        latestPose.intensity = pose_cloud_frame->points.size();
        pose_cloud_frame->push_back(latestPose);

        latestPoseInfo.x = Ps.back().x();
        latestPoseInfo.y = Ps.back().y();
        latestPoseInfo.z = Ps.back().z();
        Eigen::Quaterniond qs_last(Rs.back());
        latestPoseInfo.qw = qs_last.w();
        latestPoseInfo.qx = qs_last.x();
        latestPoseInfo.qy = qs_last.y();
        latestPoseInfo.qz = qs_last.z();
        latestPoseInfo.idx = pose_cloud_frame->points.size();
        latestPoseInfo.time = time_new_odom;

        pose_info_cloud_frame->push_back(latestPoseInfo);

        //optimize sliding window
        num_kf_sliding++;
        if(num_kf_sliding >= 1 || !first_opt) {
            optimizeSlidingWindowWithLandMark();
            num_kf_sliding = 0;
        }

        buildLocalPoseGraph();

        if (!loop_closure_on)
            return;

        //add poses to global graph
        if (pose_cloud_frame->points.size() == slide_window_width)
        {
            gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(pose_info_each_frame->points[0].qw,
                    pose_info_each_frame->points[0].qx,
                    pose_info_each_frame->points[0].qy,
                    pose_info_each_frame->points[0].qz);
            gtsam::Point3 transition = gtsam::Point3(pose_each_frame->points[0].x,
                    pose_each_frame->points[0].y,
                    pose_each_frame->points[0].z);

            // Initialization for global pose graph
            glocal_pose_graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(rotation, transition), prior_noise));
            glocal_init_estimate.insert(0, gtsam::Pose3(rotation, transition));

            for (int i = 0; i < 7; ++i)
            {
                last_pose[i] = abs_poses[abs_poses.size()-slide_window_width][i];
            }
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];
        }

        else if(pose_cloud_frame->points.size() > slide_window_width)
        {
            for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
                i <= keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {
                gtsam::Rot3 rotationLast = gtsam::Rot3::Quaternion(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz);
                gtsam::Point3 transitionLast = gtsam::Point3(pose_each_frame->points[i-1].x,
                        pose_each_frame->points[i-1].y,
                        pose_each_frame->points[i-1].z);

                gtsam::Rot3 rotationCur = gtsam::Rot3::Quaternion(pose_info_each_frame->points[i].qw,
                                                                  pose_info_each_frame->points[i].qx,
                                                                  pose_info_each_frame->points[i].qy,
                                                                  pose_info_each_frame->points[i].qz);
                gtsam::Point3 transitionCur = gtsam::Point3(pose_each_frame->points[i].x,
                                                            pose_each_frame->points[i].y,
                                                            pose_each_frame->points[i].z);
                gtsam::Pose3 poseFrom = gtsam::Pose3(rotationLast, transitionLast);
                gtsam::Pose3 poseTo = gtsam::Pose3(rotationCur, transitionCur);

                glocal_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1,
                                                                   i,
                                                                   poseFrom.between(poseTo),
                                                                   odom_noise));
                glocal_init_estimate.insert(i, poseTo);
            }
        }

        isam->update(glocal_pose_graph, glocal_init_estimate);
        isam->update();

        glocal_pose_graph.resize(0);
        glocal_init_estimate.clear();

        if (pose_cloud_frame->points.size() > slide_window_width)
        {
            for (int i = 0; i < 7; ++i)
            {
                last_pose[i] = abs_poses[abs_poses.size()-slide_window_width][i];
            }
            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];
        }
    }

    void buildLocalPoseGraph() {
        if (pose_cloud_frame->points.size() == slide_window_width) {
            pose_each_frame->push_back(pose_cloud_frame->points[0]);
            pose_info_each_frame->push_back(pose_info_cloud_frame->points[0]);
        } else if(pose_cloud_frame->points.size() > slide_window_width) {
            int ii = imu_idx_in_kf[imu_idx_in_kf.size() - slide_window_width - 1];
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            Eigen::Vector3d Ptmp = Ps[Ps.size() - slide_window_width];
            Eigen::Vector3d Vtmp = Vs[Ps.size() - slide_window_width];
            Eigen::Matrix3d Rtmp = Rs[Ps.size() - slide_window_width];
            Eigen::Vector3d Batmp = Eigen::Vector3d::Zero();
            Eigen::Vector3d Bgtmp = Eigen::Vector3d::Zero();

            for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
                i < keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {

                double dt1 = each_odom_buf[i-1]->header.stamp.toSec() - imu_buf[ii]->header.stamp.toSec();
                double dt2 = imu_buf[ii+1]->header.stamp.toSec() - each_odom_buf[i-1]->header.stamp.toSec();
                double w1 = dt2 / (dt1 + dt2);
                double w2 = dt1 / (dt1 + dt2);
                dx = w1 * imu_buf[ii]->linear_acceleration.x + w2 * imu_buf[ii+1]->linear_acceleration.x;
                dy = w1 * imu_buf[ii]->linear_acceleration.y + w2 * imu_buf[ii+1]->linear_acceleration.y;
                dz = w1 * imu_buf[ii]->linear_acceleration.z + w2 * imu_buf[ii+1]->linear_acceleration.z;

                rx = w1 * imu_buf[ii]->angular_velocity.x + w2 * imu_buf[ii+1]->angular_velocity.x;
                ry = w1 * imu_buf[ii]->angular_velocity.y + w2 * imu_buf[ii+1]->angular_velocity.y;
                rz = w1 * imu_buf[ii]->angular_velocity.z + w2 * imu_buf[ii+1]->angular_velocity.z;
                Eigen::Vector3d a0(dx, dy, dz);
                Eigen::Vector3d gy0(rx, ry, rz);
                ii++;
                double integStartTime = each_odom_buf[i-1]->header.stamp.toSec();

                while(imu_buf[ii]->header.stamp.toSec() < each_odom_buf[i]->header.stamp.toSec()) {
                    double t = imu_buf[ii]->header.stamp.toSec();
                    double dt = t - integStartTime;
                    integStartTime = imu_buf[ii]->header.stamp.toSec();
                    dx = imu_buf[ii]->linear_acceleration.x;
                    dy = imu_buf[ii]->linear_acceleration.y;
                    dz = imu_buf[ii]->linear_acceleration.z;

                    rx = imu_buf[ii]->angular_velocity.x;
                    ry = imu_buf[ii]->angular_velocity.y;
                    rz = imu_buf[ii]->angular_velocity.z;

                    if(dx > 15.0) dx = 15.0;
                    if(dy > 15.0) dy = 15.0;
                    if(dz > 18.0) dz = 18.0;

                    if(dx < -15.0) dx = -15.0;
                    if(dy < -15.0) dy = -15.0;
                    if(dz < -18.0) dz = -18.0;

                    Eigen::Vector3d a1(dx, dy, dz);
                    Eigen::Vector3d gy1(rx, ry, rz);

                    Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                    Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                    Rtmp *= deltaQ(un_gyr * dt).toRotationMatrix();
                    Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                    Ptmp += dt * Vtmp + 0.5 * dt * dt * un_acc;
                    Vtmp += dt * un_acc;

                    a0 = a1;
                    gy0 = gy1;

                    ii++;
                }

                dt1 = each_odom_buf[i]->header.stamp.toSec() - imu_buf[ii-1]->header.stamp.toSec();
                dt2 = imu_buf[ii]->header.stamp.toSec() - each_odom_buf[i]->header.stamp.toSec();
                w1 = dt2 / (dt1 + dt2);
                w2 = dt1 / (dt1 + dt2);
                dx = w1 * dx + w2 * imu_buf[ii]->linear_acceleration.x;
                dy = w1 * dy + w2 * imu_buf[ii]->linear_acceleration.y;
                dz = w1 * dz + w2 * imu_buf[ii]->linear_acceleration.z;

                rx = w1 * rx + w2 * imu_buf[ii]->angular_velocity.x;
                ry = w1 * ry + w2 * imu_buf[ii]->angular_velocity.y;
                rz = w1 * rz + w2 * imu_buf[ii]->angular_velocity.z;

                if(dx > 15.0) dx = 15.0;
                if(dy > 15.0) dy = 15.0;
                if(dz > 18.0) dz = 18.0;

                if(dx < -15.0) dx = -15.0;
                if(dy < -15.0) dy = -15.0;
                if(dz < -18.0) dz = -18.0;

                Eigen::Vector3d a1(dx, dy, dz);
                Eigen::Vector3d gy1(rx, ry, rz);

                Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                Rtmp *= deltaQ(un_gyr * dt1).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ptmp += dt1 * Vtmp + 0.5 * dt1 * dt1 * un_acc;
                Vtmp += dt1 * un_acc;

                ii--;

                Eigen::Quaterniond qqq(Rtmp);

                PointXYZI latestPose;
                PointPoseInfo latestPoseInfo;
                latestPose.x = Ptmp.x();
                latestPose.y = Ptmp.y();
                latestPose.z = Ptmp.z();
                pose_each_frame->push_back(latestPose);

                latestPoseInfo.x = Ptmp.x();
                latestPoseInfo.y = Ptmp.y();
                latestPoseInfo.z = Ptmp.z();
                latestPoseInfo.qw = qqq.w();
                latestPoseInfo.qx = qqq.x();
                latestPoseInfo.qy = qqq.y();
                latestPoseInfo.qz = qqq.z();
                latestPoseInfo.time = each_odom_buf[i]->header.stamp.toSec();
                pose_info_each_frame->push_back(latestPoseInfo);
            }
            pose_each_frame->push_back(pose_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width]);
            pose_info_each_frame->push_back(pose_info_cloud_frame->points[pose_cloud_frame->points.size() - slide_window_width]);
            int j = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width];

            double dt1 = each_odom_buf[j-1]->header.stamp.toSec() - imu_buf[ii]->header.stamp.toSec();
            double dt2 = imu_buf[ii+1]->header.stamp.toSec() - each_odom_buf[j-1]->header.stamp.toSec();
            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);
            dx = w1 * imu_buf[ii]->linear_acceleration.x + w2 * imu_buf[ii+1]->linear_acceleration.x;
            dy = w1 * imu_buf[ii]->linear_acceleration.y + w2 * imu_buf[ii+1]->linear_acceleration.y;
            dz = w1 * imu_buf[ii]->linear_acceleration.z + w2 * imu_buf[ii+1]->linear_acceleration.z;

            rx = w1 * imu_buf[ii]->angular_velocity.x + w2 * imu_buf[ii+1]->angular_velocity.x;
            ry = w1 * imu_buf[ii]->angular_velocity.y + w2 * imu_buf[ii+1]->angular_velocity.y;
            rz = w1 * imu_buf[ii]->angular_velocity.z + w2 * imu_buf[ii+1]->angular_velocity.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            Eigen::Vector3d a0(dx, dy, dz);
            Eigen::Vector3d gy0(rx, ry, rz);
            ii++;
            double integStartTime = each_odom_buf[j-1]->header.stamp.toSec();

            while(imu_buf[ii]->header.stamp.toSec() < each_odom_buf[j]->header.stamp.toSec()) {
                double t = imu_buf[ii]->header.stamp.toSec();
                double dt = t - integStartTime;
                integStartTime = imu_buf[ii]->header.stamp.toSec();
                dx = imu_buf[ii]->linear_acceleration.x;
                dy = imu_buf[ii]->linear_acceleration.y;
                dz = imu_buf[ii]->linear_acceleration.z;

                rx = imu_buf[ii]->angular_velocity.x;
                ry = imu_buf[ii]->angular_velocity.y;
                rz = imu_buf[ii]->angular_velocity.z;

                if(dx > 15.0) dx = 15.0;
                if(dy > 15.0) dy = 15.0;
                if(dz > 18.0) dz = 18.0;

                if(dx < -15.0) dx = -15.0;
                if(dy < -15.0) dy = -15.0;
                if(dz < -18.0) dz = -18.0;

                Eigen::Vector3d a1(dx, dy, dz);
                Eigen::Vector3d gy1(rx, ry, rz);

                Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
                Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
                Rtmp *= deltaQ(un_gyr * dt).toRotationMatrix();
                Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
                Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
                Ptmp += dt * Vtmp + 0.5 * dt * dt * un_acc;
                Vtmp += dt * un_acc;

                a0 = a1;
                gy0 = gy1;

                ii++;
            }

            dt1 = each_odom_buf[j]->header.stamp.toSec() - imu_buf[ii-1]->header.stamp.toSec();
            dt2 = imu_buf[ii]->header.stamp.toSec() - each_odom_buf[j]->header.stamp.toSec();
            w1 = dt2 / (dt1 + dt2);
            w2 = dt1 / (dt1 + dt2);
            dx = w1 * dx + w2 * imu_buf[ii]->linear_acceleration.x;
            dy = w1 * dy + w2 * imu_buf[ii]->linear_acceleration.y;
            dz = w1 * dz + w2 * imu_buf[ii]->linear_acceleration.z;

            rx = w1 * rx + w2 * imu_buf[ii]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[ii]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[ii]->angular_velocity.z;

            if(dx > 15.0) dx = 15.0;
            if(dy > 15.0) dy = 15.0;
            if(dz > 18.0) dz = 18.0;

            if(dx < -15.0) dx = -15.0;
            if(dy < -15.0) dy = -15.0;
            if(dz < -18.0) dz = -18.0;

            Eigen::Vector3d a1(dx, dy, dz);
            Eigen::Vector3d gy1(rx, ry, rz);

            Eigen::Vector3d un_acc_0 = Rtmp * (a0 - Batmp) - g;
            Eigen::Vector3d un_gyr = 0.5 * (gy0 + gy1) - Bgtmp;
            Rtmp *= deltaQ(un_gyr * dt1).toRotationMatrix();
            Eigen::Vector3d un_acc_1 = Rtmp * (a1 - Batmp) - g;
            Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
            Ptmp += dt1 * Vtmp + 0.5 * dt1 * dt1 * un_acc;
            Vtmp += dt1 * un_acc;

            vector<double*> paraBetweenEachFrame;
            int numPara = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width] - keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1];
            double dQuat[numPara][4];
            double dTrans[numPara][3];
            for(int i = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1] + 1;
                i < keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width]; i++) {
                Eigen::Vector3d tmpTrans = Eigen::Vector3d(pose_each_frame->points[i].x,
                                                           pose_each_frame->points[i].y,
                                                           pose_each_frame->points[i].z) -
                        Eigen::Vector3d(pose_each_frame->points[i-1].x,
                        pose_each_frame->points[i-1].y,
                        pose_each_frame->points[i-1].z);
                tmpTrans = Eigen::Quaterniond(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz).inverse() * tmpTrans;
                dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0] = tmpTrans.x();
                dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1] = tmpTrans.y();
                dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2] = tmpTrans.z();
                paraBetweenEachFrame.push_back(dTrans[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1]);

                Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_each_frame->points[i-1].qw,
                        pose_info_each_frame->points[i-1].qx,
                        pose_info_each_frame->points[i-1].qy,
                        pose_info_each_frame->points[i-1].qz).inverse() *
                        Eigen::Quaterniond(pose_info_each_frame->points[i].qw,
                                           pose_info_each_frame->points[i].qx,
                                           pose_info_each_frame->points[i].qy,
                                           pose_info_each_frame->points[i].qz);
                dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][0] = tmpQuat.w();
                dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][1] = tmpQuat.x();
                dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][2] = tmpQuat.y();
                dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1][3] = tmpQuat.z();
                paraBetweenEachFrame.push_back(dQuat[i-keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width - 1]-1]);
            }

            int jj = keyframe_id_in_frame[pose_cloud_frame->points.size() - slide_window_width];

            Eigen::Vector3d tmpTrans = Ptmp - Eigen::Vector3d(pose_each_frame->points[jj-1].x,
                    pose_each_frame->points[jj-1].y,
                    pose_each_frame->points[jj-1].z);
            tmpTrans = Eigen::Quaterniond(pose_info_each_frame->points[jj-1].qw,
                    pose_info_each_frame->points[jj-1].qx,
                    pose_info_each_frame->points[jj-1].qy,
                    pose_info_each_frame->points[jj-1].qz).inverse() * tmpTrans;

            dTrans[numPara-1][0] = tmpTrans.x();
            dTrans[numPara-1][1] = tmpTrans.y();
            dTrans[numPara-1][2] = tmpTrans.z();
            paraBetweenEachFrame.push_back(dTrans[numPara-1]);

            Eigen::Quaterniond qtmp(Rtmp);
            Eigen::Quaterniond tmpQuat = Eigen::Quaterniond(pose_info_each_frame->points[jj-1].qw,
                    pose_info_each_frame->points[jj-1].qx,
                    pose_info_each_frame->points[jj-1].qy,
                    pose_info_each_frame->points[jj-1].qz).inverse() * qtmp;
            dQuat[numPara-1][0] = tmpQuat.w();
            dQuat[numPara-1][1] = tmpQuat.x();
            dQuat[numPara-1][2] = tmpQuat.y();
            dQuat[numPara-1][3] = tmpQuat.z();
            paraBetweenEachFrame.push_back(dQuat[numPara-1]);

            optimizeLocalGraph(paraBetweenEachFrame);
        }

    }

    void correctPoses()
    {
        if (loop_closed == true)
        {
            recent_edge_keyframes.clear();
            recent_surf_keyframes.clear();

            int numPoses = glocal_estimated.size();

            vector<Eigen::Quaterniond> quaternionRel;
            vector<Eigen::Vector3d> transitionRel;

            for(int i = abs_poses.size() - slide_window_width; i < abs_poses.size() - 1; i++) {
                Eigen::Quaterniond quaternionFrom(abs_poses[i][0],
                        abs_poses[i][1],
                        abs_poses[i][2],
                        abs_poses[i][3]);
                Eigen::Vector3d transitionFrom(abs_poses[i][4],
                        abs_poses[i][5],
                        abs_poses[i][6]);

                Eigen::Quaterniond quaternionTo(abs_poses[i+1][0],
                        abs_poses[i+1][1],
                        abs_poses[i+1][2],
                        abs_poses[i+1][3]);
                Eigen::Vector3d transitionTo(abs_poses[i+1][4],
                        abs_poses[i+1][5],
                        abs_poses[i+1][6]);

                quaternionRel.push_back(quaternionFrom.inverse() * quaternionTo);
                transitionRel.push_back(quaternionFrom.inverse() * (transitionTo - transitionFrom));
            }

            for (int i = 0; i < numPoses; ++i)
            {
                pose_each_frame->points[i].x = glocal_estimated.at<gtsam::Pose3>(i).translation().x();
                pose_each_frame->points[i].y = glocal_estimated.at<gtsam::Pose3>(i).translation().y();
                pose_each_frame->points[i].z = glocal_estimated.at<gtsam::Pose3>(i).translation().z();

                pose_info_each_frame->points[i].x = pose_each_frame->points[i].x;
                pose_info_each_frame->points[i].y = pose_each_frame->points[i].y;
                pose_info_each_frame->points[i].z = pose_each_frame->points[i].z;
                pose_info_each_frame->points[i].qw = glocal_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().w();
                pose_info_each_frame->points[i].qx = glocal_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().x();
                pose_info_each_frame->points[i].qy = glocal_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().y();
                pose_info_each_frame->points[i].qz = glocal_estimated.at<gtsam::Pose3>(i).rotation().toQuaternion().z();
            }

            for(int i = 0; i <= pose_cloud_frame->points.size() - slide_window_width; i++) {
                pose_cloud_frame->points[i].x = pose_each_frame->points[keyframe_id_in_frame[i]].x;
                pose_cloud_frame->points[i].y = pose_each_frame->points[keyframe_id_in_frame[i]].y;
                pose_cloud_frame->points[i].z = pose_each_frame->points[keyframe_id_in_frame[i]].z;

                pose_info_cloud_frame->points[i].x = pose_each_frame->points[keyframe_id_in_frame[i]].x;
                pose_info_cloud_frame->points[i].y = pose_each_frame->points[keyframe_id_in_frame[i]].y;
                pose_info_cloud_frame->points[i].z = pose_each_frame->points[keyframe_id_in_frame[i]].z;
                pose_info_cloud_frame->points[i].qw = pose_info_each_frame->points[keyframe_id_in_frame[i]].qw;
                pose_info_cloud_frame->points[i].qx = pose_info_each_frame->points[keyframe_id_in_frame[i]].qx;
                pose_info_cloud_frame->points[i].qy = pose_info_each_frame->points[keyframe_id_in_frame[i]].qy;
                pose_info_cloud_frame->points[i].qz = pose_info_each_frame->points[keyframe_id_in_frame[i]].qz;

                abs_poses[i+1][0] = pose_info_cloud_frame->points[i].qw;
                abs_poses[i+1][1] = pose_info_cloud_frame->points[i].qx;
                abs_poses[i+1][2] = pose_info_cloud_frame->points[i].qy;
                abs_poses[i+1][3] = pose_info_cloud_frame->points[i].qz;
                abs_poses[i+1][4] = pose_info_cloud_frame->points[i].x;
                abs_poses[i+1][5] = pose_info_cloud_frame->points[i].y;
                abs_poses[i+1][6] = pose_info_cloud_frame->points[i].z;

                Rs[i+1] = Eigen::Quaterniond(abs_poses[i+1][0],
                        abs_poses[i+1][1],
                        abs_poses[i+1][2],
                        abs_poses[i+1][3]).toRotationMatrix();

                Ps[i+1][0] = abs_poses[i+1][4];
                Ps[i+1][1] = abs_poses[i+1][5];
                Ps[i+1][2] = abs_poses[i+1][6];
            }

            for(int i = abs_poses.size() - slide_window_width; i < abs_poses.size() - 1; i++) {
                Eigen::Quaterniond integratedQuaternion(abs_poses[i][0],
                        abs_poses[i][1],
                        abs_poses[i][2],
                        abs_poses[i][3]);
                Eigen::Vector3d integratedTransition(abs_poses[i][4],
                        abs_poses[i][5],
                        abs_poses[i][6]);

                integratedTransition = integratedTransition + integratedQuaternion * transitionRel[i - abs_poses.size() + slide_window_width];
                integratedQuaternion = integratedQuaternion * quaternionRel[i - abs_poses.size() + slide_window_width];

                abs_poses[i+1][0] = integratedQuaternion.w();
                abs_poses[i+1][1] = integratedQuaternion.x();
                abs_poses[i+1][2] = integratedQuaternion.y();
                abs_poses[i+1][3] = integratedQuaternion.z();
                abs_poses[i+1][4] = integratedTransition.x();
                abs_poses[i+1][5] = integratedTransition.y();
                abs_poses[i+1][6] = integratedTransition.z();

                Rs[i+1] = Eigen::Quaterniond(abs_poses[i+1][0],
                        abs_poses[i+1][1],
                        abs_poses[i+1][2],
                        abs_poses[i+1][3]).toRotationMatrix();

                Ps[i+1][0] = abs_poses[i+1][4];
                Ps[i+1][1] = abs_poses[i+1][5];
                Ps[i+1][2] = abs_poses[i+1][6];

                pose_cloud_frame->points[i].x = abs_poses[i+1][4];
                pose_cloud_frame->points[i].y = abs_poses[i+1][5];
                pose_cloud_frame->points[i].z = abs_poses[i+1][6];

                pose_info_cloud_frame->points[i].x = abs_poses[i+1][4];
                pose_info_cloud_frame->points[i].y = abs_poses[i+1][5];
                pose_info_cloud_frame->points[i].z = abs_poses[i+1][6];
                pose_info_cloud_frame->points[i].qw = abs_poses[i+1][0];
                pose_info_cloud_frame->points[i].qx = abs_poses[i+1][1];
                pose_info_cloud_frame->points[i].qy = abs_poses[i+1][2];
                pose_info_cloud_frame->points[i].qz = abs_poses[i+1][3];
            }

            abs_pose = abs_poses.back();
            for (int i = 0; i < 7; ++i)
            {
                last_pose[i] = abs_poses[abs_poses.size() - slide_window_width][i];
            }

            select_pose.x = last_pose[4];
            select_pose.y = last_pose[5];
            select_pose.z = last_pose[6];

            loop_closed = false;
            marg = false;
        }
    }

    void publishOdometry()
    {
        if(pose_info_cloud_frame->points.size() >= slide_window_width) {
            odom_mapping.header.stamp = ros::Time().fromSec(time_new_odom);
            odom_mapping.pose.pose.orientation.w = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].qw;
            odom_mapping.pose.pose.orientation.x = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].qx;
            odom_mapping.pose.pose.orientation.y = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].qy;
            odom_mapping.pose.pose.orientation.z = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].qz;
            odom_mapping.pose.pose.position.x = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].x;
            odom_mapping.pose.pose.position.y = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].y;
            odom_mapping.pose.pose.position.z = pose_info_cloud_frame->points[pose_info_cloud_frame->points.size()-slide_window_width].z;
            pub_odom.publish(odom_mapping);
        }

        sensor_msgs::PointCloud2 msgs;


        if (pub_poses.getNumSubscribers() && pose_info_cloud_frame->points.size() >= slide_window_width)
        {
            pcl::toROSMsg(*pose_each_frame, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_poses.publish(msgs);
        }


        PointPoseInfo Tbl;
        Tbl.qw = q_bl.w();
        Tbl.qx = q_bl.x();
        Tbl.qy = q_bl.y();
        Tbl.qz = q_bl.z();
        Tbl.x = t_bl.x();
        Tbl.y = t_bl.y();
        Tbl.z = t_bl.z();

        // publish the corner and surf feature points in lidar_init frame
        if (pub_edge.getNumSubscribers())
        {
            for (int i = 0; i < edge_last_ds->points.size(); ++i)
            {
                transformPoint(&edge_last_ds->points[i], &edge_last_ds->points[i], q_bl, t_bl);
                transformPoint(&edge_last_ds->points[i], &edge_last_ds->points[i]);
            }
            pcl::toROSMsg(*edge_last_ds, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_edge.publish(msgs);
        }

        if (pub_surf.getNumSubscribers())
        {
            for (int i = 0; i < surf_last_ds->points.size(); ++i)
            {
                transformPoint(&surf_last_ds->points[i], &surf_last_ds->points[i], q_bl, t_bl);
                transformPoint(&surf_last_ds->points[i], &surf_last_ds->points[i]);
            }
            pcl::toROSMsg(*surf_last_ds, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_surf.publish(msgs);
        }

        if (pub_full.getNumSubscribers())
        {
            for (int i = 0; i < full_cloud->points.size(); ++i)
            {
                transformPoint(&full_cloud->points[i], &full_cloud->points[i], q_bl, t_bl);
                transformPoint(&full_cloud->points[i], &full_cloud->points[i]);
            }
            pcl::toROSMsg(*full_cloud, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_full.publish(msgs);
        }
    }

    void clearCloud()
    {
        edge_local_map->clear();
        edge_local_map_ds->clear();
        surf_local_map->clear();
        surf_local_map_ds->clear();

        if(surf_lasts_ds.size() > slide_window_width + 5) {
            surf_lasts_ds[surf_lasts_ds.size() - slide_window_width - 6]->clear();
        }

        if(pre_integrations.size() > slide_window_width + 5) {
            pre_integrations[pre_integrations.size() - slide_window_width - 6] = nullptr;
        }

        if(last_marginalization_parameter_blocks.size() > slide_window_width + 5) {
            last_marginalization_parameter_blocks[last_marginalization_parameter_blocks.size() - slide_window_width - 6] = nullptr;
        }

    }

    void loopClosureThread()
    {
        if (!loop_closure_on)
            return;

        ros::Rate rate(1);
        while (ros::ok())
        {
            rate.sleep();
            performLoopClosure();
        }
    }

    bool detectLoopClosure()
    {
        latest_key_frames->clear();
        latest_key_frames_ds->clear();
        his_key_frames->clear();
        his_key_frames_ds->clear();

        std::lock_guard<std::mutex> lock(mutual_exclusion);

        // Look for the closest key frames
        std::vector<int> pt_search_idxLoop;
        std::vector<float> pt_search_sq_distsLoop;

        kd_tree_his_key_poses->setInputCloud(pose_cloud_frame);
        kd_tree_his_key_poses->radiusSearch(select_pose, lc_search_radius, pt_search_idxLoop, pt_search_sq_distsLoop, 0);

        closest_his_idx = -1;
        for (int i = 0; i < pt_search_idxLoop.size(); ++i)
        {
            int idx = pt_search_idxLoop[i];
            if (abs(pose_info_cloud_frame->points[idx].time - time_new_odom) > global_lc_time_thres)
            {
                closest_his_idx = idx;
                break;
            }
        }

        vector<int> localMapId;
        if (closest_his_idx == -1) {
            double max_time = 0.0;
            int max_id = -1;
            for (int i = 0; i < pt_search_idxLoop.size(); ++i)
            {
                int idx = pt_search_idxLoop[i];
                double d_time = abs(pose_info_cloud_frame->points[idx].time - time_new_odom);
                if (d_time > local_lc_time_thres && d_time < global_lc_time_thres)
                {
                    if(d_time > max_time) {
                        max_time = d_time;
                        max_id = idx;
                    }
                }
            }
            if(max_id == -1)
                return false;
            closest_his_idx = max_id;
        }

        // ROS_INFO("******************* Loop closure ready to detect! *******************");

        // Combine the corner and surf frames to form the latest frame
        latest_frame_idx_loop = pose_cloud_frame->points.size() - slide_window_width;

        for (int j = 0; j < 1; j = j + 1)
        {
            if (latest_frame_idx_loop-j < 0)
                continue;
            Eigen::Quaterniond q_po(pose_info_cloud_frame->points[latest_frame_idx_loop-j].qw,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].qx,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].qy,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].qz);

            Eigen::Vector3d t_po(pose_info_cloud_frame->points[latest_frame_idx_loop-j].x,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].y,
                    pose_info_cloud_frame->points[latest_frame_idx_loop-j].z);

            Eigen::Quaterniond q_tmp = q_po * q_bl;
            Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

            *latest_key_frames += *transformCloud(edge_frames[latest_frame_idx_loop-j], q_tmp, t_tmp);
            *latest_key_frames += *transformCloud(surf_frames[latest_frame_idx_loop-j], q_tmp, t_tmp);
        }

        ds_filter_his_frames.setInputCloud(latest_key_frames);
        ds_filter_his_frames.filter(*latest_key_frames_ds);


        // Form the history frame for loop closure detection
        if(true) {
            for (int j = -lc_map_width; j <= lc_map_width; ++j)
            {
                if (closest_his_idx + j < 0 || closest_his_idx + j > latest_frame_idx_loop)
                    continue;

                Eigen::Quaterniond q_po(pose_info_cloud_frame->points[closest_his_idx+j].qw,
                        pose_info_cloud_frame->points[closest_his_idx+j].qx,
                        pose_info_cloud_frame->points[closest_his_idx+j].qy,
                        pose_info_cloud_frame->points[closest_his_idx+j].qz);

                Eigen::Vector3d t_po(pose_info_cloud_frame->points[closest_his_idx+j].x,
                        pose_info_cloud_frame->points[closest_his_idx+j].y,
                        pose_info_cloud_frame->points[closest_his_idx+j].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                *his_key_frames += *transformCloud(edge_frames[closest_his_idx+j], q_tmp, t_tmp);
                *his_key_frames += *transformCloud(surf_frames[closest_his_idx+j], q_tmp, t_tmp);
            }
        } else {
            for (int j = 0; j < localMapId.size(); ++j)
            {
                if (localMapId[j] < 0 || localMapId[j] > latest_frame_idx_loop)
                    continue;

                Eigen::Quaterniond q_po(pose_info_cloud_frame->points[localMapId[j]].qw,
                        pose_info_cloud_frame->points[localMapId[j]].qx,
                        pose_info_cloud_frame->points[localMapId[j]].qy,
                        pose_info_cloud_frame->points[localMapId[j]].qz);

                Eigen::Vector3d t_po(pose_info_cloud_frame->points[localMapId[j]].x,
                        pose_info_cloud_frame->points[localMapId[j]].y,
                        pose_info_cloud_frame->points[localMapId[j]].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                *his_key_frames += *transformCloud(edge_frames[localMapId[j]], q_tmp, t_tmp);
                *his_key_frames += *transformCloud(surf_frames[localMapId[j]], q_tmp, t_tmp);
            }
        }


        ds_filter_his_frames.setInputCloud(his_key_frames);
        ds_filter_his_frames.filter(*his_key_frames_ds);

        return true;
    }

    void performLoopClosure()
    {
        if (pose_cloud_frame->points.empty())
            return;

        if (!loop_to_close)
        {
            if (detectLoopClosure())
                loop_to_close = true;
            if (!loop_to_close)
                return;
        }

        loop_to_close = false;

        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(30);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(5);

        icp.setInputSource(latest_key_frames_ds);
        icp.setInputTarget(his_key_frames_ds);
        pcl::PointCloud<PointType>::Ptr alignedCloud(new pcl::PointCloud<PointType>());
        icp.align(*alignedCloud);

        // std::cout << "ICP converg flag:" << icp.hasConverged() << ". Fitness score: " << icp.getFitnessScore() << endl;

        if (!icp.hasConverged() || icp.getFitnessScore() > lc_icp_thres)
            return;

        Timer t_loop("Loop Closure");
        // ROS_INFO("******************* Loop closure detected! *******************");

        Eigen::Matrix4d correctedTranform;
        correctedTranform = icp.getFinalTransformation().cast<double>();
        Eigen::Quaterniond quaternionIncre(correctedTranform.block<3, 3>(0, 0));
        Eigen::Vector3d transitionIncre(correctedTranform.block<3, 1>(0, 3));
        Eigen::Quaterniond quaternionToCorrect(pose_info_cloud_frame->points[latest_frame_idx_loop].qw,
                                               pose_info_cloud_frame->points[latest_frame_idx_loop].qx,
                                               pose_info_cloud_frame->points[latest_frame_idx_loop].qy,
                                               pose_info_cloud_frame->points[latest_frame_idx_loop].qz);
        Eigen::Vector3d transitionToCorrect(pose_info_cloud_frame->points[latest_frame_idx_loop].x,
                                            pose_info_cloud_frame->points[latest_frame_idx_loop].y,
                                            pose_info_cloud_frame->points[latest_frame_idx_loop].z);

        Eigen::Quaterniond quaternionCorrected = quaternionIncre * quaternionToCorrect;
        Eigen::Vector3d transitionCorrected = quaternionIncre * transitionToCorrect + transitionIncre;

        gtsam::Rot3 rotationFrom = gtsam::Rot3::Quaternion(quaternionCorrected.w(), quaternionCorrected.x(), quaternionCorrected.y(), quaternionCorrected.z());
        gtsam::Point3 transitionFrom = gtsam::Point3(transitionCorrected.x(), transitionCorrected.y(), transitionCorrected.z());

        gtsam::Rot3 rotationTo = gtsam::Rot3::Quaternion(pose_info_cloud_frame->points[closest_his_idx].qw,
                                                         pose_info_cloud_frame->points[closest_his_idx].qx,
                                                         pose_info_cloud_frame->points[closest_his_idx].qy,
                                                         pose_info_cloud_frame->points[closest_his_idx].qz);
        gtsam::Point3 transitionTo = gtsam::Point3(pose_info_cloud_frame->points[closest_his_idx].x,
                                                   pose_info_cloud_frame->points[closest_his_idx].y,
                                                   pose_info_cloud_frame->points[closest_his_idx].z);

        gtsam::Pose3 poseFrom = gtsam::Pose3(rotationFrom, transitionFrom);
        gtsam::Pose3 poseTo = gtsam::Pose3(rotationTo, transitionTo);
        gtsam::Vector vector6(6);
        double noiseScore = icp.getFitnessScore();
        vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        constraint_noise = gtsam::noiseModel::Diagonal::Variances(vector6);

        std::lock_guard<std::mutex> lock(mutual_exclusion);

        glocal_pose_graph.add(gtsam::BetweenFactor<gtsam::Pose3>(keyframe_id_in_frame[latest_frame_idx_loop],
                                                           keyframe_id_in_frame[closest_his_idx],
                                                           poseFrom.between(poseTo),
                                                           constraint_noise));
        isam->update(glocal_pose_graph);
        isam->update();
        glocal_pose_graph.resize(0);

        loop_closed = true;

        glocal_estimated = isam->calculateEstimate();
        correctPoses();

        if (last_marginalization_info) {
            delete last_marginalization_info;
        }
        last_marginalization_info = nullptr;

        // ROS_INFO("******************* Loop closure finished! *******************");
        // t_loop.tic_toc();
    }

    void publishCompleteMap()
    {
        if (pose_cloud_frame->points.size() > 10)
        {
            for (int i = 0; i < pose_info_cloud_frame->points.size(); i = i + mapping_interval)
            {
                Eigen::Quaterniond q_po(pose_info_cloud_frame->points[i].qw,
                                        pose_info_cloud_frame->points[i].qx,
                                        pose_info_cloud_frame->points[i].qy,
                                        pose_info_cloud_frame->points[i].qz);

                Eigen::Vector3d t_po(pose_info_cloud_frame->points[i].x,
                                     pose_info_cloud_frame->points[i].y,
                                     pose_info_cloud_frame->points[i].z);

                Eigen::Quaterniond q_tmp = q_po * q_bl;
                Eigen::Vector3d t_tmp = q_po * t_bl + t_po;

                PointPoseInfo Ttmp;
                Ttmp.qw = q_tmp.w();
                Ttmp.qx = q_tmp.x();
                Ttmp.qy = q_tmp.y();
                Ttmp.qz = q_tmp.z();
                Ttmp.x = t_tmp.x();
                Ttmp.y = t_tmp.y();
                Ttmp.z = t_tmp.z();

                *global_map += *transformCloud(full_clouds[i], &Ttmp);
            }

            ds_filter_global_map.setInputCloud(global_map);
            ds_filter_global_map.filter(*global_map_ds);

            sensor_msgs::PointCloud2 msgs;
            pcl::toROSMsg(*global_map_ds, msgs);
            msgs.header.stamp = ros::Time().fromSec(time_new_odom);
            msgs.header.frame_id = frame_id;
            pub_map.publish(msgs);
            global_map->clear();
            global_map_ds->clear();
        }
    }

    void mapVisualizationThread()
    {
        ros::Rate rate(0.02);
        while (ros::ok())
        {
            rate.sleep();
            // ROS_INFO("Publishing the map");
            publishCompleteMap();
        }

        if(!save_pcd)
            return;

        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;

        PointPoseInfo Tbl;
        Tbl.qw = q_bl.w();
        Tbl.qx = q_bl.x();
        Tbl.qy = q_bl.y();
        Tbl.qz = q_bl.z();
        Tbl.x = t_bl.x();
        Tbl.y = t_bl.y();
        Tbl.z = t_bl.z();

        for (int i = 0; i < pose_info_cloud_frame->points.size(); i = i + mapping_interval)
        {
            *global_map += *transformCloud(transformCloud(surf_frames[i], &Tbl), &pose_info_cloud_frame->points[i]);
        }
        ds_filter_global_map.setInputCloud(global_map);
        ds_filter_global_map.filter(*global_map_ds);
        pcl::io::savePCDFileASCII("/home/mli/MengLi/pcd/global_map.pcd", *global_map_ds);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
        global_map->clear();
        global_map_ds->clear();
    }

    void run()
    {
        if (new_surf && new_edge && new_odom && new_each_odom && new_full_cloud)
        {
            new_edge = false;
            new_surf = false;
            new_odom = false;
            new_each_odom = false;
            new_full_cloud = false;

            std::lock_guard<std::mutex> lock(mutual_exclusion);

            //cout<<"map_pub_cnt: "<<++map_pub_cnt<<endl;

            Timer t_map("BackendFusion");
            buildLocalMapWithLandMark();
            downSampleCloud();
            saveKeyFramesAndFactors();
            publishOdometry();
            clearCloud();
            // t_map.tic_toc();

            runtime += t_map.toc();
            // cout<<"Backend average run time: "<<runtime / each_odom_buf.size()<<endl;
        }
    }
};

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);

    ros::init(argc, argv, "lili_om");

    ROS_INFO("\033[1;32m---->\033[0m BackendFusion Started.");

    BackendFusion be;

    std::thread threadLoopClosure(&BackendFusion::loopClosureThread, &be);
    std::thread threadMapVisualization(&BackendFusion::mapVisualizationThread, &be);

    ros::Rate rate(200);

    while (ros::ok())
    {
        ros::spinOnce();

        be.run();

        rate.sleep();
    }

    threadLoopClosure.join();
    threadMapVisualization.join();

    return 0;
}
