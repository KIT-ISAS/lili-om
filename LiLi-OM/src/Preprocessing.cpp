#include "utils/common.h"
#include "utils/timer.h"
#include "utils/math_tools.h"

class Preprocessing {
private:
    ros::NodeHandle nh;

    ros::Subscriber sub_Lidar_cloud;
    ros::Subscriber sub_imu;

    ros::Publisher pub_surf;
    ros::Publisher pub_edge;
    ros::Publisher pub_cutted_cloud;

    int pre_num = 0;

    pcl::PointCloud<PointXYZINormal> lidar_cloud_in;
    pcl::PointCloud<PointXYZINormal> lidar_cloud_cutted;
    std_msgs::Header cloud_header;

    vector<sensor_msgs::ImuConstPtr> imu_buf;
    int idx_imu = 0;
    double current_time_imu = -1;

    Eigen::Vector3d gyr_0;
    Eigen::Quaterniond q_iMU = Eigen::Quaterniond::Identity();
    bool first_imu = false;

    std::deque<sensor_msgs::PointCloud2> cloud_queue;
    sensor_msgs::PointCloud2 current_cloud_msg;
    double time_scan_next;

    int N_SCANS = 6;
    int H_SCANS = 4000;

    string frame_id = "lili_om";
    double edge_thres, surf_thres;
    double runtime = 0;

public:
    Preprocessing(): nh("~") {

        if (!getParameter("/preprocessing/surf_thres", surf_thres))
        {
            ROS_WARN("surf_thres not set, use default value: 0.2");
            surf_thres = 0.2;
        }

        if (!getParameter("/preprocessing/edge_thres", edge_thres))
        {
            ROS_WARN("edge_thres not set, use default value: 4.0");
            edge_thres = 4.0;
        }

        if (!getParameter("/common/frame_id", frame_id))
        {
            ROS_WARN("frame_id not set, use default value: lili_om");
            frame_id = "lili_om";
        }

        sub_Lidar_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/livox_ros_points", 100, &Preprocessing::cloudHandler, this);
        sub_imu = nh.subscribe<sensor_msgs::Imu>("/livox/imu", 200, &Preprocessing::imuHandler, this);

        pub_surf = nh.advertise<sensor_msgs::PointCloud2>("/surf_features", 100);
        pub_edge = nh.advertise<sensor_msgs::PointCloud2>("/edge_features", 100);
        pub_cutted_cloud = nh.advertise<sensor_msgs::PointCloud2>("/lidar_cloud_cutted", 100);
    }

    ~Preprocessing(){}

    template <typename PointT>
    void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in, pcl::PointCloud<PointT> &cloud_out, float thres) {

        if (&cloud_in != &cloud_out) {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i) {
            if (cloud_in.points[i].x * cloud_in.points[i].x +
                    cloud_in.points[i].y * cloud_in.points[i].y +
                    cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
                continue;
            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }
        if (j != cloud_in.points.size()) {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    template <typename PointT>
    double getDepth(PointT pt) {
        return sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
    }

    PointXYZINormal undistortion(PointXYZINormal pt, const Eigen::Quaterniond quat) {
        double dt = 0.1;
        int line = int(pt.intensity);
        double dt_i = pt.intensity - line;

        double ratio_i = dt_i / dt;
        if(ratio_i >= 1.0) {
            ratio_i = 1.0;
        }

        Eigen::Quaterniond q0 = Eigen::Quaterniond::Identity();
        Eigen::Quaterniond q_si = q0.slerp(ratio_i, q_iMU);

        Eigen::Vector3d pt_i(pt.x, pt.y, pt.z);
        Eigen::Vector3d pt_s = q_si * pt_i;

        PointXYZINormal p_out;
        p_out.x = pt_s.x();
        p_out.y = pt_s.y();
        p_out.z = pt_s.z();
        p_out.intensity = pt.intensity;
        p_out.curvature = pt.curvature;
        return p_out;
    }

    void solveRotation(double dt, Eigen::Vector3d angular_velocity) {
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity);
        q_iMU *= deltaQ(un_gyr * dt);
        gyr_0 = angular_velocity;
    }

    void processIMU(double t_cur) {
        double rx = 0, ry = 0, rz = 0;
        int i = idx_imu;
        if(i >= imu_buf.size())
            i--;
        while(imu_buf[i]->header.stamp.toSec() < t_cur) {

            double t = imu_buf[i]->header.stamp.toSec();
            if (current_time_imu < 0)
                current_time_imu = t;
            double dt = t - current_time_imu;
            current_time_imu = imu_buf[i]->header.stamp.toSec();

            rx = imu_buf[i]->angular_velocity.x;
            ry = imu_buf[i]->angular_velocity.y;
            rz = imu_buf[i]->angular_velocity.z;
            solveRotation(dt, Eigen::Vector3d(rx, ry, rz));
            i++;
            if(i >= imu_buf.size())
                break;
        }

        if(i < imu_buf.size()) {
            double dt1 = t_cur - current_time_imu;
            double dt2 = imu_buf[i]->header.stamp.toSec() - t_cur;

            double w1 = dt2 / (dt1 + dt2);
            double w2 = dt1 / (dt1 + dt2);

            rx = w1 * rx + w2 * imu_buf[i]->angular_velocity.x;
            ry = w1 * ry + w2 * imu_buf[i]->angular_velocity.y;
            rz = w1 * rz + w2 * imu_buf[i]->angular_velocity.z;
            solveRotation(dt1, Eigen::Vector3d(rx, ry, rz));
        }
        current_time_imu = t_cur;
        idx_imu = i;
    }

    void imuHandler(const sensor_msgs::ImuConstPtr& imu_in) {
        imu_buf.push_back(imu_in);

        if(imu_buf.size() > 600)
            imu_buf[imu_buf.size() - 601] = nullptr;

        if (current_time_imu < 0)
            current_time_imu = imu_in->header.stamp.toSec();

        if (!first_imu)
        {
            first_imu = true;
            double rx = 0, ry = 0, rz = 0;
            rx = imu_in->angular_velocity.x;
            ry = imu_in->angular_velocity.y;
            rz = imu_in->angular_velocity.z;
            Eigen::Vector3d angular_velocity(rx, ry, rz);
            gyr_0 = angular_velocity;
        }
    }

    void cloudHandler( const sensor_msgs::PointCloud2ConstPtr &lidar_cloud_msg) {
        // cache point cloud
        cloud_queue.push_back(*lidar_cloud_msg);
        if (cloud_queue.size() <= 2)
            return;
        else
        {
            current_cloud_msg = cloud_queue.front();
            cloud_queue.pop_front();

            cloud_header = current_cloud_msg.header;
            cloud_header.frame_id = frame_id;
            time_scan_next = cloud_queue.front().header.stamp.toSec();
        }

        int tmp_idx = 0;
        if(idx_imu > 0)
            tmp_idx = idx_imu - 1;
        if (imu_buf.empty() || imu_buf[tmp_idx]->header.stamp.toSec() > time_scan_next)
        {
            ROS_WARN("Waiting for IMU data ...");
            return;
        }

        Timer t_pre("Preprocessing");
        pcl::fromROSMsg(current_cloud_msg, lidar_cloud_in);

        lidar_cloud_cutted.clear();

        std::vector<int> indices;

        pcl::removeNaNFromPointCloud(lidar_cloud_in, lidar_cloud_in, indices);
        removeClosedPointCloud(lidar_cloud_in, lidar_cloud_in, 0.1);

        int cloud_size = lidar_cloud_in.points.size();

        if(imu_buf.size() > 0)
            processIMU(time_scan_next);
        if(isnan(q_iMU.w()) || isnan(q_iMU.x()) || isnan(q_iMU.y()) || isnan(q_iMU.z())) {
            q_iMU = Eigen::Quaterniond::Identity();
        }

        PointXYZINormal point;
        PointXYZINormal point_undis;
        PointXYZINormal mat[N_SCANS][H_SCANS];
        double t_interval = 0.1 / (H_SCANS-1);
        pcl::PointCloud<PointXYZINormal>::Ptr surf_features(new pcl::PointCloud<PointXYZINormal>());
        pcl::PointCloud<PointXYZINormal>::Ptr edge_features(new pcl::PointCloud<PointXYZINormal>());

        for (int i = 0; i < cloud_size; i++) {
            point.x = lidar_cloud_in.points[i].x;
            point.y = lidar_cloud_in.points[i].y;
            point.z = lidar_cloud_in.points[i].z;
            point.intensity = lidar_cloud_in.points[i].intensity;
            point.curvature = lidar_cloud_in.points[i].curvature;

            int scan_id = 0;
            if (N_SCANS == 6)
                scan_id = (int)point.intensity;
            if(scan_id < 0)
                continue;

            point_undis = undistortion(point, q_iMU);
            lidar_cloud_cutted.push_back(point_undis);

            double dep = point_undis.x*point_undis.x + point_undis.y*point_undis.y + point_undis.z*point_undis.z;
            if(dep > 40000.0 || dep < 4.0 || point_undis.curvature < 0.05 || point_undis.curvature > 25.45)
                continue;
            int col = int(round((point_undis.intensity - scan_id) / t_interval));
            if (col >= H_SCANS || col < 0)
                continue;
            if (mat[scan_id][col].curvature != 0)
                continue;
            mat[scan_id][col] = point_undis;
        }

        for(int i = 5; i < H_SCANS - 12; i = i + 6) {
            vector<Eigen::Vector3d> near_pts;
            Eigen::Vector3d center(0, 0, 0);
            int num = 36;
            for(int j = 0; j < 6; j++) {
                for(int k = 0; k < N_SCANS; k++) {
                    if(mat[k][i+j].curvature <= 0) {
                        num--;
                        continue;
                    }
                    Eigen::Vector3d pt(mat[k][i+j].x,
                            mat[k][i+j].y,
                            mat[k][i+j].z);
                    center += pt;
                    near_pts.push_back(pt);
                }
            }
            if(num < 25)
                continue;
            center /= num;
            // Covariance matrix
            Eigen::Matrix3d matA1 = Eigen::Matrix3d::Zero();
            for (int j = 0; j < near_pts.size(); j++)
            {
                Eigen::Vector3d zero_mean = near_pts[j] - center;
                matA1 += (zero_mean * zero_mean.transpose());
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matA1);

            vector<int> idsx_edge;
            vector<int> idsy_edge;
            for(int k = 0; k < N_SCANS; k++) {
                double max_s = 0;
                double max_s1 = 0;
                int idx = i;
                for(int j = 0; j < 6; j++) {
                    if(mat[k][i+j].curvature <= 0) {
                        continue;
                    }
                    double g1 = getDepth(mat[k][i+j-4]) + getDepth(mat[k][i+j-3]) +
                            getDepth(mat[k][i+j-2]) + getDepth(mat[k][i+j-1]) - 8*getDepth(mat[k][i+j]) +
                            getDepth(mat[k][i+j+1]) + getDepth(mat[k][i+j+2]) + getDepth(mat[k][i+j+3]) +
                            getDepth(mat[k][i+j+4]);

                    g1 = g1 / (8 * getDepth(mat[k][i+j]) + 1e-3);

                    if(g1 > 0.06) {
                        if(g1 > max_s) {
                            max_s = g1;
                            idx = i+j;
                        }
                    } else if(g1 < -0.06) {
                        if(g1 < max_s1) {
                        }
                    }
                }
                if(max_s != 0) {
                    idsx_edge.push_back(k);
                    idsy_edge.push_back(idx);
                }
            }

            vector<Eigen::Vector3d> near_pts_edge;
            Eigen::Vector3d center_edge(0, 0, 0);
            for(int j = 0; j < idsx_edge.size(); j++) {
                Eigen::Vector3d pt(mat[idsx_edge[j]][idsy_edge[j]].x,
                        mat[idsx_edge[j]][idsy_edge[j]].y,
                        mat[idsx_edge[j]][idsy_edge[j]].z);
                center_edge += pt;
                near_pts_edge.push_back(pt);
            }
            center_edge /= idsx_edge.size();
            // Covariance matrix
            Eigen::Matrix3d matA_edge = Eigen::Matrix3d::Zero();
            for (int j = 0; j < near_pts_edge.size(); j++)
            {
                Eigen::Vector3d zero_mean = near_pts_edge[j] - center_edge;
                matA_edge += (zero_mean * zero_mean.transpose());
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver_edge(matA_edge);

            if(eigen_solver_edge.eigenvalues()[2] > edge_thres * eigen_solver_edge.eigenvalues()[1] && idsx_edge.size() > 3) {
                Eigen::Vector3d unitDirection = eigen_solver_edge.eigenvectors().col(2);
                for(int j = 0; j < idsx_edge.size(); j++) {
                    if(mat[idsx_edge[j]][idsy_edge[j]].curvature <= 0 && mat[idsx_edge[j]][idsy_edge[j]].intensity <= 0)
                        continue;
                    mat[idsx_edge[j]][idsy_edge[j]].normal_x = unitDirection.x();
                    mat[idsx_edge[j]][idsy_edge[j]].normal_y = unitDirection.y();
                    mat[idsx_edge[j]][idsy_edge[j]].normal_z = unitDirection.z();

                    edge_features->points.push_back(mat[idsx_edge[j]][idsy_edge[j]]);
                    mat[idsx_edge[j]][idsy_edge[j]].curvature *= -1;
                }
            }

            if(eigen_solver.eigenvalues()[0] < surf_thres * eigen_solver.eigenvalues()[1]) {
                Eigen::Vector3d unitDirection = eigen_solver.eigenvectors().col(0);
                for(int j = 0; j < 6; j++) {
                    for(int k = 0; k < N_SCANS; k++) {
                        if(mat[k][i+j].curvature <= 0) {
                            continue;
                        }
                        mat[k][i+j].normal_x = unitDirection.x();
                        mat[k][i+j].normal_y = unitDirection.y();
                        mat[k][i+j].normal_z = unitDirection.z();

                        surf_features->points.push_back(mat[k][i+j]);
                        mat[k][i+j].curvature *= -1;
                    }
                }
            }
        }

        sensor_msgs::PointCloud2 surf_features_msg;
        pcl::toROSMsg(*surf_features, surf_features_msg);
        surf_features_msg.header.stamp = cloud_header.stamp;
        surf_features_msg.header.frame_id = frame_id;
        pub_surf.publish(surf_features_msg);

        sensor_msgs::PointCloud2 edge_features_msg;
        pcl::toROSMsg(*edge_features, edge_features_msg);
        edge_features_msg.header.stamp = cloud_header.stamp;
        edge_features_msg.header.frame_id = frame_id;
        pub_edge.publish(edge_features_msg);

        sensor_msgs::PointCloud2 cloud_cutted_msg;
        pcl::toROSMsg(lidar_cloud_cutted, cloud_cutted_msg);
        cloud_cutted_msg.header.stamp = cloud_header.stamp;
        cloud_cutted_msg.header.frame_id = frame_id;
        pub_cutted_cloud.publish(cloud_cutted_msg);

        q_iMU = Eigen::Quaterniond::Identity();
        //t_pre.tic_toc();
        runtime += t_pre.toc();
        //cout<<"pre_num: "<<++pre_num<<endl;
        //cout<<"Preprocessing average run time: "<<runtime / pre_num<<endl;
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "lili_om");

    Preprocessing Pre;

    ROS_INFO("\033[1;32m---->\033[0m Preprocessing Started.");
    ros::spin();
    return 0;
}
