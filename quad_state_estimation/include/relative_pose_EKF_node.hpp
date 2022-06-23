/*
    AER 1810 Quadrotor Landing Project
    Relative Pose EKF Node
*/

#include "relative_pose_EKF.hpp"

#include "ros/ros.h"
#include "tf/transform_broadcaster.h"

#include "std_msgs/Header.h"
#include "sensor_msgs/Imu.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "geometry_msgs/PointStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "apriltag_ros/AprilTagDetectionArray.h"

class RelativePoseEKFNode
{
    public:
        RelativePoseEKFNode(ros::NodeHandle nh);

    private:
        // Methods
        void IMUSubCallback(const sensor_msgs::Imu &imu_msg);
        void AprilTagSubCallback(const apriltag_ros::AprilTagDetectionArray &apriltag_msg);
        
        void FilterUpdateCallback(const ros::TimerEvent &event);

        // Properties
        RelativePoseEKF rel_pose_ekf;

        std::string IMU_topic;
        std::string apriltag_topic;

        std::string rel_pose_topic;
        std::string rel_pose_report_topic;
        std::string rel_vel_topic;
        std::string rel_accel_topic;
        std::string IMU_bias_topic;
        std::string pred_length_topic;

        std::string pose_frame_name;
        std::string pose_parent_frame_name;
        std::string pose_report_frame_name;

        ros::NodeHandle node;

        ros::Subscriber IMU_sub;
        ros::Subscriber apriltag_sub;

        ros::Publisher rel_pose_pub;
        ros::Publisher rel_pose_report_pub;
        ros::Publisher rel_vel_pub;
        ros::Publisher rel_accel_pub;
        ros::Publisher IMU_bias_pub;
        ros::Publisher pred_length_pub;

        ros::Timer filter_update_timer;

        tf::TransformBroadcaster tf_broadcast;

        double update_freq;

};