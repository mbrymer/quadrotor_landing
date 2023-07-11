/*
    AER 1810 Quadrotor Landing Project
    Relative Pose Filter Node
*/

#include "relative_pose_filter.h"

#include "ros/ros.h"
#include "tf/transform_broadcaster.h"

#include "std_msgs/Header.h"
#include "sensor_msgs/Imu.h"
#include "geometry_msgs/Vector3Stamped.h"
#include "geometry_msgs/PointStamped.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "apriltag_ros/AprilTagDetectionArray.h"

class RelativePoseFilterNode
{
    public:
        RelativePoseFilterNode(ros::NodeHandle nh);

    private:
        void InitPubSubCallbacks();
        void InitPoseFilter();
        
        void IMUSubCallback(const sensor_msgs::Imu &imu_msg);
        void AprilTagSubCallback(const apriltag_ros::AprilTagDetectionArray &apriltag_msg);
        void GPSSubCallback();

        void FilterUpdateCallback(const ros::TimerEvent &event);

        RelativePoseFilter relative_pose_filter_;

        std::string IMU_topic_;
        std::string apriltag_topic_;

        std::string rel_pose_topic_;
        std::string rel_pose_report_topic_;
        std::string rel_vel_topic_;
        std::string rel_accel_topic_;
        std::string IMU_bias_topic_;
        std::string pred_length_topic_;
        std::string meas_delay_topic_;

        std::string pose_frame_name_;
        std::string pose_parent_frame_name_;
        std::string pose_report_frame_name_;

        ros::NodeHandle node_;

        ros::Subscriber IMU_sub_;
        ros::Subscriber apriltag_sub_;

        ros::Publisher rel_pose_pub_;
        ros::Publisher rel_pose_report_pub_;
        ros::Publisher rel_vel_pub_;
        ros::Publisher rel_accel_pub_;
        ros::Publisher IMU_bias_pub_;
        ros::Publisher pred_length_pub_;
        ros::Publisher meas_delay_pub_;

        ros::Timer filter_update_timer_;

        tf::TransformBroadcaster tf_broadcast_;

        double update_freq_;

};