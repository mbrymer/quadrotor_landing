/*
    AER 1810 Quadrotor Landing Project
    Relative Pose Filter Node
*/

#include "relative_pose_filter_node.h"

#include <apriltag_detection.h>

RelativePoseFilterNode::RelativePoseFilterNode(ros::NodeHandle nh) : node_{nh}
{
    InitPubSubCallbacks();
    InitPoseFilter();
    // std::cout << "In the constructor" << std::endl;
}

void RelativePoseFilterNode::InitPubSubCallbacks()
{
    // Load parameters
    node_.param<std::string>("IMU_topic", IMU_topic_, "/drone/imu");
    node_.param<std::string>("apriltag_topic", apriltag_topic_, "/tag_detections");
    node_.param<std::string>("gps_speed_topic", gps_speed_topic_, "/drone/ground_speed");
    node_.param<std::string>("rel_pose_topic", rel_pose_topic_, "/state_estimation/rel_pose_state");
    node_.param<std::string>("rel_pose_report_topic", rel_pose_report_topic_, "/state_estimation/rel_pose_reported");
    node_.param<std::string>("rel_vel_topic", rel_vel_topic_, "/state_estimation/rel_pose_velocity");
    node_.param<std::string>("rel_accel_topic", rel_accel_topic_, "/state_estimation/rel_pose_acceleration");
    node_.param<std::string>("IMU_bias_topic", IMU_bias_topic_, "/state_estimation/IMU_bias");
    node_.param<std::string>("upds_since_correction_topic", pred_length_topic_, "/state_estimation/upds_since_correction");
    node_.param<std::string>("meas_delay_topic", meas_delay_topic_, "/state_estimation/measurement_delay");

    node_.param<std::string>("pose_frame_name", pose_frame_name_, "drone/rel_pose_est");
    node_.param<std::string>("pose_parent_frame_name", pose_parent_frame_name_, "target/tag_link");
    node_.param<std::string>("pose_report_frame_name", pose_report_frame_name_, "drone/rel_pose_report");

    node_.param<double>("update_freq", update_freq_, 100.0);

    // Set publishers, subscribers and timers
    IMU_sub_ = node_.subscribe(IMU_topic_,1, &RelativePoseFilterNode::IMUSubCallback, this);
    apriltag_sub_ = node_.subscribe(apriltag_topic_,1, &RelativePoseFilterNode::AprilTagSubCallback, this);
    gps_speed_sub_ = node_.subscribe(gps_speed_topic_,1, &RelativePoseFilterNode::GPSSubCallback, this);

    rel_pose_pub_ = node_.advertise<geometry_msgs::PoseWithCovarianceStamped>(rel_pose_topic_, 1);
    rel_pose_report_pub_ = node_.advertise<geometry_msgs::PoseStamped>(rel_pose_report_topic_, 1);
    rel_vel_pub_ = node_.advertise<geometry_msgs::Vector3Stamped>(rel_vel_topic_, 1);
    rel_accel_pub_ = node_.advertise<geometry_msgs::Vector3Stamped>(rel_accel_topic_, 1);
    IMU_bias_pub_ = node_.advertise<sensor_msgs::Imu>(IMU_bias_topic_, 1);
    pred_length_pub_ = node_.advertise<geometry_msgs::PointStamped>(pred_length_topic_, 1);
    meas_delay_pub_ = node_.advertise<geometry_msgs::PointStamped>(meas_delay_topic_, 1);

    filter_update_timer_ = node_.createTimer(ros::Duration(1.0 / update_freq_), &RelativePoseFilterNode::FilterUpdateCallback, this);
}

void RelativePoseFilterNode::InitPoseFilter()
{
    // TODO: Clean this up by loading the parameters into a struct and then passing it in, rather than
    // modifying public fields
    double measurement_freq;
    double measurement_delay;
    double measurement_delay_max;
    double dyn_measurement_delay_offset;
    bool limit_apriltag_freq;
    
    node_.param<double>("measurement_freq", measurement_freq, 10.0);
    node_.param<double>("measurement_delay", measurement_delay, 0.010);
    node_.param<double>("measurement_delay_max", measurement_delay_max, 0.200);
    node_.param<double>("dyn_measurement_delay_offset", dyn_measurement_delay_offset, 0.0);
    node_.param<bool>("limit_measurement_freq", limit_apriltag_freq, false);

    relative_pose_filter_.update_freq_ = update_freq_;
    relative_pose_filter_.apriltag_freq_ = measurement_freq;
    relative_pose_filter_.measurement_delay_ = measurement_delay;
    relative_pose_filter_.measurement_delay_max_ = measurement_delay_max;
    relative_pose_filter_.dyn_measurement_delay_offset_ = dyn_measurement_delay_offset;
    relative_pose_filter_.limit_apriltag_freq_ = limit_apriltag_freq;

    node_.param<bool>("est_bias",relative_pose_filter_.est_bias_, true);
    node_.param<bool>("corner_margin_enbl",relative_pose_filter_.corner_margin_enbl_, true);
    node_.param<bool>("direct_orien_method",relative_pose_filter_.direct_orien_method_, false);
    node_.param<bool>("multirate_filter",relative_pose_filter_.multirate_filter_, false);
    node_.param<bool>("dynamic_meas_delay",relative_pose_filter_.dynamic_meas_delay_, false);

    std::vector<double> Q_a_diag;
    std::vector<double> Q_w_diag;
    std::vector<double> Q_ab_diag;
    std::vector<double> Q_wb_diag;

    node_.getParam("Q_a_diag", Q_a_diag);
    node_.getParam("Q_w_diag", Q_w_diag);
    node_.getParam("Q_ab_diag", Q_ab_diag);
    node_.getParam("Q_wb_diag", Q_wb_diag);

    relative_pose_filter_.Q_a_ = Eigen::Vector3d(Q_a_diag.data());
    relative_pose_filter_.Q_w_ = Eigen::Vector3d(Q_w_diag.data());
    relative_pose_filter_.Q_ab_ = Eigen::Vector3d(Q_ab_diag.data());
    relative_pose_filter_.Q_wb_ = Eigen::Vector3d(Q_wb_diag.data());

    std::vector<double> R_r_diag;
    std::vector<double> R_ang_diag;

    node_.getParam("R_r_diag", R_r_diag);
    node_.getParam("R_ang_diag", R_ang_diag);
    relative_pose_filter_.R_r_ = Eigen::Vector3d(R_r_diag.data());
    relative_pose_filter_.R_ang_ = Eigen::Vector3d(R_ang_diag.data());

    node_.param<double>("r_cov_init",relative_pose_filter_.r_cov_init_, 0.1);
    node_.param<double>("v_cov_init",relative_pose_filter_.v_cov_init_, 0.1);
    node_.param<double>("ang_cov_init",relative_pose_filter_.ang_cov_init_, 0.15);
    node_.param<double>("ab_cov_init",relative_pose_filter_.ab_cov_init_, 0.5);
    node_.param<double>("wb_cov_init",relative_pose_filter_.wb_cov_init_, 0.1);

    std::vector<double> ab_static;
    std::vector<double> wb_static;

    node_.getParam("accel_bias_static", ab_static);
    node_.getParam("gyro_bias_static", wb_static);
    relative_pose_filter_.ab_static_ = Eigen::Vector3d(ab_static.data());
    relative_pose_filter_.wb_static_ = Eigen::Vector3d(wb_static.data());

    std::vector<double> r_v_cv;
    std::vector<double> q_vc;

    node_.getParam("r_v_cv",r_v_cv);
    node_.getParam("q_vc",q_vc);
    relative_pose_filter_.r_v_cv_ = Eigen::Vector3d(r_v_cv.data());
    relative_pose_filter_.q_vc_ = Eigen::Quaterniond(q_vc.data());
    quaternion_norm(relative_pose_filter_.q_vc_);

    node_.getParam("camera_width",relative_pose_filter_.camera_width_);
    node_.getParam("camera_height",relative_pose_filter_.camera_height_);

    std::vector<double> camera_K;
    node_.getParam("camera_K", camera_K);
    relative_pose_filter_.camera_K_ = Eigen::Matrix3d(camera_K.data()).transpose();

    node_.getParam("n_tags", relative_pose_filter_.n_tags_);
    node_.getParam("tag_in_view_margin", relative_pose_filter_.tag_in_view_margin_);

    std::vector<double> tag_widths;
    std::vector<double> tag_positions;

    node_.getParam("tag_widths",tag_widths);
    node_.getParam("tag_positions",tag_positions);

    relative_pose_filter_.tag_widths_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tag_widths.data(),tag_widths.size());
    relative_pose_filter_.tag_positions_.resize(3,relative_pose_filter_.n_tags_); // Brute force copy. TODO: Figure out the proper Eigen one liner for this
    for (int i = 0; i<relative_pose_filter_.n_tags_; ++i)
    {
        for (int j = 0; j<3; ++j)
        {
            relative_pose_filter_.tag_positions_(j,i) = tag_positions[3*i+j];
        }
    }

    relative_pose_filter_.InitializeParams();
}

void RelativePoseFilterNode::IMUSubCallback(const sensor_msgs::Imu &imu_msg)
{
    // std::cout << "In the IMU sub callback" << std::endl;
    relative_pose_filter_.mtx_IMU_.lock();
    relative_pose_filter_.IMU_accel_ << imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z;
    relative_pose_filter_.IMU_ang_vel_ << imu_msg.angular_velocity.x,imu_msg.angular_velocity.y,imu_msg.angular_velocity.z;
    relative_pose_filter_.mtx_IMU_.unlock();
}

void RelativePoseFilterNode::GPSSubCallback(const geometry_msgs::TwistStamped &gps_speed_msg) {
    // std::cout << "In the GPS speed sub callback" << std::endl;
    // TODO: Figure out what the correct message type will be on vehicle and update this accordingly
    relative_pose_filter_.mtx_gps_speed_.lock();
    Eigen::Vector2d in_plane_speed{gps_speed_msg.twist.linear.x, gps_speed_msg.twist.linear.y};
    relative_pose_filter_.gps_speed_ = in_plane_speed.norm();
    relative_pose_filter_.mtx_gps_speed_.unlock();
}

void RelativePoseFilterNode::AprilTagSubCallback(const apriltag_ros::AprilTagDetectionArray &apriltag_msg) {
    // std::cout << "In the AprilTag sub callback" << std::endl;
    if(apriltag_msg.detections.size() > 0)
    {
        relative_pose_filter_.mtx_apriltag_.lock();

        Eigen::Vector3d position{apriltag_msg.detections[0].pose.pose.pose.position.x,
            apriltag_msg.detections[0].pose.pose.pose.position.y,apriltag_msg.detections[0].pose.pose.pose.position.z};
        Eigen::Quaterniond orientation{apriltag_msg.detections[0].pose.pose.pose.orientation.w,
                                       apriltag_msg.detections[0].pose.pose.pose.orientation.x,
                                       apriltag_msg.detections[0].pose.pose.pose.orientation.y,
                                       apriltag_msg.detections[0].pose.pose.pose.orientation.z};
        relative_pose_filter_.apriltag_detection_ = AprilTagDetection(position, orientation, apriltag_msg.header.stamp.toSec());

        if(!relative_pose_filter_.state_initialized_)
        {
            relative_pose_filter_.InitializeState(false);
        }

        relative_pose_filter_.mtx_apriltag_.unlock();
    }
}

void RelativePoseFilterNode::FilterUpdateCallback(const ros::TimerEvent &event)
{
    // Update filter
    // std::cout << "In the filter update callback" << std::endl;
    relative_pose_filter_.Update(ros::Time::now().toSec());

    if (relative_pose_filter_.filter_active_)
    {
        // Update published values
        ros::Time curr_time = ros::Time::now();
        std_msgs::Header new_header;
        new_header.stamp = curr_time;
        new_header.frame_id = pose_frame_name_;

        // Relative Pose
        geometry_msgs::PoseWithCovarianceStamped rel_pose_msg;
        rel_pose_msg.header = new_header;
        rel_pose_msg.pose.pose.position.x = relative_pose_filter_.r_nom_(0);
        rel_pose_msg.pose.pose.position.y = relative_pose_filter_.r_nom_(1);
        rel_pose_msg.pose.pose.position.z = relative_pose_filter_.r_nom_(2);
        rel_pose_msg.pose.pose.orientation.w = relative_pose_filter_.q_nom_.w();
        rel_pose_msg.pose.pose.orientation.x = relative_pose_filter_.q_nom_.x();
        rel_pose_msg.pose.pose.orientation.y = relative_pose_filter_.q_nom_.y();
        rel_pose_msg.pose.pose.orientation.z = relative_pose_filter_.q_nom_.z();

        Eigen::MatrixXd pose_cov(6,6);
        pose_cov << relative_pose_filter_.cov_pert_(seq(0,2),seq(0,2)), relative_pose_filter_.cov_pert_(seq(0,2),seq(6,8)),
        relative_pose_filter_.cov_pert_(seq(6,8),seq(0,2)), relative_pose_filter_.cov_pert_(seq(6,8),seq(6,8));
        Eigen::VectorXd pose_cov_flatten = pose_cov.reshaped<Eigen::RowMajor>();
        for (int i = 0; i < pose_cov_flatten.size(); i++)
        {
            rel_pose_msg.pose.covariance[i] = pose_cov_flatten(i);
        }

        // IMU Bias
        sensor_msgs::Imu IMU_bias_msg;
        IMU_bias_msg.header = new_header;
        IMU_bias_msg.linear_acceleration.x = relative_pose_filter_.ab_nom_(0)+relative_pose_filter_.ab_static_(0);
        IMU_bias_msg.linear_acceleration.y = relative_pose_filter_.ab_nom_(1)+relative_pose_filter_.ab_static_(1);
        IMU_bias_msg.linear_acceleration.z = relative_pose_filter_.ab_nom_(2)+relative_pose_filter_.ab_static_(2);
        IMU_bias_msg.angular_velocity.x = relative_pose_filter_.wb_nom_(0)+relative_pose_filter_.wb_static_(0);
        IMU_bias_msg.angular_velocity.y = relative_pose_filter_.wb_nom_(1)+relative_pose_filter_.wb_static_(1);
        IMU_bias_msg.angular_velocity.z = relative_pose_filter_.wb_nom_(2)+relative_pose_filter_.wb_static_(2);

        // Relative velocity/acceleration
        geometry_msgs::Vector3Stamped rel_vel_msg;
        rel_vel_msg.header = new_header;
        rel_vel_msg.vector.x = relative_pose_filter_.v_nom_(0);
        rel_vel_msg.vector.y = relative_pose_filter_.v_nom_(1);
        rel_vel_msg.vector.z = relative_pose_filter_.v_nom_(2);

        geometry_msgs::Vector3Stamped rel_accel_msg;
        rel_accel_msg.header = new_header;
        rel_accel_msg.vector.x = relative_pose_filter_.accel_rel_(0);
        rel_accel_msg.vector.y = relative_pose_filter_.accel_rel_(1);
        rel_accel_msg.vector.z = relative_pose_filter_.accel_rel_(2);

        // Prediction length
        geometry_msgs::PointStamped pred_length_msg;
        pred_length_msg.header = new_header;
        pred_length_msg.point.x = relative_pose_filter_.upds_since_apriltag_correction_;

        // Publish messages
        rel_pose_pub_.publish(rel_pose_msg);
        IMU_bias_pub_.publish(IMU_bias_msg);
        rel_vel_pub_.publish(rel_vel_msg);
        rel_accel_pub_.publish(rel_accel_msg);
        pred_length_pub_.publish(pred_length_msg);

        // TF Poses
        tf::Transform rel_pose_tf;
        rel_pose_tf.setOrigin(tf::Vector3(relative_pose_filter_.r_nom_(0),
                                            relative_pose_filter_.r_nom_(1), 
                                            relative_pose_filter_.r_nom_(2)));
        rel_pose_tf.setRotation(tf::Quaternion(relative_pose_filter_.q_nom_.x(),relative_pose_filter_.q_nom_.y(),
                                                relative_pose_filter_.q_nom_.z(),relative_pose_filter_.q_nom_.w()));
        tf_broadcast_.sendTransform(tf::StampedTransform(rel_pose_tf, curr_time, pose_parent_frame_name_, pose_frame_name_));

        if (relative_pose_filter_.r_t_vt_obs_.has_value() && relative_pose_filter_.q_tv_obs_.has_value()) {
            geometry_msgs::PoseStamped rel_pose_report_msg;
            rel_pose_report_msg.header.stamp = curr_time;
            rel_pose_report_msg.header.frame_id = pose_report_frame_name_;
            rel_pose_report_msg.pose.position.x = relative_pose_filter_.r_t_vt_obs_.value()(0);
            rel_pose_report_msg.pose.position.y = relative_pose_filter_.r_t_vt_obs_.value()(1);
            rel_pose_report_msg.pose.position.z = relative_pose_filter_.r_t_vt_obs_.value()(2);
            rel_pose_report_msg.pose.orientation.x = relative_pose_filter_.q_tv_obs_.value().x();
            rel_pose_report_msg.pose.orientation.y = relative_pose_filter_.q_tv_obs_.value().y();
            rel_pose_report_msg.pose.orientation.z = relative_pose_filter_.q_tv_obs_.value().z();
            rel_pose_report_msg.pose.orientation.w = relative_pose_filter_.q_tv_obs_.value().w();

            rel_pose_report_pub_.publish(rel_pose_report_msg);
            tf::Transform rel_pose_report_tf;
            rel_pose_report_tf.setOrigin(tf::Vector3(relative_pose_filter_.r_t_vt_obs_.value()(0),
                                                    relative_pose_filter_.r_t_vt_obs_.value()(1),
                                                    relative_pose_filter_.r_t_vt_obs_.value()(2)));
            rel_pose_report_tf.setRotation(tf::Quaternion(relative_pose_filter_.q_tv_obs_.value().x() ,relative_pose_filter_.q_tv_obs_.value().y(),
                                                relative_pose_filter_.q_tv_obs_.value().z(), relative_pose_filter_.q_tv_obs_.value().w()));
            tf_broadcast_.sendTransform(tf::StampedTransform(rel_pose_report_tf, curr_time, pose_parent_frame_name_,
                                                            pose_report_frame_name_));
            
            relative_pose_filter_.r_t_vt_obs_.reset();
            relative_pose_filter_.q_tv_obs_.reset();
        }
        if (relative_pose_filter_.performed_apriltag_correction_)
        {
            geometry_msgs::PointStamped meas_delay_msg;
            meas_delay_msg.header = new_header;
            meas_delay_msg.point.x = relative_pose_filter_.measurement_delay_curr_;
            meas_delay_pub_.publish(meas_delay_msg);
        }
    }

}
