/*
    AER 1810 Quadrotor Landing Project
    Relative Pose EKF Node
*/

#include "relative_pose_EKF_node.hpp"

RelativePoseEKFNode::RelativePoseEKFNode(ros::NodeHandle nh) : node(nh)
{
    // Load parameters
    node.param<std::string>("IMU_topic",IMU_topic,"/drone/imu");
    node.param<std::string>("apriltag_topic",apriltag_topic,"/tag_detections");
    node.param<std::string>("rel_pose_topic",rel_pose_topic,"/state_estimation/rel_pose_state");
    node.param<std::string>("rel_pose_report_topic",rel_pose_report_topic,"/state_estimation/rel_pose_reported");
    node.param<std::string>("rel_vel_topic",rel_vel_topic,"/state_estimation/rel_pose_velocity");
    node.param<std::string>("rel_accel_topic",rel_accel_topic,"/state_estimation/rel_pose_acceleration");
    node.param<std::string>("IMU_bias_topic",IMU_bias_topic,"/state_estimation/IMU_bias");
    node.param<std::string>("upds_since_correction_topic",pred_length_topic,"/state_estimation/upds_since_correction");
    node.param<std::string>("meas_delay_topic",meas_delay_topic,"/state_estimation/measurement_delay");

    node.param<std::string>("pose_frame_name",pose_frame_name,"drone/rel_pose_est");
    node.param<std::string>("pose_parent_frame_name",pose_parent_frame_name,"target/tag_link");
    node.param<std::string>("pose_report_frame_name",pose_report_frame_name,"drone/rel_pose_report");

    double measurement_freq;
    double measurement_delay;
    double measurement_delay_max;
    double dyn_measurement_delay_offset;
    bool limit_measurement_freq;

    node.param<double>("update_freq",update_freq,100.0);
    node.param<double>("measurement_freq",measurement_freq,10.0);
    node.param<double>("measurement_delay",measurement_delay,0.010);
    node.param<double>("measurement_delay_max",measurement_delay_max,0.200);
    node.param<double>("dyn_measurement_delay_offset",dyn_measurement_delay_offset,0.0);
    node.param<bool>("limit_measurement_freq",limit_measurement_freq,false);

    // Set publishers, subscribers and timers
    IMU_sub = node.subscribe(IMU_topic,1, &RelativePoseEKFNode::IMUSubCallback,this);
    apriltag_sub = node.subscribe(apriltag_topic,1, &RelativePoseEKFNode::AprilTagSubCallback,this);

    rel_pose_pub = node.advertise<geometry_msgs::PoseWithCovarianceStamped>(rel_pose_topic,1);
    rel_pose_report_pub = node.advertise<geometry_msgs::PoseStamped>(rel_pose_report_topic,1);
    rel_vel_pub = node.advertise<geometry_msgs::Vector3Stamped>(rel_vel_topic,1);
    rel_accel_pub = node.advertise<geometry_msgs::Vector3Stamped>(rel_accel_topic,1);
    IMU_bias_pub = node.advertise<sensor_msgs::Imu>(IMU_bias_topic,1);
    pred_length_pub = node.advertise<geometry_msgs::PointStamped>(pred_length_topic,1);
    meas_delay_pub = node.advertise<geometry_msgs::PointStamped>(meas_delay_topic,1);

    filter_update_timer = node.createTimer(ros::Duration(1.0/update_freq),&RelativePoseEKFNode::FilterUpdateCallback,this);

    // Initialize EKF
    rel_pose_ekf.update_freq = update_freq;
    rel_pose_ekf.measurement_freq = measurement_freq;
    rel_pose_ekf.measurement_delay = measurement_delay;
    rel_pose_ekf.measurement_delay_max = measurement_delay_max;
    rel_pose_ekf.dyn_measurement_delay_offset = dyn_measurement_delay_offset;
    rel_pose_ekf.limit_measurement_freq = limit_measurement_freq;

    node.param<bool>("est_bias",rel_pose_ekf.est_bias,true);
    node.param<bool>("corner_margin_enbl",rel_pose_ekf.corner_margin_enbl,true);
    node.param<bool>("direct_orien_method",rel_pose_ekf.direct_orien_method,false);
    node.param<bool>("multirate_ekf",rel_pose_ekf.multirate_ekf,false);
    node.param<bool>("dynamic_meas_delay",rel_pose_ekf.dynamic_meas_delay,false);

    std::vector<double> Q_a_diag;
    std::vector<double> Q_w_diag;
    std::vector<double> Q_ab_diag;
    std::vector<double> Q_wb_diag;

    node.getParam("Q_a_diag",Q_a_diag);
    node.getParam("Q_w_diag",Q_w_diag);
    node.getParam("Q_ab_diag",Q_ab_diag);
    node.getParam("Q_wb_diag",Q_wb_diag);

    rel_pose_ekf.Q_a = Eigen::Vector3d(Q_a_diag.data());
    rel_pose_ekf.Q_w = Eigen::Vector3d(Q_w_diag.data());
    rel_pose_ekf.Q_ab = Eigen::Vector3d(Q_ab_diag.data());
    rel_pose_ekf.Q_wb = Eigen::Vector3d(Q_wb_diag.data());

    std::vector<double> R_r_diag;
    std::vector<double> R_ang_diag;

    node.getParam("R_r_diag",R_r_diag);
    node.getParam("R_ang_diag",R_ang_diag);
    rel_pose_ekf.R_r = Eigen::Vector3d(R_r_diag.data());
    rel_pose_ekf.R_ang = Eigen::Vector3d(R_ang_diag.data());

    node.param<double>("r_cov_init",rel_pose_ekf.r_cov_init,0.1);
    node.param<double>("v_cov_init",rel_pose_ekf.v_cov_init,0.1);
    node.param<double>("ang_cov_init",rel_pose_ekf.ang_cov_init,0.15);
    node.param<double>("ab_cov_init",rel_pose_ekf.ab_cov_init,0.5);
    node.param<double>("wb_cov_init",rel_pose_ekf.wb_cov_init,0.1);

    std::vector<double> ab_static;
    std::vector<double> wb_static;

    node.getParam("accel_bias_static",ab_static);
    node.getParam("gyro_bias_static",wb_static);
    rel_pose_ekf.ab_static = Eigen::Vector3d(ab_static.data());
    rel_pose_ekf.wb_static = Eigen::Vector3d(wb_static.data());

    std::vector<double> r_v_cv;
    std::vector<double> q_vc;

    node.getParam("r_v_cv",r_v_cv);
    node.getParam("q_vc",q_vc);
    rel_pose_ekf.r_v_cv = Eigen::Vector3d(r_v_cv.data());
    rel_pose_ekf.q_vc = Eigen::Quaterniond(q_vc.data());
    quaternion_norm(rel_pose_ekf.q_vc);

    node.getParam("camera_width",rel_pose_ekf.camera_width);
    node.getParam("camera_height",rel_pose_ekf.camera_height);

    std::vector<double> camera_K;
    node.getParam("camera_K",camera_K);
    rel_pose_ekf.camera_K = Eigen::Matrix3d(camera_K.data()).transpose();

    node.getParam("n_tags",rel_pose_ekf.n_tags);
    node.getParam("tag_in_view_margin",rel_pose_ekf.tag_in_view_margin);

    std::vector<double> tag_widths;
    std::vector<double> tag_positions;

    node.getParam("tag_widths",tag_widths);
    node.getParam("tag_positions",tag_positions);

    rel_pose_ekf.tag_widths = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(tag_widths.data(),tag_widths.size());
    rel_pose_ekf.tag_positions.resize(3,rel_pose_ekf.n_tags); // Brute force copy. TODO: Figure out the proper Eigen one liner for this
    for (int i = 0; i<rel_pose_ekf.n_tags; ++i)
    {
        for (int j = 0; j<3; ++j)
        {
            rel_pose_ekf.tag_positions(j,i) = tag_positions[3*i+j];
        }
    }

    rel_pose_ekf.initialize_params();

    // std::cout << "In the constructor" << std::endl;

}

void RelativePoseEKFNode::IMUSubCallback(const sensor_msgs::Imu &imu_msg)
{
    // std::cout << "In the IMU sub callback" << std::endl;
    rel_pose_ekf.mtx_IMU.lock();
    rel_pose_ekf.IMU_accel << imu_msg.linear_acceleration.x,imu_msg.linear_acceleration.y,imu_msg.linear_acceleration.z;
    rel_pose_ekf.IMU_ang_vel << imu_msg.angular_velocity.x,imu_msg.angular_velocity.y,imu_msg.angular_velocity.z;
    rel_pose_ekf.mtx_IMU.unlock();
}

void RelativePoseEKFNode::AprilTagSubCallback(const apriltag_ros::AprilTagDetectionArray &apriltag_msg)
{
    // std::cout << "In the AprilTag sub callback" << std::endl;
    if(apriltag_msg.detections.size()>0)
    {
        rel_pose_ekf.mtx_apriltag.lock();

        rel_pose_ekf.apriltag_pos << apriltag_msg.detections[0].pose.pose.pose.position.x,
            apriltag_msg.detections[0].pose.pose.pose.position.y,apriltag_msg.detections[0].pose.pose.pose.position.z;
        rel_pose_ekf.apriltag_orien.w() = apriltag_msg.detections[0].pose.pose.pose.orientation.w;
        rel_pose_ekf.apriltag_orien.x() = apriltag_msg.detections[0].pose.pose.pose.orientation.x;
        rel_pose_ekf.apriltag_orien.y() = apriltag_msg.detections[0].pose.pose.pose.orientation.y;
        rel_pose_ekf.apriltag_orien.z() = apriltag_msg.detections[0].pose.pose.pose.orientation.z;
        rel_pose_ekf.apriltag_time = apriltag_msg.header.stamp.toSec();
        rel_pose_ekf.measurement_ready = true;

        if(!rel_pose_ekf.state_initialized)
        {
            rel_pose_ekf.initialize_state(false);
        }

        rel_pose_ekf.mtx_apriltag.unlock();
    }
}

void RelativePoseEKFNode::FilterUpdateCallback(const ros::TimerEvent &event)
{
    // Update filter
    // std::cout << "In the filter update callback" << std::endl;
    rel_pose_ekf.filter_update(ros::Time::now().toSec());

    if (rel_pose_ekf.filter_active)
    {
        // Update published values
        ros::Time curr_time = ros::Time::now();
        std_msgs::Header new_header;
        new_header.stamp = curr_time;
        new_header.frame_id = pose_frame_name;

        // Relative Pose
        geometry_msgs::PoseWithCovarianceStamped rel_pose_msg;
        rel_pose_msg.header = new_header;
        rel_pose_msg.pose.pose.position.x = rel_pose_ekf.r_nom(0);
        rel_pose_msg.pose.pose.position.y = rel_pose_ekf.r_nom(1);
        rel_pose_msg.pose.pose.position.z = rel_pose_ekf.r_nom(2);
        rel_pose_msg.pose.pose.orientation.w = rel_pose_ekf.q_nom.w();
        rel_pose_msg.pose.pose.orientation.x = rel_pose_ekf.q_nom.x();
        rel_pose_msg.pose.pose.orientation.y = rel_pose_ekf.q_nom.y();
        rel_pose_msg.pose.pose.orientation.z = rel_pose_ekf.q_nom.z();

        Eigen::MatrixXd pose_cov(6,6);
        pose_cov << rel_pose_ekf.cov_pert(seq(0,2),seq(0,2)),rel_pose_ekf.cov_pert(seq(0,2),seq(6,8)),
        rel_pose_ekf.cov_pert(seq(6,8),seq(0,2)),rel_pose_ekf.cov_pert(seq(6,8),seq(6,8));
        Eigen::VectorXd pose_cov_flatten = pose_cov.reshaped<Eigen::RowMajor>();
        for (int i = 0; i < pose_cov_flatten.size(); i++)
        {
            rel_pose_msg.pose.covariance[i] = pose_cov_flatten(i);
        }

        // IMU Bias
        sensor_msgs::Imu IMU_bias_msg;
        IMU_bias_msg.header = new_header;
        IMU_bias_msg.linear_acceleration.x = rel_pose_ekf.ab_nom(0)+rel_pose_ekf.ab_static(0);
        IMU_bias_msg.linear_acceleration.y = rel_pose_ekf.ab_nom(1)+rel_pose_ekf.ab_static(1);
        IMU_bias_msg.linear_acceleration.z = rel_pose_ekf.ab_nom(2)+rel_pose_ekf.ab_static(2);
        IMU_bias_msg.angular_velocity.x = rel_pose_ekf.wb_nom(0)+rel_pose_ekf.wb_static(0);
        IMU_bias_msg.angular_velocity.y = rel_pose_ekf.wb_nom(1)+rel_pose_ekf.wb_static(1);
        IMU_bias_msg.angular_velocity.z = rel_pose_ekf.wb_nom(2)+rel_pose_ekf.wb_static(2);

        // Relative velocity/acceleration
        geometry_msgs::Vector3Stamped rel_vel_msg;
        rel_vel_msg.header = new_header;
        rel_vel_msg.vector.x = rel_pose_ekf.v_nom(0);
        rel_vel_msg.vector.y = rel_pose_ekf.v_nom(1);
        rel_vel_msg.vector.z = rel_pose_ekf.v_nom(2);

        geometry_msgs::Vector3Stamped rel_accel_msg;
        rel_accel_msg.header = new_header;
        rel_accel_msg.vector.x = rel_pose_ekf.accel_rel(0);
        rel_accel_msg.vector.y = rel_pose_ekf.accel_rel(1);
        rel_accel_msg.vector.z = rel_pose_ekf.accel_rel(2);

        // Prediction length
        geometry_msgs::PointStamped pred_length_msg;
        pred_length_msg.header = new_header;
        pred_length_msg.point.x = rel_pose_ekf.upds_since_correction;

        // Publish messages
        rel_pose_pub.publish(rel_pose_msg);
        IMU_bias_pub.publish(IMU_bias_msg);
        rel_vel_pub.publish(rel_vel_msg);
        rel_accel_pub.publish(rel_accel_msg);
        pred_length_pub.publish(pred_length_msg);

        // TF Poses
        tf::Transform rel_pose_tf;
        rel_pose_tf.setOrigin(tf::Vector3(rel_pose_ekf.r_nom(0),rel_pose_ekf.r_nom(1),rel_pose_ekf.r_nom(2)));
        rel_pose_tf.setRotation(tf::Quaternion(rel_pose_ekf.q_nom.x(),rel_pose_ekf.q_nom.y(),
                                                rel_pose_ekf.q_nom.z(),rel_pose_ekf.q_nom.w()));
        tf_broadcast.sendTransform(tf::StampedTransform(rel_pose_tf,curr_time,pose_parent_frame_name,pose_frame_name));

        if (rel_pose_ekf.performed_correction)
        {
            geometry_msgs::PoseStamped rel_pose_report_msg;
            rel_pose_report_msg.header.stamp = curr_time;
            rel_pose_report_msg.header.frame_id = pose_report_frame_name;
            rel_pose_report_msg.pose.position.x = rel_pose_ekf.r_t_vt_obs(0);
            rel_pose_report_msg.pose.position.y = rel_pose_ekf.r_t_vt_obs(1);
            rel_pose_report_msg.pose.position.z = rel_pose_ekf.r_t_vt_obs(2);
            rel_pose_report_msg.pose.orientation.x = rel_pose_ekf.q_tv_obs.x();
            rel_pose_report_msg.pose.orientation.y = rel_pose_ekf.q_tv_obs.y();
            rel_pose_report_msg.pose.orientation.z = rel_pose_ekf.q_tv_obs.z();
            rel_pose_report_msg.pose.orientation.w = rel_pose_ekf.q_tv_obs.w();

            geometry_msgs::PointStamped meas_delay_msg;
            meas_delay_msg.header = new_header;
            meas_delay_msg.point.x = rel_pose_ekf.measurement_delay_curr;

            rel_pose_report_pub.publish(rel_pose_report_msg);
            meas_delay_pub.publish(meas_delay_msg);

            tf::Transform rel_pose_report_tf;
            rel_pose_report_tf.setOrigin(tf::Vector3(rel_pose_ekf.r_t_vt_obs(0),rel_pose_ekf.r_t_vt_obs(1),rel_pose_ekf.r_t_vt_obs(2)));
            rel_pose_report_tf.setRotation(tf::Quaternion(rel_pose_ekf.q_tv_obs.x(),rel_pose_ekf.q_tv_obs.y(),
                                                rel_pose_ekf.q_tv_obs.z(),rel_pose_ekf.q_tv_obs.w()));
            tf_broadcast.sendTransform(tf::StampedTransform(rel_pose_report_tf,curr_time,pose_parent_frame_name,pose_report_frame_name));

        }
    }

}
