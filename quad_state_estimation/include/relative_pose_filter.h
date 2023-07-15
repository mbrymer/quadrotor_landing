/*
    AER 1810 Quadrotor Landing Project
    Relative Pose Filter Class
*/

#pragma once

#include <mutex>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <math.h>
#include <algorithm>
#include <vector>
#include <optional>
#include <quaternion_helper.h>
#include <mahony_filter.h>
#include <relative_pose_utils.h>
#include <apriltag_detection.h>

using Eigen::seq;

class RelativePoseFilter
{
    public:
        RelativePoseFilter();

        // Perform periodic pose filter update
        void Update(double t_curr);
        // Initialize state to last received AprilTag relative pose
        void InitializeState(bool reinit_bias);
        // Compute convenience values derived from parameters
        void InitializeParams();

        // Mutexes
        std::mutex mtx_IMU_;
        std::mutex mtx_apriltag_;
        std::mutex mtx_gps_speed_;
        std::mutex mtx_state_;

        // Storage
        // Inputs
        Eigen::Vector3d IMU_accel_;
        Eigen::Vector3d IMU_ang_vel_;
        std::optional<AprilTagDetection> apriltag_detection_;
        std::optional<double> gps_speed_;
        double apriltag_time_;

        // State
        // x = [r_x/y/z, v_x/y/z, theta_x/y/z, a_bias_x/y/z, w_bias_x/y/z ]
        Eigen::Vector3d r_nom_;
        Eigen::Vector3d v_nom_;
        Eigen::Vector3d accel_rel_;
        Eigen::Quaterniond q_nom_;
        Eigen::Vector3d ab_nom_;
        Eigen::Vector3d wb_nom_;
        Eigen::MatrixXd cov_pert_;

        Eigen::Vector3d ab_static_;
        Eigen::Vector3d wb_static_;

        std::optional<Eigen::Vector3d> r_t_vt_obs_;
        std::optional<Eigen::Quaterniond> q_tv_obs_;

        // Multirate EKF
        std::vector<Eigen::VectorXd> x_hist_;
        std::vector<Eigen::VectorXd> u_hist_;
        std::vector<Eigen::MatrixXd> P_hist_;

        // Filter Parameters
        double update_freq_;
        double dT_nom_;
        double apriltag_freq_;
        double measurement_delay_;
        double measurement_delay_max_;
        double dyn_measurement_delay_offset_;
        double t_last_update_;

        bool est_bias_;
        bool limit_apriltag_freq_;
        bool corner_margin_enbl_;
        bool direct_orien_method_;
        bool multirate_filter_;
        bool dynamic_meas_delay_;

        int upd_per_apriltag_meas_;
        int num_states_;
        int measurement_step_delay_;

        double measurement_delay_curr_;

        // Process and Measurement Noises
        double r_cov_init_;
        double v_cov_init_;
        double ang_cov_init_;
        double ab_cov_init_;
        double wb_cov_init_;

        Eigen::MatrixXd cov_init_;

        Eigen::Vector3d Q_a_;
        Eigen::Vector3d Q_w_;
        Eigen::Vector3d Q_ab_;
        Eigen::Vector3d Q_wb_;
        Eigen::MatrixXd Q_;

        Eigen::Vector3d R_r_;
        Eigen::Vector3d R_ang_;
        Eigen::Vector2d R_mahony_;
        Eigen::Vector2d R_gps_;
        Eigen::MatrixXd R_apriltag_;

        Eigen::MatrixXd L_Q_;
        Eigen::MatrixXd L_R_apriltag_;
        Eigen::MatrixXd L_R_mahony_;
        Eigen::MatrixXd L_R_gps_;

        // Camera calibration
        Eigen::VectorXd r_v_cv_;
        Eigen::Quaterniond q_vc_;
        Eigen::MatrixXd C_vc_;
        Eigen::Affine3d T_vc_;

        Eigen::Matrix3d camera_K_;
        int camera_width_;
        int camera_height_;

        // Target Configuration
        int n_tags_;
        double tag_in_view_margin_;

        Eigen::VectorXd tag_widths_;
        Eigen::MatrixXd tag_positions_;

        // Counters/flags
        bool state_initialized_;
        bool performed_apriltag_correction_;
        bool filter_active_;
        int upds_since_apriltag_correction_;

        // Tolerances and constants
        Eigen::Vector3d g_;

    private:
        std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::Vector3d> PredictionStep(const Eigen::VectorXd& x_km1, const Eigen::MatrixXd& P_km1, const Eigen::VectorXd& u);
        std::tuple<Eigen::VectorXd, Eigen::MatrixXd> CorrectionStep(const Eigen::VectorXd& x_check, const Eigen::VectorXd& mu_check,
                                                            const Eigen::MatrixXd& P_check, double roll_mahony, double pitch_mahony,
                                                            std::optional<double> v_gps, const std::optional<AprilTagDetection>& apriltag_detection);

        std::tuple<Eigen::Vector3d, Eigen::Quaterniond> DetectionToPose(const AprilTagDetection& apriltag_detection);

        MahonyFilter mahony_filter_;

};
