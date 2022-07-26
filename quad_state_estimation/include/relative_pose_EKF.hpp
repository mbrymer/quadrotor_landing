/*
    AER 1810 Quadrotor Landing Project
    Relative Pose EKF Class
*/

#include <mutex>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <math.h>
#include <quaternion_helper.hpp>

using Eigen::seq;

class RelativePoseEKF
{
    public:
        RelativePoseEKF();

        // Perform periodic EKF filter update
        void filter_update();
        // Initialize state to last received AprilTag relative pose
        void initialize_state(bool reinit_bias);
        // Compute convenience values derived from parameters
        void initialize_params();

        // Mutexes
        std::mutex mtx_IMU;
        std::mutex mtx_apriltag;
        std::mutex mtx_state;
        
        // Storage
        // Inputs
        Eigen::VectorXd IMU_accel;
        Eigen::VectorXd IMU_ang_vel;
        Eigen::VectorXd apriltag_pos;
        Eigen::Quaterniond apriltag_orien;

        // State
        // x = [r_x/y/z, v_x/y/z, theta_x/y/z, a_bias_x/y/z, w_bias_x/y/z ]
        Eigen::VectorXd r_nom;
        Eigen::VectorXd v_nom;
        Eigen::VectorXd accel_rel;
        Eigen::Quaterniond q_nom;
        Eigen::VectorXd ab_nom;
        Eigen::VectorXd wb_nom;
        Eigen::MatrixXd cov_pert;

        Eigen::VectorXd r_t_vt_obs;
        Eigen::Quaterniond q_tv_obs;

        // Filter Parameters
        double update_freq;
        double dT_nom;
        double measurement_freq;
        double t_last_update;
        
        bool est_bias;
        bool limit_measurement_freq;
        bool corner_margin_enbl;

        int upd_per_meas;
        int num_states;

        // Process and Measurement Noises
        double r_cov_init;
        double v_cov_init;
        double ang_cov_init;
        double ab_cov_init;
        double wb_cov_init;

        Eigen::MatrixXd cov_init;

        Eigen::VectorXd Q_a;
        Eigen::VectorXd Q_w;
        Eigen::VectorXd Q_ab;
        Eigen::VectorXd Q_wb;
        Eigen::MatrixXd Q;

        Eigen::VectorXd R_r;
        Eigen::VectorXd R_ang;
        Eigen::MatrixXd R;

        // Camera calibration
        Eigen::VectorXd r_v_cv;
        Eigen::Quaterniond q_vc;
        Eigen::MatrixXd C_vc;
        Eigen::Affine3d T_vc;

        Eigen::Matrix3d camera_K;
        int camera_width;
        int camera_height;
        double tag_width;
        double tag_in_view_margin;
        Eigen::Matrix4d tag_corners;

        // Counters/flags
        bool state_initialized;
        bool measurement_ready;
        bool performed_correction;
        bool filter_active;
        int upds_since_correction;

        // Tolerances and constants
        double small_ang_tol;
        Eigen::Vector3d g;


};