/*
    AER 1810 Quadrotor Landing Project
    Relative Pose EKF Class
*/

#include "relative_pose_EKF.hpp"

RelativePoseEKF::RelativePoseEKF()
{   
    // Default values for parameters
    IMU_accel = Eigen::VectorXd::Zero(3);
    IMU_ang_vel = Eigen::VectorXd::Zero(3);
    apriltag_pos = Eigen::VectorXd::Zero(3);
    apriltag_orien = Eigen::Quaterniond(1,0,0,0);

    // State
    r_nom = Eigen::VectorXd::Zero(3);
    v_nom = Eigen::VectorXd::Zero(3);
    accel_rel = Eigen::VectorXd::Zero(3);
    q_nom = Eigen::Quaterniond::Identity();
    ab_nom = Eigen::VectorXd::Zero(3);
    wb_nom = Eigen::VectorXd::Zero(3);
    r_t_vt_obs = Eigen::VectorXd::Zero(3);
    q_tv_obs = Eigen::Quaterniond::Identity();

    // Filter Parameters
    update_freq = 100;
    measurement_freq = 10;
    t_last_update = 0;
    est_bias = true;
    limit_measurement_freq = false;
    corner_margin_enbl = true;
    num_states = 15;

    // Process and Measurement Noises
    Q_a = 0.005*Eigen::VectorXd::Ones(3);
    Q_w = 0.0005*Eigen::VectorXd::Ones(3);
    Q_ab = 5E-5*Eigen::VectorXd::Ones(3);
    Q_wb = 5E-6*Eigen::VectorXd::Ones(3);
    
    R_r = Eigen::VectorXd::Zero(3);
    R_ang = Eigen::VectorXd::Zero(3);
    R_r << 0.005, 0.005, 0.015;
    R_ang << 0.0025, 0.0025, 0.025;

    // Camera calibration
    r_v_cv = Eigen::VectorXd::Zero(3);
    r_v_cv << 0,0,-0.073;
    q_vc = Eigen::Quaterniond(0,0.70711,-0.70711,0);
    quaternion_norm(q_vc);

    camera_K << 241.4268,0,376.5,
                0,241.4268,240.5,
                0,0,1 ;
    camera_width = 752;
    camera_height = 480;
    tag_width = 0.8;
    tag_in_view_margin = 0.02;

    // Counters/flags
    state_initialized = false;
    measurement_ready = false;
    performed_correction = false;
    filter_active = false;
    upds_since_correction = 0;

    // Tolerances and constants
    small_ang_tol = 1E-10;
    g << 0,0,-9.8;

    initialize_params();

}

void RelativePoseEKF::initialize_params()
{   
    // Filter parameters
    dT_nom = 1/update_freq;
    upd_per_meas = int(ceil(update_freq/measurement_freq));
    num_states = est_bias ? 15 : 9;

    // Build Q/R matrices and camera calibration stuff
    Eigen::VectorXd Q_stack;
    Eigen::VectorXd cov_init_stack(num_states);
    if (est_bias)
    {   
        Q_stack = Eigen::VectorXd::Zero(Q_a.size()+Q_w.size()+Q_ab.size()+Q_wb.size());
        Q_stack << Q_a, Q_w, Q_ab, Q_wb;
        cov_init_stack << r_cov_init * Eigen::VectorXd::Ones(3), v_cov_init * Eigen::VectorXd::Ones(3),
        ang_cov_init * Eigen::VectorXd::Ones(3), ab_cov_init * Eigen::VectorXd::Ones(3), wb_cov_init * Eigen::VectorXd::Ones(3);
    }
    else
    {   
        Q_stack = Eigen::VectorXd::Zero(Q_a.size()+Q_w.size());
        Q_stack << Q_a, Q_w;
        cov_init_stack << r_cov_init * Eigen::VectorXd::Ones(3), v_cov_init * Eigen::VectorXd::Ones(3),
        ang_cov_init * Eigen::VectorXd::Ones(3);
    }
    Q = Q_stack.asDiagonal();
    cov_init = cov_init_stack.asDiagonal();
    cov_pert = cov_init;

    Eigen::VectorXd R_stack(R_r.size()+R_ang.size());
    R_stack << R_r, R_ang;
    R = R_stack.asDiagonal();

    // Camera calibration stuff
    quaternion_norm(q_vc);
    C_vc = q_vc.toRotationMatrix();
    T_vc = Eigen::Translation3d(r_v_cv)*q_vc;

    tag_corners << tag_width/2, -tag_width/2, -tag_width/2, tag_width/2,
                tag_width/2, tag_width/2, -tag_width/2, -tag_width/2,
                0,0,0,0,
                1,1,1,1;
}

void RelativePoseEKF::filter_update()
{
    if (!state_initialized)
        return;
    
    // std::cout << "Starting filter update" << std::endl;

    // Clamp data, decide if should do correction
    mtx_IMU.lock();
    mtx_apriltag.lock();

    Eigen::VectorXd IMU_accel_curr = IMU_accel;
    Eigen::VectorXd IMU_ang_vel_curr = IMU_ang_vel;

    Eigen::VectorXd r_c_tc;
    Eigen::Quaterniond q_ct;
    Eigen::Affine3d T_ct;

    bool perform_correction = false;

    if (measurement_ready && (!limit_measurement_freq || (upds_since_correction+1)>=upd_per_meas))
    {
        // Perform measurement update. Extract AprilTag reading and build pose matrix
        r_c_tc = apriltag_pos;
        q_ct = apriltag_orien;
        measurement_ready = false;

        T_ct = Eigen::Translation3d(r_c_tc)*q_ct;

        if (corner_margin_enbl)
        {
            // Check criteria for a "good" detection
            // Project tag corners to pixel coordinates, verify that at least one has some margin to the edge of the image
            for (int i=0;i<n_tags;++i)
            {
                Eigen::Matrix4d tag_corners_curr;
                tag_corners_curr << tag_widths(i)/2+tag_positions(0,i), -tag_widths(i)/2+tag_positions(0,i), -tag_widths(i)/2+tag_positions(0,i), tag_widths(i)/2+tag_positions(0,i),
                    tag_widths(i)/2+tag_positions(1,i), tag_widths(i)/2+tag_positions(1,i), -tag_widths(i)/2+tag_positions(1,i), -tag_widths(i)/2+tag_positions(1,i),
                    0,0,0,0,
                    1,1,1,1;
                Eigen::MatrixXd tag_corners_c = T_ct*tag_corners_curr;
                Eigen::MatrixXd tag_corners_inv_z = tag_corners_c(seq(2,2),Eigen::all).cwiseInverse();
                Eigen::MatrixXd tag_corners_c_n = tag_corners_c * tag_corners_inv_z.asDiagonal();
                Eigen::MatrixXd tag_corners_px = camera_K*tag_corners_c_n(seq(0,2),Eigen::all);

                Eigen::VectorXd min_px = tag_corners_px.rowwise().minCoeff();
                Eigen::VectorXd max_px = tag_corners_px.rowwise().maxCoeff();

                perform_correction = (min_px(0)>camera_width*tag_in_view_margin &&
                                    min_px(1)>camera_height*tag_in_view_margin &&
                                    max_px(0)<camera_width*(1-tag_in_view_margin) &&
                                    max_px(1)<camera_height*(1-tag_in_view_margin));
                if (perform_correction)
                    break;
            }
            
        }
        else
        {
            perform_correction = true;
        }
        
        // std::cout << "Perform correction: " << std::to_string(perform_correction) << std::endl;

    }

    mtx_apriltag.unlock();
    mtx_IMU.unlock();

    // Prediction step
    double dT = dT_nom; // TODO: Make this dynamic
    Eigen::VectorXd a_nom = IMU_accel_curr - ab_nom;
    Eigen::VectorXd w_nom = IMU_ang_vel_curr - wb_nom;
    Eigen::MatrixXd C_nom = q_nom.toRotationMatrix();

    accel_rel = C_nom*a_nom + g;

    // Propagate nominal state
    Eigen::VectorXd r_check = r_nom + dT*v_nom;
    Eigen::VectorXd v_check = v_nom + dT*accel_rel;
    Eigen::Quaterniond q_check = q_nom*quaternion_exp(dT*w_nom);
    Eigen::VectorXd ab_check = ab_nom;
    Eigen::VectorXd wb_check = wb_nom;

    quaternion_norm(q_check);

    // Calculate Jacobians
    Eigen::MatrixXd F_km1 = Eigen::MatrixXd::Identity(num_states,num_states);
    Eigen::MatrixXd W_km1 = Eigen::MatrixXd::Zero(num_states,Q.rows());
    F_km1(seq(0,2),seq(3,5)) = dT*Eigen::MatrixXd::Identity(3,3);
    F_km1(seq(3,5),seq(6,8)) = -dT*C_nom*skew_symm(a_nom);

    Eigen::VectorXd w_int = dT*w_nom;
    double w_int_angle = w_int.norm();
    if (w_int_angle < small_ang_tol)
    {
        // Avoid zero division, approximate with first order Taylor series
        F_km1(seq(6,8),seq(6,8)) = Eigen::MatrixXd::Identity(3,3) - skew_symm(w_int);
    }
    else
    {
        // Large angle, use full Rodriguez formula
        Eigen::VectorXd w_int_axis = w_int/w_int_angle;
        F_km1(seq(6,8),seq(6,8)) = Eigen::AngleAxisd(-w_int_angle,w_int_axis).toRotationMatrix();
    }

    if (est_bias)
    {
        F_km1(seq(3,5),seq(9,11)) = -dT*C_nom;
        F_km1(seq(6,8),seq(12,14)) = -dT*Eigen::MatrixXd::Identity(3,3);

        W_km1(seq(3,5),seq(0,2)) = -C_nom;
        W_km1(seq(6,num_states-1),seq(3,Q.rows()-1)) = Eigen::MatrixXd::Identity(9,9);
    }
    else
    {
        W_km1(seq(3,5),seq(0,2)) = -C_nom;
        W_km1(seq(6,num_states-1),seq(3,Q.rows()-1)) = Eigen::MatrixXd::Identity(3,3);
    }

    // Propagate covariance
    Eigen::MatrixXd F_km1_T = F_km1.transpose();
    Eigen::MatrixXd W_km1_T = W_km1.transpose();
    Eigen::MatrixXd P_check = F_km1*cov_pert*F_km1_T+W_km1*Q*W_km1_T;
    
    if (perform_correction)
    {
        // Fuse motion model prediction with AprilTag readings
        // Convert AprilTag readings to vehicle state coordinates
        Eigen::MatrixXd C_check = q_check.toRotationMatrix();
        Eigen::MatrixXd C_check_T = C_check.transpose();
        r_t_vt_obs = -(q_check*T_vc*r_c_tc.homogeneous());
        q_tv_obs = (q_vc*q_ct).conjugate();
        quaternion_norm(q_tv_obs);

        // Calculate observed perturbations in measurements
        Eigen::VectorXd delta_r_obs = r_t_vt_obs - r_check;
        Eigen::Quaterniond delta_q_obs = q_check.conjugate()*q_tv_obs;
        quaternion_norm(delta_q_obs);
        Eigen::VectorXd delta_theta_obs = quaternion_log(delta_q_obs);

        // Calculate Jacobians
        Eigen::MatrixXd G_k = Eigen::MatrixXd::Zero(6,num_states);
        G_k(seq(0,2),seq(0,2)) = Eigen::MatrixXd::Identity(3,3);
        G_k(seq(0,2),seq(6,8)) = C_check*skew_symm(C_check_T*r_check);
        G_k(seq(3,5),seq(6,8)) = Eigen::MatrixXd::Identity(3,3);
        Eigen::MatrixXd G_k_T = G_k.transpose();

        Eigen::MatrixXd N_k = Eigen::MatrixXd::Zero(6,6);
        N_k << -C_check*C_vc, Eigen::MatrixXd::Zero(3,3),
                Eigen::MatrixXd::Zero(3,3), C_vc;
        
        Eigen::MatrixXd N_k_T = N_k.transpose();
        
        Eigen::MatrixXd R_k = N_k*R*N_k_T;

        // Form Kalman Gain and execute correction step
        Eigen::MatrixXd K_k = P_check*G_k_T*((G_k*P_check*G_k_T+R_k).inverse());

        Eigen::VectorXd delta_y_obs(delta_r_obs.size()+delta_theta_obs.size());
        delta_y_obs << delta_r_obs, delta_theta_obs;

        Eigen::MatrixXd P_hat = (Eigen::MatrixXd::Identity(num_states,num_states)-K_k*G_k)*P_check;
        Eigen::VectorXd delta_x_hat = K_k*delta_y_obs;

        // Inject correction update, store and reset error state
        // Perturbation state delta x = 
        // [delta r_x/y/z, delta v_x/y/z, delta theta_x/y/z, delta a_bias_x/y/z, delta w_bias_x/y/z ]
        r_nom = r_check + delta_x_hat(seq(0,2));
        v_nom = v_check + delta_x_hat(seq(3,5));
        q_nom = q_check*quaternion_exp(delta_x_hat(seq(6,8)));
        quaternion_norm(q_nom);

        if (est_bias)
        {
            ab_nom = ab_check + delta_x_hat(seq(9,11));
            wb_nom = wb_check + delta_x_hat(seq(12,14));
        }

        cov_pert = P_hat;
        upds_since_correction = 0;

        performed_correction = true;

    }
    else
    {
        // Predictor mode only
        r_nom = r_check;
        v_nom = v_check;
        q_nom = q_check;
        ab_nom = ab_check;
        wb_nom = wb_check;

        cov_pert = P_check;

        upds_since_correction += 1;

        performed_correction = false;
    }

    filter_active = true;
}

void RelativePoseEKF::initialize_state(bool reinit_bias)
{
    mtx_state.lock();

    // Initialize relative pose from last AprilTag detection
    q_nom = (q_vc * apriltag_orien).conjugate();
    quaternion_norm(q_nom);
    Eigen::Quaterniond q_nom_reverse = (apriltag_orien * q_vc).conjugate();
    r_nom = -(q_nom*T_vc*apriltag_pos.homogeneous());

    v_nom = Eigen::VectorXd::Zero(3);

    if (reinit_bias)
    {
        ab_nom = Eigen::VectorXd::Zero(3);
        wb_nom = Eigen::VectorXd::Zero(3);
    }

    cov_pert = cov_init;

    state_initialized = true;
    mtx_state.unlock();
    
}