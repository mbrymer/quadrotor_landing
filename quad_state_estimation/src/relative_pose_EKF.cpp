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
    ab_static = Eigen::VectorXd::Zero(3);
    wb_static = Eigen::VectorXd::Zero(3);
    r_t_vt_obs = Eigen::VectorXd::Zero(3);
    q_tv_obs = Eigen::Quaterniond::Identity();

    // Filter Parameters
    update_freq = 100;
    measurement_freq = 10;
    measurement_delay = 0.010;
    measurement_delay_max = 0.200;
    t_last_update = 0;
    est_bias = true;
    limit_measurement_freq = false;
    corner_margin_enbl = true;
    direct_orien_method = false;
    multirate_ekf = false;
    num_states = 15;
    measurement_delay_curr = 0.0;

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

    // Target Configuration
    n_tags = 1;
    tag_in_view_margin = 0.02;

    tag_widths = 0.8*Eigen::VectorXd::Ones(1);
    tag_positions = Eigen::MatrixXd::Zero(3,1);

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
    measurement_step_delay = std::max(int(measurement_delay/dT_nom+0.5),1);

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

}

void RelativePoseEKF::filter_update(double t_curr)
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
            // Project tag corners to pixel coordinates, verify that at least one in the bundle has some margin to the edge of the image
            for (int i=0;i<n_tags; ++i)
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

    // With multirate EKF need to perform correction first since it changes state for prediction step
    if (multirate_ekf && perform_correction)
    {
        // Extract state and covariance prediction at the time the measurement was taken
        measurement_delay_curr = dynamic_meas_delay ? std::min(t_curr-apriltag_time+dyn_measurement_delay_offset,measurement_delay_max): measurement_delay;
        int step_delay_curr = std::max(int(measurement_delay_curr/dT_nom+0.5),1);
        int ind_meas = std::max(int(x_hist.size())-step_delay_curr,0);
        Eigen::VectorXd x_check = x_hist[ind_meas];
        Eigen::MatrixXd P_check = P_hist[ind_meas];

        Eigen::VectorXd x_hat;
        Eigen::MatrixXd P_hat;

        // Execute correction step, store at time measurement was recorded
        this->correction_step(x_check,P_check,r_c_tc,q_ct,x_hat,P_hat);
        x_hist[ind_meas] = x_hat;
        P_hist[ind_meas] = P_hat;

        // Remove history before measurement
        if (ind_meas>0)
        {
            x_hist.erase(x_hist.begin(),x_hist.begin()+ind_meas);
            u_hist.erase(u_hist.begin(),u_hist.begin()+ind_meas);
            P_hist.erase(P_hist.begin(),P_hist.begin()+ind_meas);
        }

        // Update state history by propagating forward based on past IMU measurements
        for (int i = 1; i<x_hist.size() ; ++i)
        {
            Eigen::VectorXd foo;
            this->prediction_step(x_hist[i-1],P_hist[i-1],u_hist[i],x_hist[i],P_hist[i],foo);
        }

        // Sync latest estimate to state history
        r_nom = x_hist.back()(seq(0,2));
        v_nom = x_hist.back()(seq(3,5));
        q_nom = vec_to_quat(x_hist.back()(seq(6,9)));
        ab_nom = x_hist.back()(seq(10,12));
        wb_nom = x_hist.back()(seq(13,15));

        cov_pert = P_hist.back();        
    }

    // Prediction step
    // Append the current measurement for this timestep
    Eigen::VectorXd IMU_curr(6);
    IMU_curr << IMU_accel_curr, IMU_ang_vel_curr;

    // Execute prediction
    Eigen::VectorXd x_km1(r_nom.size()+v_nom.size()+4+ab_nom.size()+wb_nom.size());
    x_km1 << r_nom, v_nom, quat_to_vec(q_nom), ab_nom, wb_nom;

    Eigen::VectorXd x_check;
    Eigen::MatrixXd P_check;
    this->prediction_step(x_km1,cov_pert,IMU_curr,x_check,P_check,accel_rel);

    if (multirate_ekf)
    {
        // Append prediction to state history, store in latest state
        x_hist.push_back(x_check);
        u_hist.push_back(IMU_curr);
        P_hist.push_back(P_check);

        r_nom = x_check(seq(0,2));
        v_nom = x_check(seq(3,5));
        q_nom = vec_to_quat(x_check(seq(6,9)));
        ab_nom = x_check(seq(10,12));
        wb_nom = x_check(seq(13,15));
        cov_pert = P_check;
    }
    else if (perform_correction)
    {
        // Single state EKF, perform correction step and store result
        Eigen::VectorXd x_hat;
        Eigen::MatrixXd P_hat;

        this->correction_step(x_check,P_check,r_c_tc,q_ct,x_hat,P_hat);

        r_nom = x_hat(seq(0,2));
        v_nom = x_hat(seq(3,5));
        q_nom = vec_to_quat(x_hat(seq(6,9)));
        ab_nom = x_hat(seq(10,12));
        wb_nom = x_hat(seq(13,15));
        cov_pert = P_hat;
    }
    else
    {
        // Single state EKF, prediction only
        // Just store latest prediction
        r_nom = x_check(seq(0,2));
        v_nom = x_check(seq(3,5));
        q_nom = vec_to_quat(x_check(seq(6,9)));
        ab_nom = x_check(seq(10,12));
        wb_nom = x_check(seq(13,15));
        cov_pert = P_check;
    }

    if (perform_correction)
    {
        upds_since_correction = 0;
    }
    else
    {
        upds_since_correction += 1;
    }

    performed_correction = perform_correction;
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

    // Initialize state history with a single value
    Eigen::VectorXd x_stack = Eigen::VectorXd::Zero(r_nom.size()+v_nom.size()+4+ab_nom.size()+wb_nom.size());
    x_stack(seq(0,2)) = r_nom;
    x_stack(seq(3,5)) = v_nom;
    x_stack(seq(6,9)) = Eigen::Vector4d(q_nom.x(),q_nom.y(),q_nom.z(),q_nom.w());

    if (est_bias)
    {
        x_stack(seq(10,12)) = ab_nom;
        x_stack(seq(13,15)) = wb_nom;
    }

    x_hist = std::vector<Eigen::VectorXd>{x_stack};
    u_hist = std::vector<Eigen::VectorXd>{Eigen::VectorXd::Zero(6)};
    P_hist = std::vector<Eigen::MatrixXd>{cov_init};

    state_initialized = true;
    mtx_state.unlock();

}

void RelativePoseEKF::prediction_step(Eigen::VectorXd x_km1, Eigen::MatrixXd P_km1, Eigen::VectorXd u,
                            Eigen::VectorXd &x_check, Eigen::MatrixXd &P_check, Eigen::VectorXd &pose_accel)
{
    // Prediction step
    Eigen::VectorXd r_km1 = x_km1(seq(0,2));
    Eigen::VectorXd v_km1 = x_km1(seq(3,5));
    Eigen::Quaterniond q_km1(x_km1(9),x_km1(6),x_km1(7),x_km1(8));
    Eigen::VectorXd ab_km1 = x_km1(seq(10,12));
    Eigen::VectorXd wb_km1 = x_km1(seq(13,15));

    double dT = dT_nom; // TODO: Make this dynamic
    Eigen::VectorXd a_nom = u(seq(0,2)) - ab_km1 - ab_static;
    Eigen::VectorXd w_nom = u(seq(3,5)) - wb_km1 - wb_static;
    Eigen::MatrixXd C_km1 = q_km1.toRotationMatrix();

    pose_accel = Eigen::VectorXd::Zero(3);
    pose_accel = C_km1*a_nom + g;

    // Propagate nominal state
    Eigen::VectorXd r_check = r_km1 + dT*v_km1;
    Eigen::VectorXd v_check = v_km1 + dT*pose_accel;
    Eigen::Quaterniond q_check = q_km1*quaternion_exp(dT*w_nom);
    Eigen::VectorXd ab_check = ab_km1;
    Eigen::VectorXd wb_check = wb_km1;

    quaternion_norm(q_check);

    // Return nominal state
    x_check = Eigen::VectorXd::Zero(r_check.size()+v_check.size()+4+ab_check.size()+wb_check.size());
    x_check << r_check, v_check, quat_to_vec(q_check), ab_check, wb_check;

    // Calculate Jacobians
    Eigen::MatrixXd F_km1 = Eigen::MatrixXd::Identity(num_states,num_states);
    Eigen::MatrixXd W_km1 = Eigen::MatrixXd::Zero(num_states,Q.rows());
    F_km1(seq(0,2),seq(3,5)) = dT*Eigen::MatrixXd::Identity(3,3);
    F_km1(seq(3,5),seq(6,8)) = -dT*C_km1*skew_symm(a_nom);

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
        F_km1(seq(3,5),seq(9,11)) = -dT*C_km1;
        F_km1(seq(6,8),seq(12,14)) = -dT*Eigen::MatrixXd::Identity(3,3);

        W_km1(seq(3,5),seq(0,2)) = -C_km1;
        W_km1(seq(6,num_states-1),seq(3,Q.rows()-1)) = Eigen::MatrixXd::Identity(9,9);
    }
    else
    {
        W_km1(seq(3,5),seq(0,2)) = -C_km1;
        W_km1(seq(6,num_states-1),seq(3,Q.rows()-1)) = Eigen::MatrixXd::Identity(3,3);
    }

    // Propagate covariance
    Eigen::MatrixXd F_km1_T = F_km1.transpose();
    Eigen::MatrixXd W_km1_T = W_km1.transpose();
    P_check = F_km1*P_km1*F_km1_T+W_km1*Q*W_km1_T;
}

void RelativePoseEKF::correction_step(Eigen::VectorXd x_check, Eigen::MatrixXd P_check, Eigen::VectorXd r_c_tc, Eigen::Quaterniond q_ct,
                            Eigen::VectorXd &x_hat, Eigen::MatrixXd &P_hat)
{
    // Fuse motion model prediction with AprilTag readings
    // Unpack state
    Eigen::VectorXd r_check = x_check(seq(0,2));
    Eigen::VectorXd v_check = x_check(seq(3,5));
    Eigen::Quaterniond q_check = vec_to_quat(x_check(seq(6,9)));
    Eigen::VectorXd ab_check = x_check(seq(10,12));
    Eigen::VectorXd wb_check = x_check(seq(13,15));

    // Convert AprilTag readings to vehicle state coordinates
    Eigen::MatrixXd C_check = q_check.toRotationMatrix();
    Eigen::MatrixXd C_check_T = C_check.transpose();
    q_tv_obs = (q_vc*q_ct).conjugate();
    quaternion_norm(q_tv_obs);

    if (direct_orien_method)
    {
        // Directly use reported orientation from AprilTag when calculating observed position
        // Removes sensitivity to predicted orientation, but also removes ability for position measurements to influence orientation
        r_t_vt_obs = -(q_tv_obs*T_vc*r_c_tc.homogeneous());
    }
    else
    {
        // Linearize about predicted orientation as conventional EKF. More information about orientation but sensitive to q_check
        r_t_vt_obs = -(q_check*T_vc*r_c_tc.homogeneous());
    }

    // Calculate observed perturbations in measurements
    Eigen::VectorXd delta_r_obs = r_t_vt_obs - r_check;
    Eigen::Quaterniond delta_q_obs = q_check.conjugate()*q_tv_obs;
    quaternion_norm(delta_q_obs);
    Eigen::VectorXd delta_theta_obs = quaternion_log(delta_q_obs);

    // Calculate Jacobians
    Eigen::MatrixXd G_k = Eigen::MatrixXd::Zero(6,num_states);
    G_k(seq(0,2),seq(0,2)) = Eigen::MatrixXd::Identity(3,3);
    if (!direct_orien_method)
    {
        G_k(seq(0,2),seq(6,8)) = C_check*skew_symm(C_check_T*r_check);
    }
    G_k(seq(3,5),seq(6,8)) = Eigen::MatrixXd::Identity(3,3);
    Eigen::MatrixXd G_k_T = G_k.transpose();

    Eigen::MatrixXd N_k = Eigen::MatrixXd::Zero(6,6);
    N_k << -C_check*C_vc, Eigen::MatrixXd::Zero(3,3),
            Eigen::MatrixXd::Zero(3,3), C_vc;
    if (direct_orien_method)
    {
        N_k(seq(0,2),seq(3,5)) = skew_symm(r_check);
    }

    Eigen::MatrixXd N_k_T = N_k.transpose();

    Eigen::MatrixXd R_k = N_k*R*N_k_T;

    // Form Kalman Gain and execute correction step
    Eigen::MatrixXd K_k = P_check*G_k_T*((G_k*P_check*G_k_T+R_k).inverse());

    Eigen::VectorXd delta_y_obs(delta_r_obs.size()+delta_theta_obs.size());
    delta_y_obs << delta_r_obs, delta_theta_obs;

    P_hat = (Eigen::MatrixXd::Identity(num_states,num_states)-K_k*G_k)*P_check;
    Eigen::VectorXd delta_x_hat = K_k*delta_y_obs;

    // Inject correction update, store and reset error state
    // Perturbation state delta x =
    // [delta r_x/y/z, delta v_x/y/z, delta theta_x/y/z, delta a_bias_x/y/z, delta w_bias_x/y/z ]
    Eigen::VectorXd r_hat = r_check + delta_x_hat(seq(0,2));
    Eigen::VectorXd v_hat = v_check + delta_x_hat(seq(3,5));
    Eigen::Quaterniond q_hat = q_check*quaternion_exp(delta_x_hat(seq(6,8)));
    quaternion_norm(q_hat);

    Eigen::VectorXd ab_hat = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd wb_hat = Eigen::VectorXd::Zero(3);

    if (est_bias)
    {
        ab_hat = ab_check + delta_x_hat(seq(9,11));
        wb_hat = wb_check + delta_x_hat(seq(12,14));
    }

    x_hat = Eigen::VectorXd::Zero(r_hat.size()+v_hat.size()+4+ab_hat.size()+wb_hat.size());
    x_hat << r_hat, v_hat, quat_to_vec(q_hat), ab_hat, wb_hat;
}