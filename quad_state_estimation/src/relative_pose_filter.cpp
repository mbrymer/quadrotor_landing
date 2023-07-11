/*
    AER 1810 Quadrotor Landing Project
    Relative Pose Filter Class
*/

#include "relative_pose_filter.h"

#include <relative_pose_utils.h>
#include <relative_pose_filter_settings.h>

RelativePoseFilter::RelativePoseFilter() : mahony_filter_{kDefaultDt, kMahonyDefaultKp, kMahonyDefaultKi} {
    // Default values for parameters
    IMU_accel_ = Eigen::VectorXd::Zero(3);
    IMU_ang_vel_ = Eigen::VectorXd::Zero(3);
    apriltag_pos_ = Eigen::VectorXd::Zero(3);
    apriltag_orien_ = Eigen::Quaterniond(1,0,0,0);

    // State
    r_nom_ = Eigen::VectorXd::Zero(3);
    v_nom_ = Eigen::VectorXd::Zero(3);
    accel_rel_ = Eigen::VectorXd::Zero(3);
    q_nom_ = Eigen::Quaterniond::Identity();
    ab_nom_ = Eigen::VectorXd::Zero(3);
    wb_nom_ = Eigen::VectorXd::Zero(3);
    ab_static_ = Eigen::VectorXd::Zero(3);
    wb_static_ = Eigen::VectorXd::Zero(3);
    r_t_vt_obs_ = Eigen::VectorXd::Zero(3);
    q_tv_obs_ = Eigen::Quaterniond::Identity();

    // Filter Parameters
    update_freq_ = kDefaultLoopRate;
    measurement_freq_ = 10;
    measurement_delay_ = 0.010;
    measurement_delay_max_ = 0.200;
    t_last_update_ = 0;
    est_bias_ = true;
    limit_measurement_freq_ = false;
    corner_margin_enbl_ = true;
    direct_orien_method_ = false;
    multirate_filter_ = false;
    num_states_ = 15;
    measurement_delay_curr_ = 0.0;

    // Process and Measurement Noises
    Q_a_ = 0.005*Eigen::VectorXd::Ones(3);
    Q_w_ = 0.0005*Eigen::VectorXd::Ones(3);
    Q_ab_ = 5E-5*Eigen::VectorXd::Ones(3);
    Q_wb_ = 5E-6*Eigen::VectorXd::Ones(3);

    R_r_ = Eigen::VectorXd::Zero(3);
    R_ang_ = Eigen::VectorXd::Zero(3);
    R_r_ << 0.005, 0.005, 0.015;
    R_ang_ << 0.0025, 0.0025, 0.025;

    // Camera calibration
    r_v_cv_ = Eigen::VectorXd::Zero(3);
    r_v_cv_ << 0,0,-0.073;
    q_vc_ = Eigen::Quaterniond(0,0.70711,-0.70711,0);
    quaternion_norm(q_vc_);

    camera_K_ << 241.4268,0,376.5,
                0,241.4268,240.5,
                0,0,1 ;
    camera_width_ = 752;
    camera_height_ = 480;

    // Target Configuration
    n_tags_ = 1;
    tag_in_view_margin_ = 0.02;

    tag_widths_ = 0.8*Eigen::VectorXd::Ones(1);
    tag_positions_ = Eigen::MatrixXd::Zero(3,1);

    // Counters/flags
    state_initialized_ = false;
    measurement_ready_ = false;
    performed_correction_ = false;
    filter_active_ = false;
    upds_since_correction_ = 0;

    // Tolerances and constants
    g_ << 0,0,-kGravitationalConstant;

    InitializeParams();
}

void RelativePoseFilter::InitializeParams()
{
    // Filter parameters
    dT_nom_ = 1 / update_freq_;
    upd_per_meas_ = int(ceil(update_freq_ / measurement_freq_));
    num_states_ = est_bias_ ? 15 : 9;
    measurement_step_delay_ = std::max(int(measurement_delay_ / dT_nom_ + 0.5), 1);

    // Build Q/R matrices and camera calibration stuff
    Eigen::VectorXd Q_stack;
    Eigen::VectorXd cov_init_stack(num_states_);
    if (est_bias_)
    {
        Q_stack = Eigen::VectorXd::Zero(Q_a_.size()+Q_w_.size()+Q_ab_.size()+Q_wb_.size());
        Q_stack << Q_a_, Q_w_, Q_ab_, Q_wb_;
        cov_init_stack << r_cov_init_ * Eigen::VectorXd::Ones(3), v_cov_init_ * Eigen::VectorXd::Ones(3),
        ang_cov_init_ * Eigen::VectorXd::Ones(3), ab_cov_init_ * Eigen::VectorXd::Ones(3), wb_cov_init_ * Eigen::VectorXd::Ones(3);
    }
    else
    {
        Q_stack = Eigen::VectorXd::Zero(Q_a_.size()+Q_w_.size());
        Q_stack << Q_a_, Q_w_;
        cov_init_stack << r_cov_init_ * Eigen::VectorXd::Ones(3), v_cov_init_ * Eigen::VectorXd::Ones(3),
        ang_cov_init_ * Eigen::VectorXd::Ones(3);
    }
    Q_ = Q_stack.asDiagonal();
    L_Q_ = Eigen::MatrixXd(Eigen::LLT<Eigen::MatrixXd>(Q_));
    cov_init_ = cov_init_stack.asDiagonal();
    cov_pert_ = cov_init_;

    Eigen::VectorXd R_stack(R_r_.size() + R_ang_.size());
    R_stack << R_r_, R_ang_;
    R_apriltag_ = R_stack.asDiagonal();

    L_R_apriltag_ = Eigen::MatrixXd(Eigen::LLT<Eigen::MatrixXd>(R_apriltag_));
    L_R_mahony_ = Eigen::MatrixXd(Eigen::LLT<Eigen::MatrixXd>(R_mahony_.asDiagonal()));
    L_R_gps_ = Eigen::MatrixXd(Eigen::LLT<Eigen::MatrixXd>(R_gps_.asDiagonal()));

    // Camera calibration stuff
    quaternion_norm(q_vc_);
    C_vc_ = q_vc_.toRotationMatrix();
    T_vc_ = Eigen::Translation3d(r_v_cv_) * q_vc_;

}

void RelativePoseFilter::Update(double t_curr)
{
    // std::cout << "Starting filter update" << std::endl;

    // Clamp data, decide if should do correction
    mtx_IMU_.lock();
    mtx_apriltag_.lock();

    Eigen::Vector3d IMU_accel_curr = IMU_accel_;
    Eigen::Vector3d IMU_ang_vel_curr = IMU_ang_vel_;

    Eigen::Vector3d r_c_tc;
    Eigen::Quaterniond q_ct;
    Eigen::Affine3d T_ct;

    bool perform_correction = false;

    if (measurement_ready_ && (!limit_measurement_freq_ || (upds_since_correction_+1) >= upd_per_meas_))
    {
        // Perform measurement update. Extract AprilTag reading and build pose matrix
        r_c_tc = apriltag_pos_;
        q_ct = apriltag_orien_;
        measurement_ready_ = false;

        std::tie(r_t_vt_obs_, q_tv_obs_) = DetectionToPose(apriltag_pos_, apriltag_orien_);

        T_ct = Eigen::Translation3d(r_c_tc) * q_ct;

        if (corner_margin_enbl_)
        {
            // Check criteria for a "good" detection
            // Project tag corners to pixel coordinates, verify that at least one in the bundle has some margin to the edge of the image
            for (int i=0;i<n_tags_; ++i)
            {
                Eigen::Matrix4d tag_corners_curr;
                tag_corners_curr << tag_widths_(i)/2+tag_positions_(0,i), -tag_widths_(i)/2+tag_positions_(0,i), -tag_widths_(i)/2+tag_positions_(0,i), tag_widths_(i)/2+tag_positions_(0,i),
                    tag_widths_(i)/2+tag_positions_(1,i), tag_widths_(i)/2+tag_positions_(1,i), -tag_widths_(i)/2+tag_positions_(1,i), -tag_widths_(i)/2+tag_positions_(1,i),
                    0,0,0,0,
                    1,1,1,1;
                Eigen::MatrixXd tag_corners_c = T_ct*tag_corners_curr;
                Eigen::MatrixXd tag_corners_inv_z = tag_corners_c(seq(2,2),Eigen::all).cwiseInverse();
                Eigen::MatrixXd tag_corners_c_n = tag_corners_c * tag_corners_inv_z.asDiagonal();
                Eigen::MatrixXd tag_corners_px = camera_K_*tag_corners_c_n(seq(0,2),Eigen::all);

                Eigen::VectorXd min_px = tag_corners_px.rowwise().minCoeff();
                Eigen::VectorXd max_px = tag_corners_px.rowwise().maxCoeff();

                perform_correction = (min_px(0)>camera_width_*tag_in_view_margin_ &&
                                    min_px(1)>camera_height_*tag_in_view_margin_ &&
                                    max_px(0)<camera_width_*(1-tag_in_view_margin_) &&
                                    max_px(1)<camera_height_*(1-tag_in_view_margin_));
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

    mtx_apriltag_.unlock();
    mtx_IMU_.unlock();

    // Always update Mahony filter
    mahony_filter_.Update(IMU_accel_curr, IMU_ang_vel_curr);

    if (!state_initialized_) return;

    // With multirate EKF need to perform correction first since it changes state for prediction step
    if (multirate_filter_ && perform_correction)
    {
        // Extract state and covariance prediction at the time the measurement was taken
        measurement_delay_curr_ = dynamic_meas_delay_ ?
        std::min(t_curr - apriltag_time_ + dyn_measurement_delay_offset_, measurement_delay_max_)
        : measurement_delay_;
        int step_delay_curr = std::max(int(measurement_delay_curr_/dT_nom_ + 0.5),1);
        int ind_meas = std::max(int(x_hist_.size())-step_delay_curr,0);
        Eigen::VectorXd x_check = x_hist_[ind_meas];
        Eigen::MatrixXd P_check = P_hist_[ind_meas];

        Eigen::VectorXd x_hat;
        Eigen::MatrixXd P_hat;

        // Execute correction step, store at time measurement was recorded
        CorrectionStep(x_check,P_check,r_c_tc,q_ct,x_hat,P_hat);
        x_hist_[ind_meas] = x_hat;
        P_hist_[ind_meas] = P_hat;

        // Remove history before measurement
        if (ind_meas>0)
        {
            x_hist_.erase(x_hist_.begin(),x_hist_.begin()+ind_meas);
            u_hist_.erase(u_hist_.begin(),u_hist_.begin()+ind_meas);
            P_hist_.erase(P_hist_.begin(),P_hist_.begin()+ind_meas);
        }

        // Update state history by propagating forward based on past IMU measurements
        for (int i = 1; i<x_hist_.size() ; ++i)
        {
            Eigen::Vector3d foo;
            PredictionStep(x_hist_[i-1], P_hist_[i-1], u_hist_[i], x_hist_[i], P_hist_[i], foo);
        }

        // Sync latest estimate to state history
        r_nom_ = x_hist_.back()(seq(0,2));
        v_nom_ = x_hist_.back()(seq(3,5));
        q_nom_ = vec_to_quat(x_hist_.back()(seq(6,9)));
        ab_nom_ = x_hist_.back()(seq(10,12));
        wb_nom_ = x_hist_.back()(seq(13,15));

        cov_pert_ = P_hist_.back();        
    }

    // Prediction step
    // Append the current measurement for this timestep
    Eigen::VectorXd IMU_curr(6);
    IMU_curr << IMU_accel_curr, IMU_ang_vel_curr;

    // Execute prediction
    Eigen::VectorXd x_km1(StateIndex::NumStates);
    x_km1 << r_nom_, v_nom_, quat_to_vec(q_nom_), ab_nom_, wb_nom_;

    Eigen::VectorXd x_check;
    Eigen::VectorXd mu_check;
    Eigen::MatrixXd P_check;
    std::tie(x_check, P_check, mu_check, accel_rel_) = PredictionStep(x_km1, cov_pert_, IMU_curr);

    if (multirate_filter_)
    {
        // Append prediction to state history, store in latest state
        x_hist_.push_back(x_check);
        u_hist_.push_back(IMU_curr);
        P_hist_.push_back(P_check);

        r_nom_ = x_check(seq(0,2));
        v_nom_ = x_check(seq(3,5));
        q_nom_ = vec_to_quat(x_check(seq(6,9)));
        ab_nom_ = x_check(seq(10,12));
        wb_nom_ = x_check(seq(13,15));
        cov_pert_ = P_check;
    }
    else if (perform_correction)
    {
        // Single state EKF, perform correction step and store result
        Eigen::VectorXd x_hat;
        Eigen::MatrixXd P_hat;

        CorrectionStep(x_check,P_check,r_c_tc,q_ct,x_hat,P_hat);

        r_nom_ = x_hat(seq(0,2));
        v_nom_ = x_hat(seq(3,5));
        q_nom_ = vec_to_quat(x_hat(seq(6,9)));
        ab_nom_ = x_hat(seq(10,12));
        wb_nom_ = x_hat(seq(13,15));
        cov_pert_ = P_hat;
    }
    else
    {
        // Single state EKF, prediction only
        // Just store latest prediction
        r_nom_ = x_check(seq(0,2));
        v_nom_ = x_check(seq(3,5));
        q_nom_ = vec_to_quat(x_check(seq(6,9)));
        ab_nom_ = x_check(seq(10,12));
        wb_nom_ = x_check(seq(13,15));
        cov_pert_ = P_check;
    }

    if (perform_correction)
    {
        upds_since_correction_ = 0;
    }
    else
    {
        upds_since_correction_ += 1;
    }

    performed_correction_ = perform_correction;
    filter_active_ = true;
}

void RelativePoseFilter::InitializeState(bool reinit_bias) {
    mtx_state_.lock();

    // Initialize relative pose from last AprilTag detection
    q_nom_ = (q_vc_ * apriltag_orien_).conjugate();
    quaternion_norm(q_nom_);
    r_nom_ = -(q_nom_ * T_vc_ * apriltag_pos_.homogeneous());

    std::tie(r_nom_, q_nom_) = DetectionToPose(apriltag_pos_, apriltag_orien_);

    v_nom_ = Eigen::VectorXd::Zero(3);

    if (reinit_bias)
    {
        ab_nom_ = Eigen::VectorXd::Zero(3);
        wb_nom_ = Eigen::VectorXd::Zero(3);
    }

    cov_pert_ = cov_init_;

    // Initialize state history with a single value
    Eigen::VectorXd x_stack = Eigen::VectorXd::Zero(r_nom_.size()+v_nom_.size()+4+ab_nom_.size()+wb_nom_.size());
    x_stack(seq(0,2)) = r_nom_;
    x_stack(seq(3,5)) = v_nom_;
    x_stack(seq(6,9)) = Eigen::Vector4d(q_nom_.x(),q_nom_.y(),q_nom_.z(),q_nom_.w());

    if (est_bias_)
    {
        x_stack(seq(10,12)) = ab_nom_;
        x_stack(seq(13,15)) = wb_nom_;
    }

    x_hist_ = std::vector<Eigen::VectorXd>{x_stack};
    u_hist_ = std::vector<Eigen::VectorXd>{Eigen::VectorXd::Zero(6)};
    P_hist_ = std::vector<Eigen::MatrixXd>{cov_init_};

    state_initialized_ = true;
    mtx_state_.unlock();

}

std::tuple<Eigen::Vector3d, Eigen::Quaterniond> RelativePoseFilter::DetectionToPose(
    const Eigen::Vector3d& apriltag_pos, const Eigen::Quaterniond& apriltag_orien){
    Eigen::Quaterniond q_nom = (q_vc_ * apriltag_orien).conjugate();
    quaternion_norm(q_nom);
    Eigen::Vector3d r_nom = -(q_nom * T_vc_ * apriltag_pos.homogeneous());
    return {r_nom, q_nom};
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd,
Eigen::MatrixXd, Eigen::Vector3d> RelativePoseFilter::PredictionStep(const Eigen::VectorXd& x_km1, const Eigen::MatrixXd& P_km1, const Eigen::VectorXd& u) {
    // Prediction step
    Eigen::Vector3d r_km1 = x_km1(seq(StateIndex::X,StateIndex::Z));
    Eigen::Vector3d v_km1 = x_km1(seq(StateIndex::Vx, StateIndex::Vz));
    Eigen::Quaterniond q_km1(x_km1(StateIndex::q_w), x_km1(StateIndex::q_x),
                                    x_km1(StateIndex::q_y), x_km1(StateIndex::q_z));
    Eigen::Vector3d ab_km1 = x_km1(seq(StateIndex::ab_x,StateIndex::ab_z));
    Eigen::Vector3d wb_km1 = x_km1(seq(StateIndex::wb_x,StateIndex::wb_z));

    double dT = dT_nom_; // TODO: Make this dynamic
    Eigen::Vector3d a_nom = u(seq(0,2)) - ab_km1 - ab_static_;
    Eigen::Vector3d w_nom = u(seq(3,5)) - wb_km1 - wb_static_;
    Eigen::Matrix3d C_km1 = q_km1.toRotationMatrix();

    Eigen::Vector3d pose_accel = C_km1 * a_nom + g_;

    // Draw sigma points from stacked state and process noise
    Eigen::MatrixXd dz_sigma_points = DrawSigmaPointsZeroMean(P_km1, L_Q_, kSigmaPointKappa);
    const double L = dz_sigma_points.rows();
    const double n_z = dz_sigma_points.cols();

    // Propagate nominal state
    Eigen::Vector3d r_check = r_km1 + dT * v_km1;
    Eigen::Vector3d v_check = v_km1 + dT * pose_accel;
    Eigen::Quaterniond q_check = q_km1 * quaternion_exp(dT * w_nom);
    Eigen::Vector3d ab_check = ab_km1;
    Eigen::Vector3d wb_check = wb_km1;

    quaternion_norm(q_check);

    // Compute nominal state
    Eigen::VectorXd x_check = Eigen::VectorXd::Zero(StateIndex::NumStates);
    x_check << r_check, v_check, quat_to_vec(q_check), ab_check, wb_check;

    // Propagate sigma points
    Eigen::MatrixXd dx_check(PertIndex::NumPertStates, n_z);

    for (int i = 0; i < dz_sigma_points.cols() ; ++i){
        Eigen::Matrix3d delta_C_i = exponential_map(dz_sigma_points(seq(PertIndex::theta_x, PertIndex::theta_z), i));
        Eigen::Vector3d a_i = (a_nom - dz_sigma_points(seq(PertIndex::ab_x, PertIndex::ab_z), i)
                - dz_sigma_points(seq(ProcessNoise::a_nX, ProcessNoise::a_nZ)));
        Eigen::Vector3d w_i = (w_nom - dz_sigma_points(seq(PertIndex::wb_x, PertIndex::wb_z), i)
                - dz_sigma_points(seq(ProcessNoise::w_nX, ProcessNoise::w_nZ)));
        Eigen::Quaterniond q_km1_i = q_km1 * quaternion_exp(dz_sigma_points(seq(PertIndex::theta_x, PertIndex::theta_z, i)));
        Eigen::Quaterniond q_check_i = q_km1_i * quaternion_exp(dT * w_i);
        quaternion_norm(q_check_i);

        dx_check(seq(PertIndex::X, PertIndex::Z), i) =
        dz_sigma_points(seq(PertIndex::X, PertIndex::Z), i) + dT * dz_sigma_points(seq(PertIndex::Vx, PertIndex::Vz), i);

        dx_check(seq(PertIndex::Vx, PertIndex::Vz), i) = (v_km1 + dz_sigma_points(seq(PertIndex::Vx, PertIndex::Vz), i)
        + dT * (C_km1 * delta_C_i * a_i + g_)) - v_check;

        dx_check(seq(PertIndex::theta_x, PertIndex::theta_z), i) = quaternion_log(q_check.conjugate() * q_check_i);

        dx_check(seq(PertIndex::ab_x, PertIndex::ab_z), i) = dz_sigma_points(seq(PertIndex::ab_x, PertIndex::ab_z), i)
        + dT * dz_sigma_points(seq(ProcessNoise::ab_nX, ProcessNoise::ab_nZ));

        dx_check(seq(PertIndex::wb_x, PertIndex::wb_z), i) = dz_sigma_points(seq(PertIndex::wb_x, PertIndex::wb_z), i)
        + dT * dz_sigma_points(seq(ProcessNoise::wb_nX, ProcessNoise::wb_nZ));
    }

    // Compute statistics of propagated sigma points
    const double alpha_0 = kSigmaPointKappa / (kSigmaPointKappa + L);
    const double alpha = 0.5 / (kSigmaPointKappa + L);

    Eigen::VectorXd mu_check = alpha_0 * dx_check.col(0) +
    alpha * (dx_check.block(0, 1, PertIndex::NumPertStates, dx_check.cols() - 1).rowwise().sum());

    Eigen::MatrixXd P_check = alpha_0 * (dx_check.col(0) - mu_check) * (dx_check.col(0) - mu_check).transpose();

    for (int i = i; i < n_z; ++i) {
        const Eigen::VectorXd deviation_from_mean = dx_check.col(i) - mu_check;
        P_check += alpha * deviation_from_mean * deviation_from_mean.transpose();
    }

    return {x_check, mu_check, P_check, pose_accel};
}

std::tuple<Eigen::VectorXd, Eigen::MatrixXd> RelativePoseFilter::CorrectionStep(const Eigen::VectorXd& x_check, const Eigen::VectorXd& mu_check,
                                                            const Eigen::MatrixXd& P_check, double roll_mahony, double pitch_mahony,
                                                            std::optional<double> v_gps, const std::optional<Eigen::VectorXd>& r_c_tc,
                                                            const std::optional<Eigen::Quaterniond>& q_ct, bool valid_detection){
    // Fuse motion model prediction with measurements
    // Unpack state
    Eigen::VectorXd r_check = x_check(seq(StateIndex::X, StateIndex::Z));
    Eigen::VectorXd v_check = x_check(seq(StateIndex::Vx, StateIndex::Vz));
    Eigen::Quaterniond q_check = vec_to_quat(x_check(seq(StateIndex::q_x, StateIndex::q_w)));
    Eigen::VectorXd ab_check = x_check(seq(StateIndex::ab_x, StateIndex::ab_z));
    Eigen::VectorXd wb_check = x_check(seq(StateIndex::wb_x, StateIndex::wb_z));

    // Build measurement covariance based on available measurements
    // Mahony is always available
    bool apriltag_avail = r_c_tc.has_value() && q_ct.has_value() && valid_detection;
    int n_measurements = MahonyMeasurementIndex::NumAngles + apriltag_avail * NumAprilTagIndices + v_gps.has_value();
    int n_measurement_noise = MahonyMeasurementIndex::NumAngles +
                              apriltag_avail * NumAprilTagIndices +
                              v_gps.has_value() * NumGPSSpeedNoise;

    Eigen::MatrixXd measurement_noise_L = Eigen::MatrixXd::Zero(n_measurement_noise, n_measurement_noise);
    measurement_noise_L(seq(0,1), seq(0,1)) = L_R_mahony_;
    int ind_measurement_noise = 2;

    if (apriltag_avail) {
        measurement_noise_L(seq(ind_measurement_noise, ind_measurement_noise + NumAprilTagIndices - 1),
        seq(ind_measurement_noise, ind_measurement_noise + NumAprilTagIndices - 1)) = L_R_apriltag_;
        ind_measurement_noise += NumAprilTagIndices;
    }
    if (v_gps.has_value()) {
        measurement_noise_L(seq(ind_measurement_noise, ind_measurement_noise + NumGPSSpeedNoise - 1),
        seq(ind_measurement_noise, ind_measurement_noise + NumGPSSpeedNoise - 1)) = L_R_gps_;
    }

    // Draw sigma points from stacked 
    Eigen::MatrixXd dz_sigma_points_k = DrawSigmaPoints(mu_check, P_check, measurement_noise_L, kSigmaPointKappa);
    const double L = dz_sigma_points_k.rows();
    const double n_z = dz_sigma_points_k.cols();

    // Unpack measurement noise
    Eigen::MatrixXd n_mahony = dz_sigma_points_k(seq(NumPertStates, NumPertStates + MahonyMeasurementIndex::NumAngles - 1),
                                                 seq(0, Eigen::last));
    std::optional<Eigen::MatrixXd> n_apriltag;
    std::optional<Eigen::MatrixXd> n_gps;

    int ind_measurement_noise = NumPertStates + MahonyMeasurementIndex::NumAngles;
    if (apriltag_avail) {
        n_apriltag = dz_sigma_points_k(seq(ind_measurement_noise, ind_measurement_noise + NumAprilTagIndices - 1),
                                                 seq(0, Eigen::last));
        ind_measurement_noise += NumAprilTagIndices;
    }
    if (v_gps.has_value()) {
        n_gps = dz_sigma_points_k(seq(ind_measurement_noise, ind_measurement_noise + NumGPSSpeedNoise - 1),
                                                 seq(0, Eigen::last));
    }

    // Pass sigma points through the measurement model to build our predicted measurements
    Eigen::MatrixXd y_check = Eigen::MatrixXd::Zero(n_measurements, n_z);

    for (int i = 0; i < dz_sigma_points_k.cols(); ++i) {
        Eigen::Quaterniond delta_q_tv_i = quaternion_exp(dz_sigma_points_k(seq(theta_x, theta_z), i));
        Eigen::Quaterniond q_tv_i = q_check * delta_q_tv_i;
        Eigen::Matrix3d C_tv_i = q_tv_i.toRotationMatrix();
        Eigen::Vector3d euler_angles = C_tv_i.eulerAngles(2, 1, 0);
        y_check(MahonyMeasurementIndex::roll, i) = euler_angles(MahonyMeasurementIndex::roll) + n_mahony(MahonyMeasurementIndex::roll, i);
        y_check(MahonyMeasurementIndex::pitch, i) = euler_angles(MahonyMeasurementIndex::pitch) + n_mahony(MahonyMeasurementIndex::pitch, i);

        int ind_measurement = 2;
        if (apriltag_avail) {
            Eigen::Vector3d r_t_vt_i = r_check + dz_sigma_points_k(seq(PertIndex::X, PertIndex::Z),i);
            Eigen::Vector3d r_c_tc_i = T_vc_.inverse() * (C_tv_i.transpose() * -r_t_vt_i).homogeneous() +
                                        n_apriltag.value()(seq(r_x, r_z), i);
            Eigen::Quaterniond delta_q_ct_i = delta_q_tv_i.conjugate() * quaternion_exp(n_apriltag.value()(seq(theta_x, theta_z), i));
            Eigen::Vector3d theta_ct_i = quaternion_log(delta_q_ct_i);

            y_check(seq(ind_measurement, ind_measurement + NumAprilTagIndices - 1), i) = r_c_tc_i;
            ind_measurement += NumAprilTagIndices;
        }
        if (v_gps.has_value()) {
            Eigen::Vector2d v_t_tv_i = v_check(seq(0,1)) + dz_sigma_points_k(seq(PertIndex::Vx, PertIndex::Vy), i) + n_gps.value().col(i);
            y_check(ind_measurement, i) = v_t_tv_i.norm();
        }
    }

    // Calculate statistics of measurement sigma points
    const double alpha_0 = kSigmaPointKappa / (kSigmaPointKappa + L);
    const double alpha = 0.5 / (kSigmaPointKappa + L);

    Eigen::VectorXd mu_y = alpha_0 * y_check.col(0) +
    alpha * (y_check(seq(0, Eigen::last), seq(1, Eigen::last)).rowwise().sum());

    Eigen::MatrixXd sig_yy = alpha_0 * (y_check.col(0) - mu_y) * (y_check.col(0) - mu_y).transpose();
    Eigen::MatrixXd sig_xy = alpha_0 * (dz_sigma_points_k(seq(0,PertIndex::NumPertStates - 1),0) - mu_check) * (y_check.col(0) - mu_y).transpose();

    for (int i = 1; i < n_z; ++i) {
        const Eigen::VectorXd deviation_from_mean_x = dz_sigma_points_k(seq(0,PertIndex::NumPertStates - 1), i) - mu_check;
        const Eigen::VectorXd deviation_from_mean_y = y_check.col(i) - mu_y;
        sig_yy += alpha * deviation_from_mean_y * deviation_from_mean_y.transpose();
        sig_xy += alpha * deviation_from_mean_x * deviation_from_mean_y.transpose();
    }

    // Build measurement vector
    Eigen::VectorXd y_observed = Eigen::VectorXd::Zero(n_measurements);
    y_observed(MahonyMeasurementIndex::roll) = roll_mahony;
    y_observed(MahonyMeasurementIndex::pitch) = pitch_mahony;

    int ind_measurement = 2;
    if (apriltag_avail) {
        // Calculate observed perturbations in AprilTag orientation measurement
        Eigen::Quaterniond delta_q_obs = q_check * q_vc_ * q_ct.value();
        Eigen::Vector3d theta_obs = quaternion_log(delta_q_obs);

        y_observed(seq(ind_measurement + r_x, ind_measurement + r_z)) = r_c_tc.value();
        y_observed(seq(ind_measurement + AprilTagMeasurementIndex::theta_x,
                        ind_measurement + AprilTagMeasurementIndex::theta_z)) = theta_obs;
        ind_measurement += NumAprilTagIndices;
    }
    if (v_gps.has_value()) {
        y_observed(ind_measurement) = v_gps.value();
    }

    // Form Kalman Gain and execute correction step
    Eigen::MatrixXd K_k = sig_xy * sig_yy.inverse();
    // Eigen::MatrixXd K_k = (sig_yy.transpose().colPivHouseholderQr().solve(sig_xy.transpose())).transpose();

    Eigen::MatrixXd P_hat = P_check - K_k * sig_xy.transpose();
    Eigen::VectorXd delta_x_hat = mu_check + K_k * (y_observed - mu_y);

    // Inject correction update, store and reset error state
    Eigen::VectorXd r_hat = r_check + delta_x_hat(seq(PertIndex::X, PertIndex::Z));
    Eigen::VectorXd v_hat = v_check + delta_x_hat(seq(PertIndex::Vx, PertIndex::Vz));
    Eigen::Quaterniond q_hat = q_check*quaternion_exp(delta_x_hat(seq(PertIndex::theta_x, PertIndex::theta_z)));
    quaternion_norm(q_hat);

    Eigen::VectorXd ab_hat = ab_check + delta_x_hat(seq(PertIndex::ab_x, PertIndex::ab_z));
    Eigen::VectorXd wb_hat = wb_check + delta_x_hat(seq(PertIndex::wb_x, PertIndex::wb_z));

    Eigen::VectorXd x_hat = Eigen::VectorXd::Zero(StateIndex::NumStates);
    x_hat << r_hat, v_hat, quat_to_vec(q_hat), ab_hat, wb_hat;

    return {x_hat, P_hat};
}