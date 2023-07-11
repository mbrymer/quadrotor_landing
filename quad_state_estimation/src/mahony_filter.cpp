/*
    AER 1810 Quadrotor Landing Project
    Mahony Filter Class
*/

#include <mahony_filter.h>

MahonyFilter::MahonyFilter(double dT, double kp, double ki) : dT_{dT}, kp_{kp}, ki_{ki} {};

MahonyFilter::MahonyFilter(double dT, double kp, double ki,
                Eigen::Vector3d gyro_bias, Eigen::Vector3d accelerometer_bias) : dT_{dT}, kp_{kp}, ki_{ki},
                gyro_bias_{std::move(gyro_bias)}, accelerometer_bias_{std::move(accelerometer_bias_)} {};

void MahonyFilter::Update(const Eigen::Vector3d& acceleration, const Eigen::Vector3d& angular_velocity){
    const Eigen::Vector3d accel_hat_norm = -(q_.conjugate() * g_norm_);
    const Eigen::Vector3d accel_measured = acceleration - accelerometer_bias_;
    const double norm = accel_measured.norm();

    constexpr double kSmallAccelTol = {1E-6};
    if (norm < kSmallAccelTol) return;

    const Eigen::Vector3d accel_measured_norm = accel_measured / norm;

    const Eigen::Vector3d omega_meas = -VexSymm(0.5 *
    (accel_measured_norm * accel_hat_norm.transpose() - accel_hat_norm - accel_measured_norm.transpose()));
    const Eigen::Vector3d omega_filter = omega_meas - gyro_bias_ + kp_ * omega_meas;

    Eigen::Quaterniond q_update = q_ * quaternion_exp(dT_ * omega_filter);
    quaternion_norm(q_update);
    q_ = q_update;
    gyro_bias_ = gyro_bias_ - dT_ * ki_ * omega_meas;
}