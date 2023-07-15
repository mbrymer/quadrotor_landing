/*
    AER 1810 Quadrotor Landing Project
    Mahony Filter Class
*/

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <math.h>
#include <algorithm>
#include <quaternion_helper.h>

class MahonyFilter{

    public:
    MahonyFilter(double dT, double kp, double ki);
    MahonyFilter(double dT, double kp, double ki, Eigen::Vector3d gyro_bias, Eigen::Vector3d accelerometer_bias);

    void Update(const Eigen::Vector3d& acceleration, const Eigen::Vector3d& angular_velocity);
    void SetAccelerometerBias(Eigen::Vector3d accelerometer_bias) { accelerometer_bias_ = std::move(accelerometer_bias);};
    void SetGyroBias(Eigen::Vector3d gyro_bias) { gyro_bias_ = std::move(gyro_bias);};

    Eigen::Quaterniond GetAttitude() const {return q_;};
    Eigen::Vector3d GetGyroBias() const {return gyro_bias_;};

    private:
    Eigen::Quaterniond q_{Eigen::Quaterniond::Identity()};
    Eigen::Vector3d gyro_bias_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d accelerometer_bias_{Eigen::Vector3d::Zero()};

    double dT_;
    double kp_;
    double ki_;

    Eigen::Vector3d g_norm_{0, 0, -1};

};
