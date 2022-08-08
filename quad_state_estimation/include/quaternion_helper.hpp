/*
    AER 1810 Quadrotor Landing Project
    Quaternion Helper Functions
*/

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>

// Quaternion exponential map from pure to unit quaternion
Eigen::Quaterniond quaternion_exp(Eigen::VectorXd pure_quat);

// Quaternion logarithmic map from unit to pure quaternion
Eigen::VectorXd quaternion_log(Eigen::Quaterniond unit_quat);

// Normalize unit length quaternion and clip to single cover range
void quaternion_norm(Eigen::Quaterniond &unit_quat);

// Return the 3x3 skew symmetric matrix of a vector
Eigen::MatrixXd skew_symm(Eigen::VectorXd vector);

// Return the 4x1 vector corresponding to a quaternion
Eigen::VectorXd quat_to_vec(Eigen::Quaterniond quaternion);

// Return the quaternion corresponding to a 4x1 vector
Eigen::Quaterniond vec_to_quat(Eigen::VectorXd vec);