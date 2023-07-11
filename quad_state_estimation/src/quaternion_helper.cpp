/*
    AER 1810 Quadrotor Landing Project
    Quaternion Helper Functions
*/

#include <quaternion_helper.h>

constexpr double kSmallRotationTolerance{1E-10};

// Quaternion exponential map from pure to unit quaternion
Eigen::Quaterniond quaternion_exp(Eigen::VectorXd pure_quat) {
    Eigen::Quaterniond unit_quat;

    double norm = pure_quat.norm();

    // Real part is fine
    unit_quat.w() = cos(norm/2);

    if (norm < kSmallRotationTolerance)
    {
        // Small rotation, use Taylor series approximation for vector part
        unit_quat.vec() = pure_quat/2*(1-pow(norm,2)/24);
    }
    else
    {
        // Rotation is large enough, use true rotation axis
        unit_quat.vec() = pure_quat/norm*sin(norm/2);
    }

    quaternion_norm(unit_quat);

    return unit_quat;
}

// Quaternion logarithmic map from unit to pure quaternion
Eigen::VectorXd quaternion_log(Eigen::Quaterniond unit_quat) {
    Eigen::VectorXd pure_quat = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd quat_vec = Eigen::VectorXd::Zero(3);

    quat_vec << unit_quat.x(),unit_quat.y(),unit_quat.z();
    double vec_norm = quat_vec.norm();

    if (vec_norm < kSmallRotationTolerance)
    {
        // Approximate map with Taylor series for atan
        pure_quat = 2/unit_quat.w()*(1-pow(vec_norm/unit_quat.w(),2)/3)*quat_vec;
    }
    else
    {
        // Large angles, use full version of atan
        double phi = 2*atan2(vec_norm,unit_quat.w());
        pure_quat = phi/vec_norm*quat_vec;
    }
    
    return pure_quat;
}

// Normalize unit length quaternion and clip to single cover range
void quaternion_norm(Eigen::Quaterniond &unit_quat) {
    double q_w_lim = -0.75;
    unit_quat.normalize();

    if (unit_quat.w()<q_w_lim)
    {
        unit_quat.w() = -unit_quat.w();
        unit_quat.x() = -unit_quat.x();
        unit_quat.y() = -unit_quat.y();
        unit_quat.z() = -unit_quat.z();
    }
}

// Return the 3x3 skew symmetric matrix of a vector
Eigen::MatrixXd skew_symm(Eigen::VectorXd vector) {
    Eigen::MatrixXd mat(3,3);
    
    mat << 0, -vector(2), vector(1),
        vector(2), 0 , -vector(0),
        -vector(1), vector(0), 0;
    
    return mat;
}

// Return the 4x1 vector corresponding to a quaternion
Eigen::VectorXd quat_to_vec(Eigen::Quaterniond quaternion) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(4);
    vec << quaternion.x(),quaternion.y(),quaternion.z(),quaternion.w();
    return vec;
}

// Return the quaternion corresponding to a 4x1 vector
Eigen::Quaterniond vec_to_quat(Eigen::VectorXd vec) {
    Eigen::Quaterniond quat(vec(3),vec(0),vec(1),vec(2));
    return quat;
}

// Return the vex operator of a skew symmetric matrix
Eigen::Vector3d VexSymm(const Eigen::Matrix3d& skew_symm) {
    return Eigen::Vector3d{(skew_symm[2,1] - skew_symm[1,2]) / 2,
                           (skew_symm[0,2] - skew_symm[2,0]) / 2,
                           (skew_symm[1,0] - skew_symm[0,1]) / 2};
}

// Return the rotation matrix corresponding to a Lie algebra vector
Eigen::Matrix3d exponential_map(const Eigen::Vector3d& lie_vector) {
    const double norm = lie_vector.norm();

    // First order approximation for small angles
    if (norm < kSmallRotationTolerance) return (Eigen::Matrix3d::Identity()).colwise() + skew_symm(lie_vector);

    // Full exponential map
    const Eigen::Vector3d vector_norm = lie_vector / norm;

    return (cos(norm) * Eigen::Matrix3d::Identity() +
    (1 - cos(norm)) * vector_norm * vector_norm.transpose() + 
    sin(norm) * skew_symm(vector_norm));
}