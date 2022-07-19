/*
    AER 1810 Quadrotor Landing Project
    Quaternion Helper Functions
*/

#include <quaternion_helper.hpp>

// Quaternion exponential map from pure to unit quaternion
Eigen::Quaterniond quaternion_exp(Eigen::VectorXd pure_quat)
{
    double norm_tol = 1E-10;
    Eigen::Quaterniond unit_quat;

    double norm = pure_quat.norm();

    // Real part is fine
    unit_quat.w() = cos(norm/2);

    if (norm<norm_tol)
    {
        // Small rotation, use Taylor series approximation for vector part
        unit_quat.vec() = pure_quat/2*(1-pow(norm,2)/24);
    }
    else
    {
        // Rotation is large enough, use true rotation axis
        unit_quat.vec() = pure_quat/norm*sin(norm/2);
    }

    return unit_quat;
}

// Quaternion logarithmic map from unit to pure quaternion
Eigen::VectorXd quaternion_log(Eigen::Quaterniond unit_quat)
{
    double norm_tol = 1E-10;
    Eigen::VectorXd pure_quat = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd quat_vec = Eigen::VectorXd::Zero(3);

    quat_vec << unit_quat.x(),unit_quat.y(),unit_quat.z();
    double vec_norm = quat_vec.norm();

    if (vec_norm < norm_tol)
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
void quaternion_norm(Eigen::Quaterniond &unit_quat)
{
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
Eigen::MatrixXd skew_symm(Eigen::VectorXd vector)
{
    Eigen::MatrixXd mat(3,3);
    
    mat << 0, -vector(2), vector(1),
        vector(2), 0 , -vector(0),
        -vector(1), vector(0), 0;
    
    return mat;
}