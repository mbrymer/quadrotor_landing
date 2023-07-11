/*
    AER 1810 Quadrotor Landing Project
    Relative Pose Utils
*/

#pragma once

#include <relative_pose_utils.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <math.h>
#include <algorithm>
#include <vector>

using Eigen::seq;
typedef Eigen::LLT<Eigen::MatrixXd> Cholesky;

Eigen::MatrixXd DrawSigmaPoints(const Eigen::VectorXd& mean_state, const Eigen::MatrixXd& covariance_state,
                                const Eigen::MatrixXd& covariance_noise_cholesky, double kappa){
    // Compute Cholesky decomposition of state covariance
    Cholesky state_cholesky(covariance_state);

    const double n_state = mean_state.size();
    const double n_noise = covariance_noise_cholesky.rows();
    const double n = n_state + n_noise;
    const double scatter_factor = sqrt(n + kappa);

    Eigen::MatrixXd sigma_points = Eigen::MatrixXd::Zero(n, 2 * n + 1);
    Eigen::MatrixXd state_covariance_sigma_points(state_cholesky.matrixL());

    // Populate sigma points as [mean, + state covariance, + noise covariance, -state covariance, -noise covariance]
    sigma_points(seq(0, n_state - 1), 0) = mean_state;
    sigma_points(seq(0, n_state - 1), seq(1, n_state)) = (scatter_factor * state_covariance_sigma_points).colwise() + mean_state;
    sigma_points(seq(n_state, n - 1), seq(n_state + 1, n_state + n_noise)) = scatter_factor * covariance_noise_cholesky;
    sigma_points(seq(0, n_state - 1),
    seq(1 + n_state + n_noise, 2 * n_state + n_noise)) = (-scatter_factor * state_covariance_sigma_points).colwise() + mean_state;
    sigma_points(seq(n_state, n - 1),
    seq(2 * n_state + n_noise + 1, 2 * n_state + 2 * n_noise)) = - scatter_factor * covariance_noise_cholesky;

    return sigma_points;
}

Eigen::MatrixXd DrawSigmaPointsZeroMean(const Eigen::MatrixXd& covariance_state,
                                const Eigen::MatrixXd& covariance_noise_cholesky, double kappa){
    return DrawSigmaPoints(Eigen::VectorXd::Zero(covariance_state.rows()),
                            covariance_state, covariance_noise_cholesky, kappa);
}