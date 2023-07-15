/*
    AER 1810 Quadrotor Landing Project
    Relative Pose Utils
*/

#pragma once

#include <Eigen/Dense>

Eigen::MatrixXd DrawSigmaPoints(const Eigen::VectorXd& mean_state, const Eigen::MatrixXd& covariance_state,
                                const Eigen::MatrixXd& covariance_noise_cholesky, double kappa);

Eigen::MatrixXd DrawSigmaPointsZeroMean(const Eigen::MatrixXd& covariance_state,
                                const Eigen::MatrixXd& covariance_noise_cholesky, double kappa);
