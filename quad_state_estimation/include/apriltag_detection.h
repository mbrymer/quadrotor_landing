/*
    AER 1810 Quadrotor Landing Project
    AprilTag Detection Type
*/

#pragma once

#include <Eigen/Dense>

struct AprilTagDetection{
    AprilTagDetection(Eigen::Vector3d position_, Eigen::Quaterniond orientation_, double time_) : 
                        position(std::move(position_)), orientation(std::move(orientation_)) , time(time_) {};
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    double time;
};