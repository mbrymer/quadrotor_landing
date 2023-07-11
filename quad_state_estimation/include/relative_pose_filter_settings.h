/*
    AER 1810 Quadrotor Landing Project
    Relative Pose Filter Settings
*/

#pragma once

constexpr double kDefaultLoopRate{100};
constexpr double kDefaultDt{1 / kDefaultLoopRate};
constexpr double kMahonyDefaultKp{0.1};
constexpr double kMahonyDefaultKi{0.05};
constexpr double kSigmaPointKappa{2.0};
constexpr double kSmallAngleTolerance{1E-10};
constexpr double kGravitationalConstant{9.81};

enum StateIndex{
    X = 0,
    Y = 1,
    Z = 2,
    Vx = 3,
    Vy = 4,
    Vz = 5,
    q_x = 6,
    q_y = 7,
    q_z = 8,
    q_w = 9,
    ab_x = 10,
    ab_y = 11,
    ab_z = 12,
    wb_x = 13,
    wb_y = 14,
    wb_z = 15,
    NumStates = 16,
};

enum PertIndex{
    X = 0,
    Y = 1,
    Z = 2,
    Vx = 3,
    Vy = 4,
    Vz = 5,
    theta_x = 6,
    theta_y = 7,
    theta_z = 8,
    ab_x = 9,
    ab_y = 10,
    ab_z = 11,
    wb_x = 12,
    wb_y = 13,
    wb_z = 14,
    NumPertStates = 15,
};

enum ProcessNoise{
    a_nX = 15,
    a_nY = 16,
    a_nZ = 17,
    w_nX = 18,
    w_nY = 19,
    w_nZ = 20,
    ab_nX = 21,
    ab_nY = 22,
    ab_nZ = 23,
    wb_nX = 24,
    wb_nY = 25,
    wb_nZ = 26,
};

enum MahonyMeasurementIndex{
    roll = 0,
    pitch = 1,
    NumAngles = 2,
};

enum AprilTagMeasurementIndex{
    r_x = 0,
    r_y = 1,
    r_z = 2,
    theta_x = 3,
    theta_y = 4,
    theta_z = 5,
    NumAprilTagIndices = 6,
};

enum GPSSpeedNoiseIndex{
    n_vx = 0,
    n_vy = 1,
    NumGPSSpeedNoise = 2,
}