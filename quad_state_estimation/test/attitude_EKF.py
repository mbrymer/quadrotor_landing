#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Attitude EKF
#

from __future__ import division, print_function, absolute_import
from cmath import pi

# Import libraries
import sys, copy, threading, math
import rospy
import numpy as np
import tf

from scipy.linalg import block_diag
from quaternion_helper import quaternion_exp, skew_symm, quaternion_log, quaternion_norm, vex_symm, identity_quaternion
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_about_axis, quaternion_conjugate, rotation_matrix

# Import message types
from geometry_msgs.msg import Point, PointStamped, Vector3, Vector3Stamped, Quaternion, PoseWithCovariance, PoseWithCovarianceStamped, Pose, PoseStamped
from sensor_msgs.msg import Imu
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Float64, Header

class AttitudeEKF(object):
    "Attitude EKF Class"
    def __init__(self, update_freq):

        # Storage
        # Inputs
        self.imu_msg = Imu()

        # Outputs
        self.pose_msg = PoseStamped()

        # Parameters
        self.update_freq = update_freq
        self.dT = 1 / update_freq

        self.Q_w = 0.00005 * np.eye(3)
        self.R_acc = 5 * np.eye(3)

        self.e_3 = np.array([[0],[0],[1]])
        self.g = 9.81

        # State
        self.q_nom = identity_quaternion()
        self.cov_pert = 0.15 * np.eye(3)
        self.imu_bias = np.array([[-0.205],[-0.17],[0.065]])

        # Tolerances and constants
        self.small_ang_tol = 1E-10

        # Locks
        self.imu_lock = threading.Lock()

        # Topics
        self.pose_frame_name = "drone/attitude_EKF"

    def filter_update(self):
        # Perform filter update
        self.imu_lock.acquire()
        accel_meas = np.array([self.imu_msg.linear_acceleration.x, self.imu_msg.linear_acceleration.y,
                            self.imu_msg.linear_acceleration.z]).reshape((3,1))
        w_meas = np.array([self.imu_msg.angular_velocity.x, self.imu_msg.angular_velocity.y,
                            self.imu_msg.angular_velocity.z]).reshape((3,1))
        self.imu_lock.release()

        if(np.linalg.norm(accel_meas) < 0.01):
            # Lazy way to skip bad inputs
            return

        # Prediction step
        q_check = quaternion_norm(quaternion_multiply(self.q_nom, quaternion_exp(self.dT * w_meas.flatten())))

        w_int = self.dT * w_meas
        w_int_angle = np.linalg.norm(w_int)
        if w_int_angle < self.small_ang_tol:
            # Avoid zero division, approximate with first order Taylor series
            F_km1 = np.eye(3) - skew_symm(w_int.flatten())
        else:
            # Large angle, use full Rodriguez formula
            w_int_axis = w_int / w_int_angle
            F_km1 = rotation_matrix(-w_int_angle, w_int_axis)[0:3,0:3]

        P_check = np.linalg.multi_dot((F_km1, self.cov_pert, F_km1.T)) + self.Q_w

        # Correction step
        C_check = quaternion_matrix(q_check)[0:3,0:3]
        accel_no_bias = accel_meas - self.imu_bias

        y_a = accel_no_bias / np.linalg.norm(accel_no_bias)

        G_k = skew_symm(np.dot(C_check.T, self.e_3))
        R_k = self.R_acc

        delta_y_obs = y_a - np.dot(C_check.T, self.e_3)

        Cov_meas_inv = np.linalg.inv(np.linalg.multi_dot((G_k, P_check, G_k.T)) + R_k)
        K_k = np.linalg.multi_dot((P_check, G_k.T, Cov_meas_inv))

        P_hat = np.dot(np.eye(3) - np.dot(K_k, G_k), P_check)
        delta_x_hat = np.dot(K_k, delta_y_obs)

        # Perform injection, reset filter
        q_hat = quaternion_norm(quaternion_multiply(q_check, quaternion_exp(delta_x_hat.flatten())))

        self.q_nom = q_hat
        self.cov_pert = P_hat

        # Update output message
        curr_time = rospy.get_rostime()
        curr_header = Header(stamp=curr_time,frame_id=self.pose_frame_name)

        self.pose_msg = PoseStamped(header = curr_header,
                            pose = Pose(position = Point(),
                            orientation = Quaternion(x = self.q_nom[0], y = self.q_nom[1], z = self.q_nom[2], w = self.q_nom[3])))
        

