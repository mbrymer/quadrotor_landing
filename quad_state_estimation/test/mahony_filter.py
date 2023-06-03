#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Mahony Attitude Filter
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

class MahonyFilter(object):
    "Mahony Attitude Filter Class"
    def __init__(self, update_freq):

        # Storage
        # Inputs
        self.imu_msg = Imu()

        # Outputs
        self.pose_msg = PoseStamped()
        self.imu_bias_msg = Imu()

        # Locks
        self.lock = threading.Lock()

        # Parameters
        self.update_freq = update_freq
        self.dT = 1 / update_freq
        self.kp = 0.1
        self.ki = 0.05
        self.g = np.array([[0],[0],[-9.81]])
        self.g_norm = self.g / np.linalg.norm(self.g)

        # State
        self.q = identity_quaternion()
        # self.b = np.zeros((3,1))
        self.b = np.array([[-0.002],[0.006],[0]])
        self.imu_bias = np.array([[-0.205],[-0.17],[0.065]])

        self.num_times_called = 0

        # Locks
        self.imu_lock = threading.Lock()

        # Topics
        self.pose_frame_name = "drone/mahony_pose_est"

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

        self.lock.acquire()
        C_hat = quaternion_matrix(self.q)[0:3,0:3]
        accel_hat_norm = -np.dot(C_hat.T, self.g_norm)
        accel_meas_norm = (accel_meas - self.imu_bias) / np.linalg.norm(accel_meas)
        omega_meas = -vex_symm(1 / 2 * (np.dot(accel_meas_norm, accel_hat_norm.T) - np.dot(accel_hat_norm, accel_meas_norm.T)))

        omega_filter = w_meas - self.b + self.kp * omega_meas

        self.q = quaternion_norm(quaternion_multiply(self.q, quaternion_exp(self.dT * omega_filter.flatten())))
        self.b = self.b - self.dT * self.ki * omega_meas
        self.lock.release()

        # Update output message
        curr_time = rospy.get_rostime()
        curr_header = Header(stamp=curr_time,frame_id=self.pose_frame_name)

        self.pose_msg = PoseStamped(header = curr_header,
                            pose = Pose(position = Point(),
                            orientation = Quaternion(x = self.q[0], y = self.q[1], z = self.q[2], w = self.q[3])))
        self.imu_bias_msg = Imu(header=curr_header,
                                linear_acceleration = Vector3(),
                                angular_velocity = Vector3(x = self.b[0], y = self.b[1], z = self.b[2]))
        
        self.num_times_called += 1

    def get_quaternion(self):
        self.lock.acquire()
        quaternion_copy = copy.copy(self.q)
        self.lock.release()
        return quaternion_copy

