#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Relative Pose EKF Python Test
#

from __future__ import division, print_function, absolute_import
from cmath import pi

# Import libraries
import sys, copy, threading, math
import rospy
import numpy as np
from scipy.linalg import block_diag
from quaternion_helper import quaternion_exp, skew_symm, quaternion_log
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_about_axis, quaternion_conjugate, rotation_matrix

# Import message types
from geometry_msgs.msg import Point, PointStamped, Vector3, Quaternion, PoseWithCovariance, PoseWithCovarianceStamped, Pose
from sensor_msgs.msg import Imu
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Float64, Header

class RelativePoseEKF(object):
    "Relative Pose EKF Class"
    def __init__(self,update_freq,measurement_freq):
        
        # Storage
        # Inputs
        self.IMU_msg = Imu()
        self.apriltag_msg = AprilTagDetectionArray()
        self.measurement_ready = False

        # Outputs
        self.rel_pose_msg = PoseWithCovarianceStamped()
        self.IMU_bias_msg = Imu()

        # Locks
        self.state_lock = threading.Lock()
        self.imu_lock = threading.Lock()
        self.apriltag_lock = threading.Lock()

        # Filter Parameters
        self.update_freq = update_freq
        self.dT = 1/update_freq
        self.measurement_freq = measurement_freq
        self.upd_per_meas = math.ceil(update_freq/measurement_freq)

        # State Storage
        self.r_nom = np.zeros((3,1))
        self.v_nom = np.zeros((3,1))
        self.q_nom = np.zeros(4)
        self.ab_nom = np.zeros((3,1))
        self.wb_nom = np.zeros((3,1))
        self.cov_pert = np.zeros((15,15))

        # Process and Measurement Noises
        self.r_cov_init = 0.1
        self.v_cov_init = 0.1
        self.ang_cov_init = 0.15
        self.ab_cov_init = 0.5
        self.wb_cov_init = 0.1

        self.cov_init = np.diag(np.hstack((self.r_cov_init*np.ones(3),self.v_cov_init*np.ones(3),
                        self.ang_cov_init*np.ones(3),self.ab_cov_init*np.ones(3),self.wb_cov_init*np.ones(3))))

        self.Q_a = 10000*np.eye(3)
        self.Q_w = 10000*np.eye(3)
        self.Q_ab = 10000*np.eye(3)
        self.Q_wb = 10000*np.eye(3)

        self.Q = block_diag(self.Q_a,self.Q_w,self.Q_ab,self.Q_wb)

        self.R_r = 0.015*np.eye(3)
        self.R_ang = 0.05*np.eye(3)

        self.R = block_diag(self.R_r,self.R_ang)

        # Camera to Vehicle Transform
        self.r_v_cv = np.array([[0],[0],[-0.073]])
        self.q_vc = np.array([0.70711,-0.70711,0,0])
        self.C_vc = quaternion_matrix(self.q_vc)[0:3,0:3]

        # Counters/flags
        self.state_initialized = False
        self.upds_since_correction = 0

        # Tolerances and Constants
        self.small_ang_tol = 1E-10
        self.g = np.array([[0],[0],[-9.81]])

    def filter_update(self):
        "Perform EKF update"

        if not self.state_initialized: return

        # Clamp data for this update, decide if correction happens this step
        self.imu_lock.acquire()
        self.apriltag_lock.acquire()

        imu_curr = self.IMU_msg
        if self.measurement_ready and (self.upds_since_correction+1)>=self.upd_per_meas:
            perform_correction = True
            meas_curr = self.apriltag_msg
            self.measurement_ready = False
        else:
            perform_correction = False

        self.imu_lock.release()
        self.apriltag_lock.release()

        # Prediction step
        a_nom = (np.array([imu_curr.linear_acceleration.x,imu_curr.linear_acceleration.y,
                            imu_curr.linear_acceleration.z]).reshape((3,1))-self.ab_nom)
        w_nom = (np.array([imu_curr.angular_velocity.x,imu_curr.angular_velocity.y,
                            imu_curr.angular_velocity.z]).reshape((3,1))-self.wb_nom)
        C_nom = quaternion_matrix(self.q_nom)[0:3,0:3]

        # Propagate nominal state
        r_check = self.r_nom + self.dT*self.v_nom
        v_check = self.v_nom + self.dT*(np.dot(C_nom,a_nom) + self.g)
        q_check = quaternion_multiply(self.q_nom,quaternion_exp(self.dT*w_nom.flatten()))
        ab_check = self.ab_nom
        wb_check = self.wb_nom

        # Calculate Jacobians
        F_km1 = np.eye(15)
        F_km1[0:3,3:6] = self.dT*np.eye(3)
        F_km1[3:6,6:9] = -self.dT*np.dot(C_nom,skew_symm(a_nom))
        F_km1[3:6,9:12] = -self.dT*C_nom

        w_int = self.dT*w_nom
        w_int_angle = np.linalg.norm(w_int)
        if w_int_angle<self.small_ang_tol:
            F_km1[6:9,6:9] = np.eye(3)
        else:
            w_int_axis = w_int/w_int_angle
            F_km1[6:9,6:9] = rotation_matrix(-w_int_angle,w_int_axis)[0:3,0:3]
        F_km1[6:9,12:15] = -self.dT*np.eye(3)
        
        W_km1 = block_diag(np.vstack((np.zeros((3,3)),-C_nom)),np.eye(9))

        # Propagate covariance
        P_check = np.linalg.multi_dot((F_km1,self.cov_pert,F_km1.T)) + np.linalg.multi_dot((W_km1,self.Q,W_km1.T))

        if perform_correction:
            # Fuse motion model prediction with AprilTag readings
            r_c_tc = np.array([[meas_curr.detections[0].pose.pose.pose.position.x],
                                [meas_curr.detections[0].pose.pose.pose.position.y],
                                [meas_curr.detections[0].pose.pose.pose.position.z]])
            q_ct = np.array([meas_curr.detections[0].pose.pose.pose.orientation.x,
                                meas_curr.detections[0].pose.pose.pose.orientation.y,
                                meas_curr.detections[0].pose.pose.pose.orientation.z,
                                meas_curr.detections[0].pose.pose.pose.orientation.w])
            
            # Convert AprilTag readings to vehicle state coordinates
            C_check = quaternion_matrix(q_check)[0:3,0:3]
            r_t_vt_obs = -np.linalg.multi_dot((C_check,self.C_vc,r_c_tc))-np.dot(C_check,self.r_v_cv)
            q_tv_obs = quaternion_conjugate(quaternion_multiply(self.q_vc,q_ct))

            # Calculate observed perturbations in measurements
            delta_r_obs = r_t_vt_obs - r_check
            delta_q_obs = quaternion_multiply(quaternion_conjugate(q_check),q_tv_obs)
            delta_theta_obs = quaternion_log(delta_q_obs).reshape((3,1))

            # Calculate Jacobians
            G_k = np.zeros((6,15))
            G_k[0:3,0:3] = np.eye(3)
            G_k[0:3,6:9] = -np.dot(C_check,skew_symm(np.dot(C_check.T,r_check)))
            G_k[3:6,6:9] = np.eye(3)

            N_k = block_diag(-np.dot(C_check,self.C_vc),self.C_vc)

            R_k = np.linalg.multi_dot((N_k,self.R,N_k.T))

            # Form Kalman Gain and execute correction step
            Cov_meas_inv = np.linalg.inv(np.linalg.multi_dot((G_k,P_check,G_k.T))+R_k)
            K_k = np.linalg.multi_dot((P_check,G_k.T,Cov_meas_inv))

            P_hat = np.dot(np.eye(15)-np.dot(K_k,G_k),P_check)
            delta_x_hat = np.dot(K_k,np.vstack((delta_r_obs,delta_theta_obs)))

            # Inject correction update, store and reset error state
            self.r_nom = r_check + delta_x_hat[0:3,0:1]
            self.v_nom = v_check + delta_x_hat[3:6,0:1]
            self.q_nom = quaternion_multiply(q_check,quaternion_exp(delta_x_hat[6:9,0:1].flatten()))
            self.ab_nom = ab_check + delta_x_hat[9:12,0:1]
            self.wb_nom = wb_check + delta_x_hat[12:15,0:1]
            
            self.cov_pert = P_hat
            self.upds_since_correction = 0
        else:
            # Predictor mode only
            self.r_nom = r_check
            self.v_nom = v_check
            self.q_nom = q_check
            self.ab_nom = ab_check
            self.wb_nom = wb_check

            self.cov_pert = P_check

            self.upds_since_correction += 1

        # Pack estimate up into messages
        curr_header = Header(stamp=rospy.get_rostime())
        pose_cov = np.vstack((np.hstack((self.cov_pert[0:3,0:3],self.cov_pert[0:3,6:9])),
                            np.hstack((self.cov_pert[6:9,0:3],self.cov_pert[6:9,6:9]))))
        new_rel_pose = PoseWithCovariance(pose = Pose(position = Point(x=self.r_nom[0],y=self.r_nom[1],z=self.r_nom[2]),
                            orientation = Quaternion(x=self.q_nom[0],y=self.q_nom[1],z=self.q_nom[2],w=self.q_nom[3])),
                            covariance = pose_cov.flatten().tolist())
        self.rel_pose_msg = PoseWithCovarianceStamped(header = curr_header, pose = new_rel_pose)
        self.IMU_bias_msg = Imu(header=curr_header,
                                linear_acceleration = Vector3(x=self.ab_nom[0],y=self.ab_nom[1],z=self.ab_nom[2]),
                                angular_velocity = Vector3(x=self.wb_nom[0],y=self.wb_nom[1],z=self.wb_nom[2]))

    def initialize_state(self):
        "Initialize state to last received AprilTag relative pose detection"
        self.state_lock.acquire()
        self.r_nom[0] = self.apriltag_msg.detections[0].pose.pose.pose.position.x
        self.r_nom[1] = self.apriltag_msg.detections[0].pose.pose.pose.position.y
        self.r_nom[2] = self.apriltag_msg.detections[0].pose.pose.pose.position.z
        self.v_nom = np.zeros((3,1))
        self.q_nom[0] = self.apriltag_msg.detections[0].pose.pose.pose.orientation.x
        self.q_nom[1] = self.apriltag_msg.detections[0].pose.pose.pose.orientation.y
        self.q_nom[2] = self.apriltag_msg.detections[0].pose.pose.pose.orientation.z
        self.q_nom[3] = self.apriltag_msg.detections[0].pose.pose.pose.orientation.w

        self.cov_pert = self.cov_init

        self.state_initialized = True
        self.state_lock.release()