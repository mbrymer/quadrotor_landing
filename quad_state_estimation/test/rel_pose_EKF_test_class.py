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
import tf

from scipy.linalg import block_diag
from quaternion_helper import quaternion_exp, skew_symm, quaternion_log, quaternion_norm
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_about_axis, quaternion_conjugate, rotation_matrix

# Import message types
from geometry_msgs.msg import Point, PointStamped, Vector3, Vector3Stamped, Quaternion, PoseWithCovariance, PoseWithCovarianceStamped, Pose, PoseStamped
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
        self.magnetometer_msg = Vector3Stamped()

        # Outputs
        self.rel_pose_msg = PoseWithCovarianceStamped()
        self.rel_vel_msg = Vector3Stamped()
        self.rel_accel_msg = Vector3Stamped()
        self.rel_pose_report_msg = PoseStamped()
        self.IMU_bias_msg = Imu()
        self.pred_length_msg = PointStamped()

        self.tf_broadcast = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        # Locks
        self.state_lock = threading.Lock()
        self.imu_lock = threading.Lock()
        self.apriltag_lock = threading.Lock()
        self.magnetometer_lock = threading.Lock()

        # Filter Parameters
        self.update_freq = update_freq
        self.dT = 1/update_freq
        self.measurement_freq = measurement_freq
        self.upd_per_meas = math.ceil(update_freq/measurement_freq)
        self.est_bias = True
        self.accel_orien_corr = False
        self.use_magnetometer = False

        # State Storage
        if self.est_bias:
            self.num_states = 15
        else:
            self.num_states = 9
        self.r_nom = np.zeros((3,1))
        self.v_nom = np.zeros((3,1))
        self.q_nom = np.zeros(4)
        self.ab_nom = np.zeros((3,1))
        self.wb_nom = np.zeros((3,1))
        self.cov_pert = np.zeros((self.num_states,self.num_states))

        # Process and Measurement Noises
        self.r_cov_init = 0.1
        self.v_cov_init = 0.1
        self.ang_cov_init = 0.15
        self.ab_cov_init = 0.5
        self.wb_cov_init = 0.1

        # self.Q_a = 0.01*np.eye(3)
        # self.Q_w = 0.01*np.eye(3)
        self.Q_a = 0.005*np.eye(3)
        self.Q_w = 0.0005*np.eye(3)
        self.Q_ab = 5E-5*np.eye(3)
        self.Q_wb = 5E-6*np.eye(3)

        if self.est_bias:
            self.cov_init = np.diag(np.hstack((self.r_cov_init*np.ones(3),self.v_cov_init*np.ones(3),
                        self.ang_cov_init*np.ones(3),self.ab_cov_init*np.ones(3),self.wb_cov_init*np.ones(3))))
            self.Q = block_diag(self.Q_a,self.Q_w,self.Q_ab,self.Q_wb)
        else:
            self.cov_init = np.diag(np.hstack((self.r_cov_init*np.ones(3),self.v_cov_init*np.ones(3),
                        self.ang_cov_init*np.ones(3))))
            self.Q = block_diag(self.Q_a,self.Q_w)

        # self.R_r = 0.015*np.eye(3)
        # self.R_ang = 0.05*np.eye(3)
        self.R_r = np.diag(np.array([0.005,0.005,0.015]))
        self.R_ang = np.diag(np.array([0.0025,0.0025,0.025]))
        self.R_acc = 0.05*np.eye(3)
        self.R_mag = 0.005*np.eye(3)

        self.b_0 = np.array([[1],[0],[0]]) # Reference magnetic field direction

        if self.accel_orien_corr:
            self.R = block_diag(self.R_r,self.R_ang,self.R_acc)
        elif self.use_magnetometer:
            self.R = block_diag(self.R_r,self.R_ang,self.R_mag)
        else:
            self.R = block_diag(self.R_r,self.R_ang)

        # Camera to Vehicle Transform
        self.r_v_cv = np.array([[0],[0],[-0.073]])
        self.q_vc = quaternion_norm(np.array([0.70711,-0.70711,0,0]))
        self.C_vc = quaternion_matrix(self.q_vc)[0:3,0:3]

        # Camera and Tag Parameters
        self.camera_K = np.array([[241.4268,0,376.5],
                                    [0,241.4268,240.5],
                                    [0,0,1]])
        self.camera_width = 752
        self.camera_height = 480
        self.tag_width = 0.8 # m
        self.tag_in_view_margin = 0.02 # %

        self.tag_corners = np.array([[self.tag_width/2,-self.tag_width/2,-self.tag_width/2,self.tag_width/2],
                                    [self.tag_width/2,self.tag_width/2,-self.tag_width/2,-self.tag_width/2],
                                    [0,0,0,0],
                                    [1,1,1,1]])

        # Counters/flags
        self.state_initialized = False
        self.measurement_ready = False
        self.upds_since_correction = 0

        # Tolerances and Constants
        self.small_ang_tol = 1E-10
        self.g = np.array([[0],[0],[-9.8]])
        self.pose_frame_name = "drone/rel_pose_est"
        self.pose_report_frame_name = "drone/rel_pose_report"
        self.pose_rel_frame_name = "target/tag_link"

    def filter_update(self):
        "Perform EKF update"

        if not self.state_initialized: return

        # Clamp data for this update, decide if correction happens this step
        self.imu_lock.acquire()
        self.magnetometer_lock.acquire()
        self.apriltag_lock.acquire()

        imu_curr = self.IMU_msg
        mag_curr = self.magnetometer_msg
        if self.measurement_ready and (self.upds_since_correction+1)>=self.upd_per_meas:
            # Extract AprilTag reading, build tag pose matrix
            meas_curr = self.apriltag_msg
            self.measurement_ready = False
            r_c_tc = np.array([[meas_curr.detections[0].pose.pose.pose.position.x],
                                [meas_curr.detections[0].pose.pose.pose.position.y],
                                [meas_curr.detections[0].pose.pose.pose.position.z]])
            q_ct = np.array([meas_curr.detections[0].pose.pose.pose.orientation.x,
                                meas_curr.detections[0].pose.pose.pose.orientation.y,
                                meas_curr.detections[0].pose.pose.pose.orientation.z,
                                meas_curr.detections[0].pose.pose.pose.orientation.w])
            T_ct = quaternion_matrix(q_ct)
            T_ct[0:3,3:4] = r_c_tc
            T_ct[3,3] = 1

            # Check criteria for a "good" detection
            # Project tag corners into image, verify they have some margin to the edge of the image to guard against spurious detections
            tag_corners_c = np.dot(T_ct,self.tag_corners)
            tag_corners_c_n = tag_corners_c / tag_corners_c[2,:]
            tag_corners_px = np.dot(self.camera_K,tag_corners_c_n[0:3,:])

            min_px = np.min(tag_corners_px[0:2,:],axis=1)
            max_px = np.max(tag_corners_px[0:2,:],axis=1)

            perform_correction = (min_px[0]>self.camera_width*self.tag_in_view_margin and 
                                min_px[1]>self.camera_height*self.tag_in_view_margin and
                                max_px[0]<self.camera_width*(1-self.tag_in_view_margin) and 
                                max_px[1]<self.camera_height*(1-self.tag_in_view_margin))

        else:
            perform_correction = False

        self.imu_lock.release()
        self.magnetometer_lock.release()
        self.apriltag_lock.release()

        # Prediction step
        a_nom = (np.array([imu_curr.linear_acceleration.x,imu_curr.linear_acceleration.y,
                            imu_curr.linear_acceleration.z]).reshape((3,1))-self.ab_nom)
        w_nom = (np.array([imu_curr.angular_velocity.x,imu_curr.angular_velocity.y,
                            imu_curr.angular_velocity.z]).reshape((3,1))-self.wb_nom)
        C_nom = quaternion_matrix(self.q_nom)[0:3,0:3]

        accel_rel = np.dot(C_nom,a_nom) + self.g

        # Propagate nominal state
        r_check = self.r_nom + self.dT*self.v_nom
        v_check = self.v_nom + self.dT*accel_rel
        q_check = quaternion_norm(quaternion_multiply(self.q_nom,quaternion_exp(self.dT*w_nom.flatten())))
        ab_check = self.ab_nom
        wb_check = self.wb_nom

        # Calculate Jacobians
        F_km1 = np.eye(self.num_states)
        F_km1[0:3,3:6] = self.dT*np.eye(3)
        F_km1[3:6,6:9] = -self.dT*np.dot(C_nom,skew_symm(a_nom.flatten()))
        
        w_int = self.dT*w_nom
        w_int_angle = np.linalg.norm(w_int)
        if w_int_angle<self.small_ang_tol:
            # Avoid zero division, approximate with first order Taylor series
            F_km1[6:9,6:9] = np.eye(3)-skew_symm(w_int.flatten())
        else:
            # Large angle, use full Rodriguez formula
            w_int_axis = w_int/w_int_angle
            F_km1[6:9,6:9] = rotation_matrix(-w_int_angle,w_int_axis)[0:3,0:3]
        
        if self.est_bias:
            F_km1[3:6,9:12] = -self.dT*C_nom
            F_km1[6:9,12:15] = -self.dT*np.eye(3)
        
            W_km1 = block_diag(np.vstack((np.zeros((3,3)),-C_nom)),np.eye(9))
        else:
            W_km1 = block_diag(np.vstack((np.zeros((3,3)),-C_nom)),np.eye(3))

        # Propagate covariance
        P_check = np.linalg.multi_dot((F_km1,self.cov_pert,F_km1.T)) + np.linalg.multi_dot((W_km1,self.Q,W_km1.T))

        if perform_correction:
            # Fuse motion model prediction with AprilTag readings
            # Convert AprilTag readings to vehicle state coordinates
            C_check = quaternion_matrix(q_check)[0:3,0:3]
            r_t_vt_obs = -np.linalg.multi_dot((C_check,self.C_vc,r_c_tc))-np.dot(C_check,self.r_v_cv)
            q_tv_obs = quaternion_norm(quaternion_conjugate(quaternion_multiply(self.q_vc,q_ct)))
            a_obs = np.array([imu_curr.linear_acceleration.x,imu_curr.linear_acceleration.y,
                            imu_curr.linear_acceleration.z]).reshape((3,1))
            b_obs = np.array([mag_curr.vector.x,mag_curr.vector.y,mag_curr.vector.z]).reshape((3,1))

            # Calculate observed perturbations in measurements
            delta_r_obs = r_t_vt_obs - r_check
            delta_q_obs = quaternion_multiply(quaternion_conjugate(q_check),q_tv_obs)
            delta_theta_obs = quaternion_log(delta_q_obs).reshape((3,1))
            delta_a_obs = a_obs - (-np.dot(C_check.T,self.g) + ab_check)
            delta_b_obs = b_obs - np.dot(C_check.T,self.b_0)

            # Calculate Jacobians
            if self.accel_orien_corr:
                G_k = np.zeros((9,self.num_states))

                G_k[6:9,6:9] = -skew_symm(np.dot(C_check.T,self.g))
                G_k[6:9,9:12] = np.eye(3)
                N_k = block_diag(-np.dot(C_check,self.C_vc),self.C_vc,np.eye(3))
            elif self.use_magnetometer:
                G_k = np.zeros((9,self.num_states))

                G_k[6:9,6:9] = skew_symm(np.dot(C_check.T,self.b_0))
                N_k = block_diag(-np.dot(C_check,self.C_vc),self.C_vc,np.eye(3))
            else:
                G_k = np.zeros((6,self.num_states))
                N_k = block_diag(-np.dot(C_check,self.C_vc),self.C_vc)

            G_k[0:3,0:3] = np.eye(3)
            G_k[0:3,6:9] = np.dot(C_check,skew_symm(np.dot(C_check.T,r_check)))
            G_k[3:6,6:9] = np.eye(3)

            R_k = np.linalg.multi_dot((N_k,self.R,N_k.T))

            # Form Kalman Gain and execute correction step
            Cov_meas_inv = np.linalg.inv(np.linalg.multi_dot((G_k,P_check,G_k.T))+R_k)
            K_k = np.linalg.multi_dot((P_check,G_k.T,Cov_meas_inv))

            if self.accel_orien_corr:
                delta_y_obs = np.vstack((delta_r_obs,delta_theta_obs,delta_a_obs))
            elif self.use_magnetometer:
                delta_y_obs = np.vstack((delta_r_obs,delta_theta_obs,delta_b_obs))
            else:
                delta_y_obs = np.vstack((delta_r_obs,delta_theta_obs))
            
            P_hat = np.dot(np.eye(self.num_states)-np.dot(K_k,G_k),P_check)
            delta_x_hat = np.dot(K_k,delta_y_obs)

            # Inject correction update, store and reset error state
            self.r_nom = r_check + delta_x_hat[0:3,0:1]
            self.v_nom = v_check + delta_x_hat[3:6,0:1]
            self.q_nom = quaternion_norm(quaternion_multiply(q_check,quaternion_exp(delta_x_hat[6:9].flatten())))

            if self.est_bias:
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

        # Pack estimate up into messages and publish transform
        curr_time = rospy.get_rostime()
        curr_header = Header(stamp=curr_time,frame_id=self.pose_frame_name)
        pose_cov = np.vstack((np.hstack((self.cov_pert[0:3,0:3],self.cov_pert[0:3,6:9])),
                            np.hstack((self.cov_pert[6:9,0:3],self.cov_pert[6:9,6:9]))))
        new_rel_pose = PoseWithCovariance(pose = Pose(position = Point(x=self.r_nom[0],y=self.r_nom[1],z=self.r_nom[2]),
                            orientation = Quaternion(x=self.q_nom[0],y=self.q_nom[1],z=self.q_nom[2],w=self.q_nom[3])),
                            covariance = pose_cov.flatten().tolist())
        self.rel_pose_msg = PoseWithCovarianceStamped(header = curr_header, pose = new_rel_pose)
        self.IMU_bias_msg = Imu(header=curr_header,
                                linear_acceleration = Vector3(x=self.ab_nom[0],y=self.ab_nom[1],z=self.ab_nom[2]),
                                angular_velocity = Vector3(x=self.wb_nom[0],y=self.wb_nom[1],z=self.wb_nom[2]))
        self.rel_vel_msg = Vector3Stamped(header=curr_header, vector = Vector3(x=self.v_nom[0], y=self.v_nom[1], z = self.v_nom[2]))
        self.rel_accel_msg = Vector3Stamped(header=curr_header, vector = Vector3(x=accel_rel[0], y=accel_rel[1], z = accel_rel[2]))
        self.pred_length_msg = PointStamped(header=curr_header,point = Point(x=self.upds_since_correction))
        self.tf_broadcast.sendTransform((tuple(self.r_nom)), (tuple(self.q_nom)), curr_time,self.pose_frame_name,self.pose_rel_frame_name)

        if perform_correction:
            self.rel_pose_report_msg = PoseStamped(header=curr_header,
            pose = Pose(position = Point(x=r_t_vt_obs[0],y=r_t_vt_obs[1],z=r_t_vt_obs[2]),
                            orientation = Quaternion(x=q_tv_obs[0],y=q_tv_obs[1],z=q_tv_obs[2],w=q_tv_obs[3])))
            self.tf_broadcast.sendTransform((tuple(r_t_vt_obs)), (tuple(q_tv_obs)), curr_time,self.pose_report_frame_name,self.pose_rel_frame_name)

    def initialize_state(self,reinit_bias):
        "Initialize state to last received AprilTag relative pose detection"
        self.state_lock.acquire()
        
        # Assume identity rotation to start. Will need to initialize this based on GPS in real implementation for stability most likely
        r_c_tc = np.array([[self.apriltag_msg.detections[0].pose.pose.pose.position.x],
                                [self.apriltag_msg.detections[0].pose.pose.pose.position.y],
                                [self.apriltag_msg.detections[0].pose.pose.pose.position.z]])
        q_ct = np.array([self.apriltag_msg.detections[0].pose.pose.pose.orientation.x,
                                self.apriltag_msg.detections[0].pose.pose.pose.orientation.y,
                                self.apriltag_msg.detections[0].pose.pose.pose.orientation.z,
                                self.apriltag_msg.detections[0].pose.pose.pose.orientation.w])
        
        self.r_nom = -np.dot(self.C_vc,r_c_tc)-self.r_v_cv
        self.v_nom = np.zeros((3,1))
        self.q_nom = quaternion_norm(quaternion_conjugate(quaternion_multiply(self.q_vc,q_ct)))

        if reinit_bias:
            self.ab_nom = 0
            self.wb_nom = 0

        self.cov_pert = self.cov_init

        self.state_initialized = True
        self.state_lock.release()