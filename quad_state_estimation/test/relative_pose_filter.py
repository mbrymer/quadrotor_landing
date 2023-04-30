#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Relative Pose Filter Python Test
#

from __future__ import division, print_function, absolute_import
from cmath import pi

# Import libraries
import sys, copy, threading, math, time
import rospy
import numpy as np
import tf

from scipy.linalg import block_diag
from quaternion_helper import quaternion_exp, skew_symm, quaternion_log, quaternion_norm, exponential_map, identity_quaternion
from tf.transformations import quaternion_matrix, quaternion_multiply, quaternion_about_axis, quaternion_conjugate, rotation_matrix

# Import message types
from geometry_msgs.msg import Point, PointStamped, Vector3, Vector3Stamped, Quaternion, PoseWithCovariance, PoseWithCovarianceStamped, Pose, PoseStamped
from sensor_msgs.msg import Imu
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Float64, Header

class RelativePoseFilter(object):
    "Relative Pose Filter Class"
    def __init__(self,update_freq,measurement_freq):
        
        # Storage
        # Inputs
        self.IMU_msg = Imu()
        self.apriltag_msg = AprilTagDetectionArray()
        # self.gps_speed_msg = Vector3Stamped()

        # Outputs
        self.rel_pose_msg = PoseWithCovarianceStamped()
        self.rel_vel_msg = Vector3Stamped()
        self.rel_accel_msg = Vector3Stamped()
        self.rel_pose_report_msg = PoseStamped()
        self.IMU_bias_msg = Imu()
        self.pred_length_msg = PointStamped()
        self.mu_check_pose_msg = PoseWithCovarianceStamped()
        self.mu_check_vel_msg = Vector3Stamped()
        self.timing_msg = PointStamped()

        self.tf_broadcast = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        # Locks
        self.state_lock = threading.Lock()
        self.imu_lock = threading.Lock()
        self.apriltag_lock = threading.Lock()
        # self.gps_speed_lock = threading.Lock()

        # Filter Parameters
        self.update_freq = update_freq
        self.dT = 1 / update_freq
        self.measurement_freq = measurement_freq
        self.upd_per_meas = math.ceil(update_freq / measurement_freq)
        self.measurement_delay = 0.060 # s
        self.measurement_step_delay = max(int(round(self.measurement_delay / self.dT)), 1)
        self.use_gps = True
        self.use_mahony = True
        self.multirate_EKF = False
        self.kappa = 2

        # State Storage
        self.num_states = 15
        self.r_nom = np.zeros((3,1))
        self.v_nom = np.zeros((3,1))
        self.q_nom = identity_quaternion()
        self.ab_nom = np.zeros((3,1))
        self.wb_nom = np.zeros((3,1))
        self.cov_pert = np.zeros((self.num_states,self.num_states))

        self.ab_const = np.array([0.324, 0.078, -0.1])
        self.wb_const = np.array([-0.019, -0.013, 0])

        # Process and Measurement Noises
        self.r_cov_init = 0.1
        self.v_cov_init = 0.1
        self.ang_cov_init = 0.15
        self.ab_cov_init = 0.5
        self.wb_cov_init = 0.1

        # self.Q_a = 0.00025*np.eye(3) # Hardware values
        # self.Q_w = 0.00045*np.eye(3) # 0.0005
        # self.Q_ab = 7.0E-6*np.eye(3) # 5E-5
        # self.Q_wb = 5.0E-6*np.eye(3) # 5E-6
        self.Q_a = 0.005*np.eye(3) # Sim values
        self.Q_w = 0.0005*np.eye(3)
        self.Q_ab = 5E-5*np.eye(3)
        self.Q_wb = 5E-6*np.eye(3)

        self.cov_init = np.diag(np.hstack((self.r_cov_init*np.ones(3), self.v_cov_init*np.ones(3),
                    self.ang_cov_init * np.ones(3), self.ab_cov_init * np.ones(3), self.wb_cov_init * np.ones(3))))
        self.Q = block_diag(self.Q_a,self.Q_w,self.Q_ab,self.Q_wb)
        self.L_Q = np.linalg.cholesky(self.Q)

        # self.R_r = 0.015*np.eye(3)
        # self.R_ang = 0.05*np.eye(3)
        # self.R_r =  np.diag(np.array([0.0015,0.0015,0.006])) # Hardware values
        # self.R_ang = np.diag(np.array([0.0015,0.0015,0.04])) # 
        self.R_r = np.diag(np.array([0.005,0.005,0.015])) # Sim Values
        self.R_ang = np.diag(np.array([0.0025,0.0025,0.025]))
        self.R_mahony = 0.01 * np.eye(2)
        self.R_gps = 0.1
        
        self.R_tag = block_diag(self.R_r, self.R_ang)
        self.L_tag = np.linalg.cholesky(self.R_tag)

        # State, input, measurement and covariance histories for multi-rate EKF
        self.x_hist = [np.zeros((self.num_states + 1, 1))]
        self.mu_hist = [np.zeros((self.num_states, 1))]
        self.u_hist = [np.zeros((6, 1))]
        self.P_hist = [np.zeros((self.num_states, self.num_states))]
        self.y_hist = [np.zeros((2 * self.use_mahony + 1 * self.use_gps, 1))]

        # Camera to Vehicle Transform
        # self.r_v_cv = np.array([[0.06036412],[-0.00145196],[-0.04439579]]) # Hardware values
        self.r_v_cv = np.array([[0], [0], [-0.073]]) # Sim Values
        # self.q_vc = quaternion_norm(np.array([-0.7035177, 0.7106742, 0.0014521, -0.0017207])) # Hardware values
        self.q_vc = quaternion_norm(np.array([0.70711, -0.70711, 0, 0])) # Sim values
        self.C_vc = quaternion_matrix(self.q_vc)[0:3, 0:3]

        # Camera and Tag Parameters
        self.camera_K = np.array([[241.4268, 0, 376.5],
                                  [0, 241.4268, 240.5],
                                  [0, 0, 1]]) # Sim values
        # self.camera_K = np.array([[448.24317239815787,0,325.2550869383401],
        #                             [0,447.88113349903773,242.3517901446831],
        #                             [0,0,1]]) # Hardware values
        # self.camera_width = 640 # Hardware
        self.camera_width = 752 # Sim values
        self.camera_height = 480
        # self.tag_width = 0.199375 # Hardware # m
        self.tag_width = 0.8 # Sim
        self.tag_in_view_margin = 0.02 # %

        self.tag_corners = np.array([[self.tag_width/2, -self.tag_width/2, -self.tag_width/2, self.tag_width/2],
                                    [self.tag_width/2, self.tag_width/2, -self.tag_width/2, -self.tag_width/2],
                                    [0, 0, 0, 0],
                                    [1, 1, 1, 1]])

        # Counters/flags
        self.state_initialized = False
        self.apriltag_ready = False
        self.filter_run_once = False
        self.upds_since_correction = 0
        self.pub_counter = 0

        # Tolerances and Constants
        self.g = np.array([[0], [0], [-9.8]])
        self.pose_frame_name = "drone/rel_pose_est"
        self.pose_report_frame_name = "drone/rel_pose_report"
        self.pose_rel_frame_name = "target/tag_link"

        # Timing
        self.time_correction = 0.0
        self.time_filter_recalculate = 0.0
        self.time_prediction = 0.0

    def filter_update(self):
        "Perform relative pose filter update"

        if not self.state_initialized: return

        tic = time.time()

        # Clamp data for this update, decide if correction happens this step
        self.imu_lock.acquire()
        self.apriltag_lock.acquire()
        # self.gps_speed_lock.acquire()

        imu_curr = self.IMU_msg
        # gps_speed_curr = self.gps_speed_msg
        if self.measurement_ready and (self.upds_since_correction+1) >= self.upd_per_meas and self.filter_run_once:
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

            perform_apriltag_correction = (min_px[0]>self.camera_width*self.tag_in_view_margin and 
                                min_px[1]>self.camera_height*self.tag_in_view_margin and
                                max_px[0]<self.camera_width*(1-self.tag_in_view_margin) and 
                                max_px[1]<self.camera_height*(1-self.tag_in_view_margin))

        else:
            perform_apriltag_correction = False

        self.imu_lock.release()
        self.apriltag_lock.release()
        # self.gps_speed_lock.release() 

        # With multi-rate EKF need to perform correction first since it changes state we propagate from on this timestep
        if self.multirate_EKF and perform_apriltag_correction:
            # Extract state and covariance prediction at the time the measurement was recorded
            ind_meas = max(len(self.x_hist) - self.measurement_step_delay,0)
            x_check = self.x_hist[ind_meas]
            mu_check = self.mu_hist[ind_meas]
            P_check = self.P_hist[ind_meas]

            # Execute correction step, store at time measurement was recorded
            x_hat, P_hat = self.correction_step(x_check, mu_check, P_check, 0.0, 0.0, 0.0, r_c_tc, q_ct)

            self.time_correction = time.time() - tic
            tic = time.time()

            self.x_hist[ind_meas] = x_hat
            self.P_hist[ind_meas] = P_hat

            rospy.loginfo('Fused measurement. History length = {}, fuse index = {}, achieved delay of {}'.format(len(self.x_hist),ind_meas,len(self.x_hist)-ind_meas))

            # Remove history before measurement
            del self.x_hist[0:ind_meas]
            del self.mu_hist[0:ind_meas]
            del self.u_hist[0:ind_meas]
            del self.P_hist[0:ind_meas]

            # Update state history by propagating forwards based on past IMU measurements
            for i in range(1, len(self.x_hist)):
                x_check_i, mu_check_i, P_check_i = self.prediction_step(self.x_hist[i-1], self.u_hist[i], self.P_hist[i-1])[0:3]
                self.x_hist[i] = x_check_i
                self.mu_hist[i] = mu_check_i
                self.P_hist[i] = P_check_i
            
            # Sync latest estimate to state history
            self.r_nom = x_check_i[0:3,0:1]
            self.v_nom = x_check_i[3:6,0:1]
            self.q_nom = x_check_i[6:10,0:1].flatten()
            self.ab_nom = x_check_i[10:13,0:1]
            self.wb_nom = x_check_i[13:16,0:1]
            self.cov_pert = P_check_i

            self.time_filter_recalculate = time.time() - tic
            tic = time.time()

        # Current timestep    
        # Prediction step
        # Append the current measurement for this timestep
        a_meas = np.array([imu_curr.linear_acceleration.x,imu_curr.linear_acceleration.y,
                            imu_curr.linear_acceleration.z]).reshape((3,1))
        w_meas = np.array([imu_curr.angular_velocity.x,imu_curr.angular_velocity.y,
                            imu_curr.angular_velocity.z]).reshape((3,1))
        imu_meas = np.vstack((a_meas,w_meas))

        # Execute prediction
        x_check, mu_check, P_check, accel_rel = self.prediction_step(np.vstack((self.r_nom, self.v_nom, self.q_nom.reshape((4,1)), self.ab_nom, self.wb_nom))
                                                , imu_meas, self.cov_pert)

        if self.multirate_EKF:
            # Append prediction to state history, store in latest state
            self.x_hist.append(x_check)
            self.mu_hist.append(mu_check)
            self.u_hist.append(imu_meas)
            self.P_hist.append(P_check)

            self.r_nom = x_check[0:3,0:1]
            self.v_nom = x_check[3:6,0:1]
            self.q_nom = x_check[6:10,0:1].flatten()
            self.ab_nom = x_check[10:13,0:1]
            self.wb_nom = x_check[13:16,0:1]
            self.cov_pert = P_check
        elif perform_apriltag_correction:
            # Single state EKF, perform correction step and store corrected estimate
            x_hat, P_hat = self.correction_step(x_check, mu_check, P_check, 0.0, 0.0, 0.0, r_c_tc, q_ct)
            self.r_nom = x_hat[0:3, 0:1]
            self.v_nom = x_hat[3:6, 0:1]
            self.q_nom = x_hat[6:10, 0:1].flatten()
            self.ab_nom = x_hat[10:13, 0:1]
            self.wb_nom = x_hat[13:16, 0:1]
            self.cov_pert = P_hat
        else:
            # Single state EKF, prediction only
            # Just store latest prediction
            self.r_nom = x_check[0:3, 0:1]
            self.v_nom = x_check[3:6, 0:1]
            self.q_nom = x_check[6:10, 0:1].flatten()
            self.ab_nom = x_check[10:13, 0:1]
            self.wb_nom = x_check[13:16, 0:1]
            self.cov_pert = P_check

        if perform_apriltag_correction:
            self.upds_since_correction = 0
        else:
            self.upds_since_correction += 1

        self.time_prediction = time.time() - tic

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
        mu_check_cov = np.vstack((np.hstack((P_check[0:3,0:3], P_check[0:3,6:9])),
                            np.hstack((P_check[6:9,0:3], P_check[6:9,6:9]))))
        mu_check_quaternion = quaternion_exp(mu_check[6:9])
        mu_check_pose = PoseWithCovariance(pose = Pose(position = Point(x = mu_check[0],y = mu_check[1],z = mu_check[2]),
                            orientation = Quaternion(x = mu_check_quaternion[0],y = mu_check_quaternion[1],z = mu_check_quaternion[2],w = mu_check_quaternion[3])),
                            covariance = mu_check_cov.flatten().tolist())
        self.mu_check_pose_msg = PoseWithCovarianceStamped(header = curr_header, pose = mu_check_pose)
        self.mu_check_vel_msg = Vector3Stamped(header = curr_header, vector = Vector3(x = mu_check[3], y = mu_check[4], z = mu_check[5]))
        self.pred_length_msg = PointStamped(header = curr_header,point = Point(x = self.upds_since_correction))
        self.timing_msg = PointStamped(header = curr_header, point = Point(x = self.time_prediction * 1.0E3, y = self.time_correction * 1.0E3, z = self.time_filter_recalculate * 1.0E3))
        self.tf_broadcast.sendTransform((tuple(self.r_nom)), (tuple(self.q_nom)), curr_time, self.pose_frame_name, self.pose_rel_frame_name)

        if perform_apriltag_correction:
            # Report out observed vehicle position and orientation from AprilTag readings alone
            q_tv_obs = quaternion_norm(quaternion_conjugate(quaternion_multiply(self.q_vc, q_ct)))
            C_tv_obs = quaternion_matrix(q_tv_obs)[0:3,0:3]
            r_t_vt_obs = -np.linalg.multi_dot((C_tv_obs, self.C_vc, r_c_tc))-np.dot(C_tv_obs, self.r_v_cv)

            self.rel_pose_report_msg = PoseStamped(header=curr_header,
            pose = Pose(position = Point(x = r_t_vt_obs[0], y = r_t_vt_obs[1], z = r_t_vt_obs[2]),
                            orientation = Quaternion(x = q_tv_obs[0], y = q_tv_obs[1], z = q_tv_obs[2], w = q_tv_obs[3])))
            self.tf_broadcast.sendTransform((tuple(r_t_vt_obs)), (tuple(q_tv_obs)), curr_time, self.pose_report_frame_name, self.pose_rel_frame_name)
        
        if not self.filter_run_once:
            self.filter_run_once = True

        self.pub_counter += 1

    def initialize_state(self, reinit_bias):
        "Initialize state to last received AprilTag relative pose detection"
        self.state_lock.acquire()
        
        r_c_tc = np.array([[self.apriltag_msg.detections[0].pose.pose.pose.position.x],
                                [self.apriltag_msg.detections[0].pose.pose.pose.position.y],
                                [self.apriltag_msg.detections[0].pose.pose.pose.position.z]])
        q_ct = np.array([self.apriltag_msg.detections[0].pose.pose.pose.orientation.x,
                                self.apriltag_msg.detections[0].pose.pose.pose.orientation.y,
                                self.apriltag_msg.detections[0].pose.pose.pose.orientation.z,
                                self.apriltag_msg.detections[0].pose.pose.pose.orientation.w])
        
        self.q_nom = quaternion_norm(quaternion_conjugate(quaternion_multiply(self.q_vc,q_ct)))
        C_tv_nom = quaternion_matrix(self.q_nom)[0:3,0:3]
        self.r_nom = -np.dot(C_tv_nom,np.dot(self.C_vc,r_c_tc)+self.r_v_cv)
        self.v_nom = np.zeros((3,1))

        if reinit_bias:
            self.ab_nom = np.zeros((3,1))
            self.wb_nom = np.zeros((3,1))

        self.cov_pert = self.cov_init

        # Initialize state history with current state
        self.x_hist = [np.vstack((self.r_nom,self.v_nom,self.q_nom.reshape((4,1)),self.ab_nom,self.wb_nom))]
        self.u_hist = [np.zeros((6,1))]
        self.P_hist = [self.cov_pert]

        self.state_initialized = True
        self.filter_run_once = False
        self.state_lock.release()

    def prediction_step(self, x_km1, u, P_km1):
        "Execute prediction step of pose filter. Propagate state forward based on IMU measurements"
        tic = time.time()

        a_meas = u[0:3,0:1]
        w_meas = u[3:6,0:1]

        r_km1 = x_km1[0:3,0:1]
        v_km1 = x_km1[3:6,0:1]
        q_km1 = x_km1[6:10,0:1].flatten()
        ab_km1 = x_km1[10:13,0:1]
        wb_km1 = x_km1[13:16,0:1]

        a_nom = (a_meas - ab_km1)
        w_nom = (w_meas - wb_km1)
        C_nom = quaternion_matrix(q_km1)[0:3,0:3]

        accel_rel = np.dot(C_nom, a_nom) + self.g

        # Draw sigma points from stacked state and process noise
        dz_sig_km1 = self.generate_sigma_points(np.zeros((self.num_states,1)), P_km1, self.L_Q)
        L = dz_sig_km1.shape[0]
        n_z = dz_sig_km1.shape[1]

        toc_cholesky = time.time()

        # Propagate nominal state
        r_check = r_km1 + self.dT * v_km1
        v_check = v_km1 + self.dT * accel_rel
        q_check = quaternion_norm(quaternion_multiply(q_km1, quaternion_exp(self.dT * w_nom.flatten())))
        ab_check = ab_km1
        wb_check = wb_km1

        # Propagate sigma points
        dx_check = np.zeros((self.num_states, n_z))

        # Easily vectorizable ones
        dx_check[0:3, :] = dz_sig_km1[0:3, :] + self.dT * dz_sig_km1[3:6, :]
        dx_check[9:12,:] = dz_sig_km1[9:12, :] + self.dT * dz_sig_km1[21:24, :]
        dx_check[12:15,:] = dz_sig_km1[12:15, :] + self.dT * dz_sig_km1[24:27, :]

        # Harder to vectorize
        for i in np.arange(n_z):
            dx_check[3:6, i:i+1] = (v_km1 + dz_sig_km1[3:6, i:i+1] +
            self.dT * (np.linalg.multi_dot((C_nom, exponential_map(dz_sig_km1[6:9, i:i+1]), (a_nom - dz_sig_km1[9:12, i:i+1] - dz_sig_km1[15:18, i:i+1]))) + self.g) - v_check)

            q_km1_i = quaternion_multiply(q_km1, quaternion_exp(dz_sig_km1[6:9, i]))
            q_check_i = quaternion_norm(quaternion_multiply(q_km1_i, quaternion_exp(self.dT * (w_nom - dz_sig_km1[12:15,i:i+1] - dz_sig_km1[18:21, i:i+1]).flatten())))
            dx_check[6:9,i] = quaternion_log(quaternion_multiply(quaternion_conjugate(q_check), q_check_i))

        toc_propagate = time.time()

        # Compute statistics of propagated sigma points
        alpha_0 = self.kappa / (self.kappa + L)
        alpha = 0.5 / (self.kappa + L)
        mu_check =  alpha_0 * dx_check[:, 0:1] + alpha * np.sum(dx_check[:,1:], axis = 1).reshape((self.num_states, 1))
        P_check = alpha_0 * np.dot((dx_check[:, 0:1] - mu_check), (dx_check[:, 0:1] - mu_check).T)
        
        for i in np.arange(1, n_z):
            P_check += alpha * np.dot((dx_check[:, i:i+1] - mu_check), (dx_check[:, i:i+1] - mu_check).T)
        
        x_check = np.vstack((r_check, v_check, q_check.reshape((4,1)), ab_check, wb_check))

        toc_statistics = time.time()

        rospy.loginfo('Prediction Step: Cholesky = {:.2f} ms, propagation = {:.2f} ms, statistics = {:.2f} ms'.format(
            (toc_cholesky - tic) * 1000.0, (toc_propagate - toc_cholesky) * 1000.0, (toc_statistics - toc_propagate) * 1000.0))

        # Return propagated nominal state, perturbation state and relative acceleration (debugging)
        return x_check, mu_check, P_check, accel_rel

    def correction_step(self, x_check, mu_check, P_check, roll, pitch, v_gps, r_c_tc, q_ct):
        "Execute correction step of pose filter. Fuse measurements with predicted state"
        tic = time.time()

        # Stack state and measurement noises and draw sigma points
        # R = self.R_mahony
        # gps_avail = v_gps is not None
        # apriltag_avail = r_c_tc is not None and q_ct is not None
        # if gps_avail:
        #     R = block_diag(R,self.R_gps)
        # if apriltag_avail:
        #     R = block_diag(R, self.R_tag)
        L_R = self.L_tag # Start with just AprilTag
        n_meas = L_R.shape[0]

        # Draw sigma points from stacked state and measurement noise
        dz_sig_k = self.generate_sigma_points(mu_check, P_check, L_R)
        L = dz_sig_k.shape[0]
        n_z = dz_sig_k.shape[1]

        toc_cholesky = time.time()

        # Extract predicted state
        r_check = x_check[0:3,0:1]
        v_check = x_check[3:6,0:1]
        q_check = x_check[6:10,0:1].flatten()
        ab_check = x_check[10:13,0:1]
        wb_check = x_check[13:16,0:1]

        C_check = quaternion_matrix(q_check)[0:3,0:3]

        # Pass sigma points through measurement model and calculate corresponding predicted measurements
        y_check = np.zeros((n_meas, n_z))

        for i in np.arange(n_z):
            r_t_vt_i = r_check + dz_sig_k[0:3,i:i+1]
            C_tv_i = np.dot(C_check, exponential_map(dz_sig_k[6:9,i]))
            y_check[0:3,i:i+1] = -np.dot(self.C_vc.T, np.dot(C_tv_i.T, r_t_vt_i) + self.r_v_cv) + dz_sig_k[15:18, i:i+1]

            delta_q_trans = quaternion_multiply(quaternion_multiply(q_check, quaternion_exp(dz_sig_k[6:9,i])), quaternion_conjugate(q_check))
            delta_q_ct_i = quaternion_norm(quaternion_multiply(quaternion_conjugate(delta_q_trans), quaternion_exp(dz_sig_k[18:21,i])))
            y_check[3:6,i:i+1] = quaternion_log(delta_q_ct_i).reshape((3,1))

        toc_propagate = time.time()

        # Calculate statistics of measurement sigma points
        alpha_0 = self.kappa / (self.kappa + L)
        alpha = 0.5 / (self.kappa + L)
        mu_y = alpha_0 * y_check[:,0:1] + alpha * np.sum(y_check[:,1:], axis = 1).reshape((n_meas, 1))
        sig_yy = alpha_0 * np.dot(y_check[:,0:1] - mu_y, (y_check[:,0:1] - mu_y).T)
        sig_xy = alpha_0 * np.dot(dz_sig_k[0:self.num_states,0:1] - mu_check, (y_check[:,0:1] - mu_y).T)

        for i in np.arange(1, n_z):
            delta_y = y_check[:,i:i+1] - mu_y
            delta_x = dz_sig_k[0:self.num_states,i:i+1] - mu_check
            sig_yy += alpha * np.dot(delta_y, delta_y.T)
            sig_xy += alpha * np.dot(delta_x, delta_y.T)

        toc_statistics = time.time()

        # Calculate observed perturbations in AprilTag orientation measurement
        delta_q_obs = quaternion_norm(quaternion_multiply(quaternion_multiply(q_check, self.q_vc), q_ct))
        theta_obs = quaternion_log(delta_q_obs).reshape((3,1))

        # Form Kalman Gain and execute correction step
        # K_k = np.dot(sig_xy, np.linalg.inv(sig_yy))
        K_k = np.linalg.solve(sig_yy.T, sig_xy.T).T
        y_obs = np.vstack((r_c_tc, theta_obs))
        
        P_hat = P_check - np.dot(K_k, sig_xy.T)
        delta_x_hat = mu_check + np.dot(K_k, y_obs - mu_y)

        # Inject correction update and form corrected state
        r_hat = r_check + delta_x_hat[0:3,0:1]
        v_hat = v_check + delta_x_hat[3:6,0:1]
        q_hat = quaternion_norm(quaternion_multiply(q_check, quaternion_exp(delta_x_hat[6:9].flatten())))
        ab_hat = ab_check + delta_x_hat[9:12,0:1]
        wb_hat = wb_check + delta_x_hat[12:15,0:1]

        x_hat = np.vstack((r_hat, v_hat, q_hat.reshape((4,1)), ab_hat, wb_hat))

        toc_update = time.time()

        rospy.loginfo('Correction Step. Cholesky = {:.2f} ms, propagation = {:.2f} ms, statistics = {:.2f} ms, update = {:.2f}'.format(
            (toc_cholesky - tic) * 1000.0, (toc_propagate - toc_cholesky) * 1000.0, (toc_statistics - toc_propagate) * 1000.0, (toc_update - toc_statistics) * 1000.0))
        
        # Return corrected state and covariance
        return x_hat, P_hat

    def generate_sigma_points(self, mu, P, L_Q):
        "Draw sigma points from state distribution, stacked with noise"
        L_P = np.linalg.cholesky(P)
        L_z = block_diag(L_P, L_Q)
        L = L_z.shape[0]
        sigma_points = np.vstack((mu, np.zeros((L_Q.shape[0],1)))) + np.hstack((np.zeros((L, 1)), math.sqrt(L + self.kappa) * L_z, -math.sqrt(L + self.kappa) * L_z))
        return sigma_points
