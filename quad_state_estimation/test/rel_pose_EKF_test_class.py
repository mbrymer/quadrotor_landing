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
        self.measurement_delay = 0.050 # s
        self.measurement_step_delay = max(int(round(self.measurement_delay/self.dT)),1)
        self.est_bias = True
        self.accel_orien_corr = False
        self.use_magnetometer = False
        self.multirate_EKF = True

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

        self.ab_const = np.array([0.324,0.078,-0.1])
        self.wb_const = np.array([-0.019,-0.013,0])

        # Process and Measurement Noises
        self.r_cov_init = 0.1
        self.v_cov_init = 0.1
        self.ang_cov_init = 0.15
        self.ab_cov_init = 0.5
        self.wb_cov_init = 0.1

        # self.Q_a = 0.01*np.eye(3)
        # self.Q_w = 0.01*np.eye(3)

        # self.Q_a = 0.00025*np.eye(3) # Hardware values
        # self.Q_w = 0.00045*np.eye(3) # 0.0005
        # self.Q_ab = 7.0E-6*np.eye(3) # 5E-5
        # self.Q_wb = 5.0E-6*np.eye(3) # 5E-6
        self.Q_a = 0.005*np.eye(3) # Sim values
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
        # self.R_r =  np.diag(np.array([0.0015,0.0015,0.006])) # Hardware values
        # self.R_ang = np.diag(np.array([0.0015,0.0015,0.04])) # 
        self.R_r = np.diag(np.array([0.005,0.005,0.015])) # Sim Values
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

        # State, input and covariance histories for multi-rate EKF
        self.x_hist = [np.zeros((self.num_states,1))]
        self.u_hist = [np.zeros((6,1))]
        self.P_hist = [np.zeros((self.num_states,self.num_states))]

        # Camera to Vehicle Transform
        # self.r_v_cv = np.array([[0.06036412],[-0.00145196],[-0.04439579]]) # Harware values
        self.r_v_cv = np.array([[0],[0],[-0.073]]) # Sim Values
        # self.q_vc = quaternion_norm(np.array([-0.7035177, 0.7106742, 0.0014521, -0.0017207])) # Hardware values
        self.q_vc = quaternion_norm(np.array([0.70711,-0.70711,0,0])) # Sim values
        self.C_vc = quaternion_matrix(self.q_vc)[0:3,0:3]

        # Camera and Tag Parameters
        self.camera_K = np.array([[241.4268,0,376.5],
                                    [0,241.4268,240.5],
                                    [0,0,1]]) # Sim values
        # self.camera_K = np.array([[448.24317239815787,0,325.2550869383401],
        #                             [0,447.88113349903773,242.3517901446831],
        #                             [0,0,1]]) # Hardware values
        # self.camera_width = 640 # Hardware
        self.camera_width = 752 # Sim values
        self.camera_height = 480
        # self.tag_width = 0.199375 # Hardware # m
        self.tag_width = 0.8 # Sim
        self.tag_in_view_margin = 0.02 # %

        self.tag_corners = np.array([[self.tag_width/2,-self.tag_width/2,-self.tag_width/2,self.tag_width/2],
                                    [self.tag_width/2,self.tag_width/2,-self.tag_width/2,-self.tag_width/2],
                                    [0,0,0,0],
                                    [1,1,1,1]])

        # Counters/flags
        self.state_initialized = False
        self.measurement_ready = False
        self.filter_run_once = False
        self.upds_since_correction = 0
        self.pub_counter = 0

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
        if self.measurement_ready and (self.upds_since_correction+1)>=self.upd_per_meas and self.filter_run_once:
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

        if perform_correction:
            # Fuse motion model prediction with AprilTag readings
            # Convert AprilTag readings to vehicle state coordinates
            q_tv_obs = quaternion_norm(quaternion_conjugate(quaternion_multiply(self.q_vc,q_ct)))
            C_tv_obs = quaternion_matrix(q_tv_obs)[0:3,0:3] # Add for direct orientation method
            # r_t_vt_obs = -np.linalg.multi_dot((C_check,self.C_vc,r_c_tc))-np.dot(C_check,self.r_v_cv)
            r_t_vt_obs = -np.linalg.multi_dot((C_tv_obs,self.C_vc,r_c_tc))-np.dot(C_tv_obs,self.r_v_cv) # Flip to this for direct orientation method

        # With multi-rate EKF need to perform correction first since it changes state we propagate from on this timestep
        if self.multirate_EKF and perform_correction:
            # Extract state and covariance prediction at the time the measurement was recorded
            ind_meas = max(len(self.x_hist)-self.measurement_step_delay,0)
            x_check = self.x_hist[ind_meas]
            P_check = self.P_hist[ind_meas]

            # Execute correction step, store at time measurement was recorded
            x_hat, P_hat = self.correction_step(x_check,P_check,r_c_tc,q_ct)

            if self.pub_counter >= 2514:
                wait = 5
            self.x_hist[ind_meas] = x_hat
            self.P_hist[ind_meas] = P_hat

            rospy.loginfo('Fused measurement. History length = {}, fuse index = {}, achieved delay of {}'.format(len(self.x_hist),ind_meas,len(self.x_hist)-ind_meas))

            # Remove history before measurement
            del self.x_hist[0:ind_meas]
            del self.u_hist[0:ind_meas]
            del self.P_hist[0:ind_meas]

            # Update state history by propagating forwards based on past IMU measurements
            for i in range(1,len(self.x_hist)):
                x_check_i,P_check_i = self.prediction_step(self.x_hist[i-1],self.u_hist[i],self.P_hist[i-1])[0:2]
                self.x_hist[i] = x_check_i
                self.P_hist[i] = P_check_i
            
            # Sync latest estimate to state history
            self.r_nom = x_check_i[0:3,0:1]
            self.v_nom = x_check_i[3:6,0:1]
            self.q_nom = x_check_i[6:10,0:1].flatten()
            self.ab_nom = x_check_i[10:13,0:1]
            self.wb_nom = x_check_i[13:16,0:1]
            self.cov_pert = P_check_i

        # Current timestep    
        # Prediction step
        # Append the current measurement for this timestep
        a_meas = np.array([imu_curr.linear_acceleration.x,imu_curr.linear_acceleration.y,
                            imu_curr.linear_acceleration.z]).reshape((3,1))
        w_meas = np.array([imu_curr.angular_velocity.x,imu_curr.angular_velocity.y,
                            imu_curr.angular_velocity.z]).reshape((3,1))
        imu_meas = np.vstack((a_meas,w_meas))

        # Execute prediction
        x_check, P_check, accel_rel = self.prediction_step(np.vstack((self.r_nom,self.v_nom,self.q_nom.reshape((4,1)),self.ab_nom,self.wb_nom))
                                                ,imu_meas,self.cov_pert)

        if self.multirate_EKF:
            # Append prediction to state history, store in latest state
            self.x_hist.append(x_check)
            self.u_hist.append(imu_meas)
            self.P_hist.append(P_check)

            self.r_nom = x_check[0:3,0:1]
            self.v_nom = x_check[3:6,0:1]
            self.q_nom = x_check[6:10,0:1].flatten()
            self.ab_nom = x_check[10:13,0:1]
            self.wb_nom = x_check[13:16,0:1]
            self.cov_pert = P_check
        elif perform_correction:
            # Single state EKF, perform correction step and store corrected estimate
            x_hat,P_hat = self.correction_step(x_check,P_check,r_c_tc,q_ct)
            self.r_nom = x_hat[0:3,0:1]
            self.v_nom = x_hat[3:6,0:1]
            self.q_nom = x_hat[6:10,0:1].flatten()
            self.ab_nom = x_hat[10:13,0:1]
            self.wb_nom = x_hat[13:16,0:1]
            self.cov_pert = P_hat
        else:
            # Single state EKF, prediction only
            # Just store latest prediction
            self.r_nom = x_check[0:3,0:1]
            self.v_nom = x_check[3:6,0:1]
            self.q_nom = x_check[6:10,0:1].flatten()
            self.ab_nom = x_check[10:13,0:1]
            self.wb_nom = x_check[13:16,0:1]
            self.cov_pert = P_check

        if perform_correction:
            self.upds_since_correction = 0
        else:
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
        
        if not self.filter_run_once:
            self.filter_run_once = True

        self.pub_counter += 1

    def initialize_state(self,reinit_bias):
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

    def prediction_step(self,x_km1,u,P_km1):
        "Execute prediction step of EKF. Propagate state forward based on IMU measurements"
        a_meas = u[0:3,0:1]
        w_meas = u[3:6,0:1]

        r_km1 = x_km1[0:3,0:1]
        v_km1 = x_km1[3:6,0:1]
        q_km1 = x_km1[6:10,0:1].flatten()
        ab_km1 = x_km1[10:13,0:1]
        wb_km1 = x_km1[13:16,0:1]

        a_nom = (a_meas-ab_km1)
        w_nom = (w_meas-wb_km1)
        C_nom = quaternion_matrix(q_km1)[0:3,0:3]

        accel_rel = np.dot(C_nom,a_nom) + self.g

        # Propagate nominal state
        r_check = r_km1 + self.dT*v_km1
        v_check = v_km1 + self.dT*accel_rel
        q_check = quaternion_norm(quaternion_multiply(q_km1,quaternion_exp(self.dT*w_nom.flatten())))
        ab_check = ab_km1
        wb_check = wb_km1

        x_check = np.vstack((r_check,v_check,q_check.reshape((4,1)),ab_check,wb_check))

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
        P_check = np.linalg.multi_dot((F_km1,P_km1,F_km1.T)) + np.linalg.multi_dot((W_km1,self.Q,W_km1.T))

        # Return propagated state, covariance and relative acceleration (debugging)
        return x_check,P_check,accel_rel

    def correction_step(self,x_check,P_check,r_c_tc,q_ct):
        "Execute correction step of EKF. Fuse AprilTag measurements with predicted state"

        # Fuse motion model prediction with AprilTag readings
        # Convert AprilTag readings to vehicle state coordinates
        q_tv_obs = quaternion_norm(quaternion_conjugate(quaternion_multiply(self.q_vc,q_ct)))
        C_tv_obs = quaternion_matrix(q_tv_obs)[0:3,0:3] # Add for direct orientation method
        # r_t_vt_obs = -np.linalg.multi_dot((C_check,self.C_vc,r_c_tc))-np.dot(C_check,self.r_v_cv)
        r_t_vt_obs = -np.linalg.multi_dot((C_tv_obs,self.C_vc,r_c_tc))-np.dot(C_tv_obs,self.r_v_cv) # Flip to this for direct orientation method

        # Extract predicted state
        r_check = x_check[0:3,0:1]
        v_check = x_check[3:6,0:1]
        q_check = x_check[6:10,0:1].flatten()
        ab_check = x_check[10:13,0:1]
        wb_check = x_check[13:16,0:1]

        C_check = quaternion_matrix(q_check)[0:3,0:3]

        # Calculate observed perturbations in measurements
        delta_r_obs = r_t_vt_obs - r_check
        delta_q_obs = quaternion_multiply(quaternion_conjugate(q_check),q_tv_obs)
        delta_theta_obs = quaternion_log(delta_q_obs).reshape((3,1))

        # Calculate Jacobians
        G_k = np.zeros((6,self.num_states))
        N_k = block_diag(-np.dot(C_check,self.C_vc),self.C_vc)
        N_k[0:3,3:6] = skew_symm(r_check) # add for direct orientation method

        G_k[0:3,0:3] = np.eye(3)
        # G_k[0:3,6:9] = np.dot(C_check,skew_symm(np.dot(C_check.T,r_check))) # Remove for direct orientation method
        G_k[3:6,6:9] = np.eye(3)

        R_k = np.linalg.multi_dot((N_k,self.R,N_k.T))

        # Form Kalman Gain and execute correction step
        Cov_meas_inv = np.linalg.inv(np.linalg.multi_dot((G_k,P_check,G_k.T))+R_k)
        K_k = np.linalg.multi_dot((P_check,G_k.T,Cov_meas_inv))

        delta_y_obs = np.vstack((delta_r_obs,delta_theta_obs))
        
        P_hat = np.dot(np.eye(self.num_states)-np.dot(K_k,G_k),P_check)
        delta_x_hat = np.dot(K_k,delta_y_obs)

        # Inject correction update and form corrected state
        r_hat = r_check + delta_x_hat[0:3,0:1]
        v_hat = v_check + delta_x_hat[3:6,0:1]
        q_hat = quaternion_norm(quaternion_multiply(q_check,quaternion_exp(delta_x_hat[6:9].flatten())))

        if self.est_bias:
            ab_hat = ab_check + delta_x_hat[9:12,0:1]
            wb_hat = wb_check + delta_x_hat[12:15,0:1]
        else:
            ab_hat = np.zeros((3,1))
            wb_hat = np.zeros((3,1))

        x_hat = np.vstack((r_hat,v_hat,q_hat.reshape((4,1)),ab_hat,wb_hat))
        
        # Return corrected state and covariance
        return x_hat, P_hat
