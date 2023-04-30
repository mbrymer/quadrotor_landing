#!/usr/bin/env python

#
# Quadrotor Landing Project
# Relative Pose Filter Python Node
#

from __future__ import division, print_function, absolute_import

# Import libraries
import rospy
import threading, copy
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_about_axis, quaternion_multiply

from relative_pose_filter import RelativePoseFilter
from mahony_filter import MahonyFilter

# Import message types
from geometry_msgs.msg import PointStamped, Vector3, Quaternion, PoseStamped,PoseWithCovarianceStamped, Vector3Stamped
from sensor_msgs.msg import Imu, CameraInfo
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Float64

class RelativePoseFilterNode(object):
    "Node for relative pose estimation based on AprilTag detections"
    def __init__(self):
        
        # Rates:
        self.update_freq = 100 # Hz
        self.measurement_freq = 10 # Hz

        # Objects:
        self.rel_pose_filter = RelativePoseFilter(self.update_freq, self.measurement_freq)
        self.mahony_filter = MahonyFilter(self.update_freq)

        self.camera_info_msg = CameraInfo()
        self.camera_info_lock = threading.Lock()

        # Subscribers:
        self.IMU_topic = '/drone/imu'
        self.magnetometer_topic = 'drone/fake_magnetometer'
        self.apriltag_topic = '/tag_detections'
        self.camera_info_topic = '/drone/camera_sensor/camera_na/camera_info'

        self.IMU_sub = rospy.Subscriber(self.IMU_topic,Imu,callback=self.IMU_sub_callback)
        self.apriltag_sub = rospy.Subscriber(self.apriltag_topic,AprilTagDetectionArray,callback=self.apriltag_sub_callback)
        self.camera_info_sub = rospy.Subscriber(self.camera_info_topic,CameraInfo,callback=self.camera_info_sub_callback)

        # Publishers:
        self.rel_pose_topic = '/state_estimation/rel_pose_state'
        self.rel_pose_report_topic = 'state_estimation/rel_pose_reported'
        self.rel_vel_topic = '/state_estimation/rel_pose_velocity'
        self.rel_accel_topic = '/state_estimation/rel_pose_acceleration'
        self.mu_check_pose_topic = '/state_estimation/mu_check_rel_pose'
        self.mu_check_vel_topic = '/state_estimation/mu_check_rel_pose_velocity'
        self.IMU_bias_topic = '/state_estimation/IMU_bias'
        self.pred_length_topic = '/state_estimation/upds_since_correction'
        self.timing_topic = '/state_estimation/pose_filter_timing'

        self.mahony_filter_topic = '/state_estimation/mahony_rel_pose'
        self.mahony_filter_bias_topic = '/state_estimation/mahony_IMU_bias'
        self.attitude_EKF_topic = '/state_estimation/attitude_EKF_rel_pose'

        self.rel_pose_pub = rospy.Publisher(self.rel_pose_topic,PoseWithCovarianceStamped,queue_size=1)
        self.rel_pose_report_pub = rospy.Publisher(self.rel_pose_report_topic,PoseStamped,queue_size=1)
        self.rel_vel_pub = rospy.Publisher(self.rel_vel_topic,Vector3Stamped,queue_size=1)
        self.rel_accel_pub = rospy.Publisher(self.rel_accel_topic,Vector3Stamped,queue_size=1)
        self.mu_check_pose_pub = rospy.Publisher(self.mu_check_pose_topic, PoseWithCovarianceStamped, queue_size = 1)
        self.mu_check_vel_topic = rospy.Publisher(self.mu_check_vel_topic, Vector3Stamped, queue_size=1)
        self.IMU_bias_pub = rospy.Publisher(self.IMU_bias_topic,Imu,queue_size=1)
        self.pred_length_pub = rospy.Publisher(self.pred_length_topic,PointStamped,queue_size=1)
        self.timing_pub = rospy.Publisher(self.timing_topic, PointStamped, queue_size = 1)

        self.mahony_filter_pose_pub = rospy.Publisher(self.mahony_filter_topic,PoseStamped,queue_size=1)
        self.mahony_filter_bias_pub = rospy.Publisher(self.mahony_filter_bias_topic,Imu,queue_size=1)
        self.attitude_EKF_pose_pub = rospy.Publisher(self.attitude_EKF_topic,PoseStamped,queue_size=1)

        # Timers:
        self.update_timer = rospy.Timer(rospy.Duration(1.0 / self.update_freq), self.filter_update_callback)

    def IMU_sub_callback(self,msg):
        self.rel_pose_filter.imu_lock.acquire()
        self.rel_pose_filter.IMU_msg = msg
        self.rel_pose_filter.imu_lock.release()

        self.mahony_filter.imu_lock.acquire()
        self.mahony_filter.imu_msg = msg
        self.mahony_filter.imu_lock.release()

    def camera_info_sub_callback(self,msg):
        self.camera_info_lock.acquire()
        self.camera_info_msg = msg
        self.camera_info_lock.release()
        
    def apriltag_sub_callback(self,msg):
        if len(msg.detections)>0:
            self.rel_pose_filter.apriltag_lock.acquire()
            self.rel_pose_filter.apriltag_msg = msg

            self.rel_pose_filter.measurement_ready = True
        
            if not self.rel_pose_filter.state_initialized:
                self.rel_pose_filter.initialize_state(False)
            
            self.rel_pose_filter.apriltag_lock.release()

    def filter_update_callback(self,event):
        # Execute filter updates
        self.mahony_filter.filter_update()
        self.rel_pose_filter.filter_update()

        # Publish state estimate
        self.rel_pose_pub.publish(self.rel_pose_filter.rel_pose_msg)
        self.rel_pose_report_pub.publish(self.rel_pose_filter.rel_pose_report_msg)
        self.rel_vel_pub.publish(self.rel_pose_filter.rel_vel_msg)
        self.rel_accel_pub.publish(self.rel_pose_filter.rel_accel_msg)
        self.mu_check_pose_pub.publish(self.rel_pose_filter.mu_check_pose_msg)
        self.mu_check_vel_topic.publish(self.rel_pose_filter.mu_check_vel_msg)
        self.IMU_bias_pub.publish(self.rel_pose_filter.IMU_bias_msg)
        self.pred_length_pub.publish(self.rel_pose_filter.pred_length_msg)
        self.timing_pub.publish(self.rel_pose_filter.timing_msg)
        self.mahony_filter_pose_pub.publish(self.mahony_filter.pose_msg)
        self.mahony_filter_bias_pub.publish(self.mahony_filter.imu_bias_msg)

if __name__ == '__main__':
    try:
        # Initialize node
        rospy.init_node('relative_pose_filter_python')
        RelativePoseFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
	    pass