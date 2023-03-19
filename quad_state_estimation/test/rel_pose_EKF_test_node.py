#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Relative Pose EKF Python Test Node
#

from __future__ import division, print_function, absolute_import

# Import libraries
import rospy
import threading, copy
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_about_axis, quaternion_multiply

from rel_pose_EKF_test_class import RelativePoseEKF
from mahony_filter import MahonyFilter

# Import message types
from geometry_msgs.msg import PointStamped, Vector3, Quaternion, PoseStamped,PoseWithCovarianceStamped, Vector3Stamped
from sensor_msgs.msg import Imu, CameraInfo
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Float64

class RelativePoseEKFNode(object):
    "Node for EKF relative pose estimation based on AprilTag detections"
    def __init__(self):
        
        # Rates:
        self.update_freq = 100 # Hz
        self.measurement_freq = 10 # Hz

        # Objects:
        self.rel_pose_ekf = RelativePoseEKF(self.update_freq,self.measurement_freq)
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
        self.magnetometer_sub = rospy.Subscriber(self.magnetometer_topic,Vector3Stamped,callback=self.magnetometer_sub_callback)

        # Publishers:
        self.rel_pose_topic = '/state_estimation/rel_pose_state'
        self.rel_pose_report_topic = 'state_estimation/rel_pose_reported'
        self.rel_vel_topic = '/state_estimation/rel_pose_velocity'
        self.rel_accel_topic = '/state_estimation/rel_pose_acceleration'
        self.IMU_bias_topic = '/state_estimation/IMU_bias'
        self.pred_length_topic = '/state_estimation/upds_since_correction'
        self.mahony_filter_topic = '/state_estimation/mahony_rel_pose'
        self.mahony_filter_bias_topic = '/state_estimation/mahony_IMU_bias'

        self.rel_pose_pub = rospy.Publisher(self.rel_pose_topic,PoseWithCovarianceStamped,queue_size=1)
        self.rel_pose_report_pub = rospy.Publisher(self.rel_pose_report_topic,PoseStamped,queue_size=1)
        self.rel_vel_pub = rospy.Publisher(self.rel_vel_topic,Vector3Stamped,queue_size=1)
        self.rel_accel_pub = rospy.Publisher(self.rel_accel_topic,Vector3Stamped,queue_size=1)
        self.IMU_bias_pub = rospy.Publisher(self.IMU_bias_topic,Imu,queue_size=1)
        self.pred_length_pub = rospy.Publisher(self.pred_length_topic,PointStamped,queue_size=1)
        self.mahony_filter_pose_pub = rospy.Publisher(self.mahony_filter_topic,PoseStamped,queue_size=1)
        self.mahony_filter_bias_pub = rospy.Publisher(self.mahony_filter_bias_topic,Imu,queue_size=1)

        # Timers:
        self.update_timer = rospy.Timer(rospy.Duration(1.0/self.update_freq),self.filter_update_callback)

    def IMU_sub_callback(self,msg):
        self.rel_pose_ekf.imu_lock.acquire()
        self.rel_pose_ekf.IMU_msg = msg
        self.rel_pose_ekf.imu_lock.release()

        self.mahony_filter.imu_lock.acquire()
        self.mahony_filter.imu_msg = msg
        self.mahony_filter.imu_lock.release()

    def magnetometer_sub_callback(self,msg):
        self.rel_pose_ekf.magnetometer_lock.acquire()
        self.rel_pose_ekf.magnetometer_msg = msg
        self.rel_pose_ekf.magnetometer_lock.release()

    def camera_info_sub_callback(self,msg):
        self.camera_info_lock.acquire()
        self.camera_info_msg = msg
        self.camera_info_lock.release()
        
    def apriltag_sub_callback(self,msg):
        if len(msg.detections)>0:
            self.rel_pose_ekf.apriltag_lock.acquire()
            # self.camera_info_lock.acquire()
            self.rel_pose_ekf.apriltag_msg = msg
            # camera_info_header = copy.deepcopy(self.camera_info_msg.header)
            # curr_time = rospy.get_rostime()
            # rospy.loginfo('Received detection. Header time = {:.3f}, current time = {:.3f}'.format(msg.header.stamp.to_sec(),curr_time.to_sec()))
            # rospy.loginfo('Camera info delay: {:.1f} ms to AprilTag, {:.1f} ms to now'.format(1000*(msg.header.stamp.to_sec()-camera_info_header.stamp.to_sec()),
            #                                                                                 1000*(curr_time.to_sec()-camera_info_header.stamp.to_sec())))
            self.rel_pose_ekf.measurement_ready = True
        
            if not self.rel_pose_ekf.state_initialized:
                self.rel_pose_ekf.initialize_state(False)
            
            self.rel_pose_ekf.apriltag_lock.release()
            # self.camera_info_lock.release()

    def filter_update_callback(self,event):
        # Execute filter updates
        self.rel_pose_ekf.filter_update()
        self.mahony_filter.filter_update()

        # Publish state estimate
        self.rel_pose_pub.publish(self.rel_pose_ekf.rel_pose_msg)
        self.rel_vel_pub.publish(self.rel_pose_ekf.rel_vel_msg)
        self.rel_accel_pub.publish(self.rel_pose_ekf.rel_accel_msg)
        self.IMU_bias_pub.publish(self.rel_pose_ekf.IMU_bias_msg)
        self.rel_pose_report_pub.publish(self.rel_pose_ekf.rel_pose_report_msg)
        self.pred_length_pub.publish(self.rel_pose_ekf.pred_length_msg)
        self.mahony_filter_pose_pub.publish(self.mahony_filter.pose_msg)
        self.mahony_filter_bias_pub.publish(self.mahony_filter.imu_bias_msg)

if __name__ == '__main__':
    try:
        # Initialize node
        rospy.init_node('rel_pose_EKF_test')
        RelativePoseEKFNode()
        rospy.spin()
    except rospy.ROSInterruptException:
	    pass