#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Relative Pose EKF Python Test Node
#

from __future__ import division, print_function, absolute_import

# Import libraries
import rospy
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_about_axis, quaternion_multiply

from rel_pose_EKF_test_class import RelativePoseEKF

# Import message types
from geometry_msgs.msg import Point, Vector3, Quaternion, PoseStamped,PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
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

        # Subscribers:
        self.IMU_topic = '/drone/imu'
        self.apriltag_topic = 'tag_detections'

        self.IMU_sub = rospy.Subscriber(self.IMU_topic,Imu,callback=self.IMU_sub_callback)
        self.apriltag_sub = rospy.Subscriber(self.apriltag_topic,AprilTagDetectionArray,callback=self.apriltag_sub_callback)

        # Publishers:
        self.rel_pose_topic = '/state_estimation/rel_pose_state'
        self.rel_pose_report_topic = 'state_estimation/rel_pose_reported'
        self.IMU_bias_topic = '/state_estimation/IMU_bias'
        self.rel_pose_pub = rospy.Publisher(self.rel_pose_topic,PoseWithCovarianceStamped,queue_size=1)
        self.rel_pose_report_pub = rospy.Publisher(self.rel_pose_report_topic,PoseStamped,queue_size=1)
        self.IMU_bias_pub = rospy.Publisher(self.IMU_bias_topic,Imu,queue_size=1)

        # Timers:
        self.update_timer = rospy.Timer(rospy.Duration(1.0/self.update_freq),self.filter_update_callback)

    def IMU_sub_callback(self,msg):
        # TODO: Thread safety
        self.rel_pose_ekf.imu_lock.acquire()
        self.rel_pose_ekf.IMU_msg = msg
        self.rel_pose_ekf.imu_lock.release()

    def apriltag_sub_callback(self,msg):
        # TODO: Thread safety
        if len(msg.detections)>0:
            self.rel_pose_ekf.apriltag_lock.acquire()
            self.rel_pose_ekf.apriltag_msg = msg
            # curr_time = rospy.get_rostime()
            # rospy.loginfo('Received detection. Header time = {:.3f}, current time = {:.3f}'.format(msg.header.stamp.to_sec(),curr_time.to_sec()))
            self.rel_pose_ekf.measurement_ready = True
        
            if not self.rel_pose_ekf.state_initialized:
                self.rel_pose_ekf.initialize_state(False)
            
            self.rel_pose_ekf.apriltag_lock.release()

    def filter_update_callback(self,event):
        # Execute filter update
        self.rel_pose_ekf.filter_update()

        # Publish state estimate
        self.rel_pose_pub.publish(self.rel_pose_ekf.rel_pose_msg)
        self.IMU_bias_pub.publish(self.rel_pose_ekf.IMU_bias_msg)
        self.rel_pose_report_pub.publish(self.rel_pose_ekf.rel_pose_report_msg)

if __name__ == '__main__':
    try:
        # Initialize node
        rospy.init_node('rel_pose_EKF_test')
        RelativePoseEKFNode()
        rospy.spin()
    except rospy.ROSInterruptException:
	    pass