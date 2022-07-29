#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Transform Extractor Test Node
#

from __future__ import division, print_function, absolute_import

# Import libraries
import rospy
import numpy as np
import tf

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from apriltag_ros.msg import AprilTagDetection, AprilTagDetectionArray
from std_msgs.msg import Header

def apriltag_sub_callback(msg):
    curr_time = rospy.get_rostime()
    rospy.loginfo('Received detection. Header time = {:.3f}, current time = {:.3f}'.format(msg.header.stamp.to_sec(),curr_time.to_sec()))
    rospy.loginfo('Delay = {:.1f} ms'.format(1000*(curr_time.to_sec()-msg.header.stamp.to_sec())))

if __name__ == '__main__':
    
    # Initialize node
    rospy.init_node('camera_delay')

    apriltag_topic = '/tag_detections'
    apriltag_sub = rospy.Subscriber(apriltag_topic,AprilTagDetectionArray,callback=apriltag_sub_callback)
    rospy.spin()
