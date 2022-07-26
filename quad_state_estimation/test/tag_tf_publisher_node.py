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
from std_msgs.msg import Header

if __name__ == '__main__':
    
    # Initialize node
    rospy.init_node('tag_frame_publisher_node')
    tf_broadcaster = tf.TransformBroadcaster()

    run_rate = 100 # Hz

    base_frame_1 = "target/tag_link"
    frame_name_1 = "target/tag0_link"
    r_frame_1 = np.array([0,0,0])
    q_frame_1 = np.array([0,0,1,0])

    base_frame_2 = "target/tag_link"
    frame_name_2 = "target/tag1_link"
    r_frame_2 = np.array([0,0,0])
    q_frame_2 = np.array([0,0,1,0])

    rate = rospy.Rate(run_rate)
        
    rospy.loginfo('TF publisher node started, beginning to publish transforms:')

    while not rospy.is_shutdown():
        curr_time = rospy.get_rostime()

        tf_broadcaster.sendTransform((tuple(r_frame_1)), (tuple(q_frame_1)), curr_time,frame_name_1,base_frame_1)
        tf_broadcaster.sendTransform((tuple(r_frame_2)), (tuple(q_frame_2)), curr_time,frame_name_2,base_frame_2)

        rate.sleep()
