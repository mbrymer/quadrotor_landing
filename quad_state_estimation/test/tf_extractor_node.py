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
    rospy.init_node('tf_extractor_node')
    tf_listener = tf.TransformListener()

    run_rate = 100 # Hz

    base_frame_1 = "target/tag_link"
    relative_frame_1 = "drone/base_link"
    pose_name_1 = "drone/rel_pose_ground_truth"
    tf_pub_1 = rospy.Publisher(pose_name_1,PoseStamped,queue_size=1)

    base_frame_2 = "drone/camera_sensor/camera_na_optical_link"
    relative_frame_2 = "target/tag_link"
    pose_name_2 = "drone/apriltag_ground_truth"
    tf_pub_2 = rospy.Publisher(pose_name_2,PoseStamped,queue_size=1)

    rate = rospy.Rate(run_rate)
        
    rospy.loginfo('TF extractor node started, beginning to publish transforms:')

    while not rospy.is_shutdown():
        curr_time = rospy.get_rostime()

        try:
            (trans_1,rot_1) = tf_listener.lookupTransform(base_frame_1,relative_frame_1,curr_time)
        except:
            continue
        
        pose_msg_1 = PoseStamped(header=Header(stamp=curr_time,frame_id = pose_name_1),
                                pose = Pose(position = Point(x = trans_1[0], y = trans_1[1], z = trans_1[2]),
                                orientation = Quaternion(x = rot_1[0], y = rot_1[1], z = rot_1[2], w = rot_1[3])))
        tf_pub_1.publish(pose_msg_1)

        try:
            (trans_2,rot_2) = tf_listener.lookupTransform(base_frame_2,relative_frame_2,curr_time)
        except:
            continue
        
        pose_msg_2 = PoseStamped(header=Header(stamp=curr_time,frame_id = pose_name_2),
                                pose = Pose(position = Point(x = trans_2[0], y = trans_2[1], z = trans_2[2]),
                                orientation = Quaternion(x = rot_2[0], y = rot_2[1], z = rot_2[2], w = rot_2[3])))
        tf_pub_2.publish(pose_msg_2)

        rate.sleep()
