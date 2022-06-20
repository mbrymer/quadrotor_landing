#!/usr/bin/env python

#
# AER 1810 Quadrotor Landing Project
# Fake Magnetometer Node
#

from __future__ import division, print_function, absolute_import

# Import libraries
import rospy
import threading
import numpy as np

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, Vector3Stamped, Vector3
from std_msgs.msg import Header

from tf.transformations import quaternion_matrix

class FakeMagnetometer:
    def __init__(self):
        # Parameters
        self.loop_rate = 100 # Hz

        self.field_direction = np.array([1,0,0])
        self.bias = np.array([0,0,0])
        self.field_noise = 0.05

        self.pose_lock = threading.Lock()
        self.pose_topic = "drone/ground_truth/pose"
        self.pose_val = Pose()

        self.pose_sub = rospy.Subscriber(self.pose_topic,Pose,callback=self.pose_callback)

        self.magnetometer_timer = rospy.Timer(rospy.Duration(1.0/self.loop_rate),self.runnable)

        self.magnetometer_topic = "drone/fake_magnetometer"
        self.magnetometer_msg = Vector3Stamped()
        self.magnetometer_pub = rospy.Publisher(self.magnetometer_topic,Vector3Stamped,queue_size=1)

    def pose_callback(self,msg):
        self.pose_lock.acquire()
        self.pose_val = msg
        self.pose_lock.release()

    def runnable(self,event):
        # Project fake field direction into quad CSYS and publish
        self.pose_lock.acquire()
        pose_local = self.pose_val
        self.pose_lock.release()

        q_pose = np.array([pose_local.orientation.x, pose_local.orientation.y, pose_local.orientation.z, pose_local.orientation.w])
        C = quaternion_matrix(q_pose)[0:3,0:3]

        # Project and add some noise
        b_proj = np.dot(C.T,self.field_direction) + np.random.normal(scale=self.field_noise,size=(3)) + self.bias
        b_proj = b_proj/np.linalg.norm(b_proj)

        # Publish
        curr_time = rospy.get_rostime()
        self.magnetometer_msg = Vector3Stamped(header=Header(stamp=curr_time,frame_id = "drone/base_link"),
                                            vector = Vector3(x = b_proj[0], y = b_proj[1], z = b_proj[2]))
        self.magnetometer_pub.publish(self.magnetometer_msg)

if __name__ == '__main__':
    
    try:
        # Initialize node
        rospy.init_node('fake_magnetometer_node')
        FakeMagnetometer()
        rospy.loginfo('Fake magnetometer node started, publishing made up measurements')
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
