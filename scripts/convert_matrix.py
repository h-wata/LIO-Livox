#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
import rospy
import tf.transformations as tr
from geometry_msgs.msg import Pose, PoseStamped, Transform, TransformStamped
from nav_msgs.msg import Odometry


class PoseCoverter():

    def __init__(self):
        rospy.Subscriber('/Odometry', Odometry, self.odom_cb, queue_size=1)
        self.odom_array = np.zeros((1, 12))
        self.time_array = np.zeros((1, 1))

    def odom_cb(self, msg):
        """odometry callback"""
        g = self.msg_to_se3(msg.pose.pose)
        arr = np.delete(g.flatten(), slice(12, 16))
        arr = arr.reshape(1, 12)
        self.odom_array = np.append(self.odom_array, arr, axis=0)
        time = np.array([msg.header.stamp.secs + msg.header.stamp.nsecs * 1.0e-9]).reshape(1, 1)
        self.time_array = np.append(self.time_array, time, axis=0)
        path = rospy.get_param('~save_file_dir')
        file_name = path + str(msg.header.stamp.secs) + "_" + '{:0=9}'.format(msg.header.stamp.nsecs) + ".odom"
        np.savetxt(file_name, g, fmt='%.5f', delimiter=' ')

    def pose_to_pq(self, msg):
        """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.position.x, msg.position.y, msg.position.z])
        q = np.array([msg.orientation.x, msg.orientation.y,
                      msg.orientation.z, msg.orientation.w])
        return p, q

    def pose_stamped_to_pq(self, msg):
        """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return self.pose_to_pq(msg.pose)

    def transform_to_pq(self, msg):
        """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
        q = np.array([msg.rotation.x, msg.rotation.y,
                      msg.rotation.z, msg.rotation.w])
        return p, q

    def transform_stamped_to_pq(self, msg):
        """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

        @param msg: ROS message to be converted
        @return:
        - p: position as a np.array
        - q: quaternion as a numpy array (order = [x,y,z,w])
        """
        return self.transform_to_pq(msg.transform)

    def msg_to_se3(self, msg):
        """Conversion from geometric ROS messages into SE(3)

        @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
        C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
        @return: a 4x4 SE(3) matrix as a numpy array
        @note: Throws TypeError if we receive an incorrect type.
        """
        if isinstance(msg, Pose):
            p, q = self.pose_to_pq(msg)
        elif isinstance(msg, PoseStamped):
            p, q = self.pose_stamped_to_pq(msg)
        elif isinstance(msg, Transform):
            p, q = self.transform_to_pq(msg)
        elif isinstance(msg, TransformStamped):
            p, q = self.transform_stamped_to_pq(msg)
        else:
            raise TypeError("Invalid type for conversion to SE(3)")
        norm = np.linalg.norm(q)
        if np.abs(norm - 1.0) > 1e-3:
            raise ValueError(
                "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                    str(q), np.linalg.norm(q)))
        elif np.abs(norm - 1.0) > 1e-6:
            q = q / norm
        g = tr.quaternion_matrix(q)
        g[0:3, -1] = p
        return g


if __name__ == '__main__':
    rospy.init_node('odom_convert_matrix')
    path = rospy.get_param('~save_file_dir')
    pose_converter = PoseCoverter()
    rospy.spin()
    if(rospy.is_shutdown()):
        np.savetxt(path + "/odom_poses.txt", pose_converter.odom_array, fmt='%.5f', delimiter=' ')
        np.savetxt(path + "/times.txt", pose_converter.time_array, fmt='%.20f', delimiter=' ')
