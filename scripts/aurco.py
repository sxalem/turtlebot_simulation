#!/usr/bin/env python3
"""
Aruco detection node with robust axis drawing via cv2.drawFrameAxes.

Publishes:
  - /measured_data   : [marker_id, x_w, y_w, z_cam] (Float32MultiArray)

Subscribes:
  - /turtlebot/kobuki/realsense/color/image_color
  - /turtlebot/kobuki/realsense/color/camera_info
  - /turtlebot/kobuki/odom    (Odometry)
"""
import sys
import rospy
import cv2
import numpy as np
import math
import tf
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry

class ArucoDetection:

    def __init__(self):
        # 1) Initialize ROS node
        rospy.init_node('aruco_detection', anonymous=True)
        rospy.loginfo("Starting ArucoDetection node...")

        # 2) Internal state
        self.bridge       = CvBridge()
        self.camera       = None
        self.distance_cof = None
        self.robot_pose   = None  # [x, y, yaw]

        # 3) ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_ARUCO_ORIGINAL
        )
        # Fallback for DetectorParameters
        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters()
            rospy.logwarn("Using cv2.aruco.DetectorParameters() fallback")

        # 4) Publisher
        self.measured_pub = rospy.Publisher(
            '/measured_data',
            Float32MultiArray,
            queue_size=1
        )

        # 5) Subscribers (use the exact “sensors” topics)
        rospy.Subscriber(
            '/turtlebot/kobuki/realsense/color/camera_info',
            CameraInfo,
            self.camera_info_callback
        )
        rospy.Subscriber(
            '/turtlebot/kobuki/odom',
            Odometry,
            self.odom_callback
        )
        rospy.Subscriber(
            '/turtlebot/kobuki/realsense/color/image_color',
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24
        )

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera is None:
            self.camera       = np.array(msg.K).reshape((3,3))
            self.distance_cof = np.array(msg.D)
            rospy.loginfo("Camera calibration received.")

    def odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )
        p = msg.pose.pose.position
        self.robot_pose = np.array([p.x, p.y, yaw])

    def image_callback(self, img_msg: Image):
        # Wait until camera calibration & odom available
        if self.camera is None or self.distance_cof is None:
            rospy.logwarn_throttle(5, "Waiting for camera calibration...")
            return
        if self.robot_pose is None:
            rospy.logwarn_throttle(5, "Waiting for odometry...")
            return

        # Convert ROS image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridgeError: {e}")
            return

        # Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame, self.aruco_dict, parameters=self.aruco_params
        )
        if ids is None or len(ids) == 0:
            return

        # Draw and estimate pose
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.05, self.camera, self.distance_cof
        )

        for i, marker_id in enumerate(ids.flatten()):
            rvec = rvecs[i]
            tvec = tvecs[i].copy()
            tvec[0][2] += 0.035  # lift frame

            # Draw axes
            cv2.drawFrameAxes(
                frame,
                self.camera, self.distance_cof,
                rvec, tvec,
                0.05
            )

            # Build camera→marker transform
            R_c_m, _ = cv2.Rodrigues(rvec)
            T_c_m = np.eye(4)
            T_c_m[:3,:3] = R_c_m
            T_c_m[:3, 3] = tvec.flatten()

            # Fixed robot_base→camera
            T_r_c = np.array([
                [0, 0, 1,  0.122],
                [1, 0, 0, -0.033],
                [0, 1, 0,  0.082],
                [0, 0, 0,      1]
            ])

            # robot_base→marker
            T_r_m = T_r_c.dot(T_c_m)
            x_cam, y_cam, z_cam = T_r_m[0,3], T_r_m[1,3], T_r_m[2,3]

            # World coords via robot_pose
            xr, yr, yaw = self.robot_pose
            x_w = x_cam * math.cos(yaw) - y_cam * math.sin(yaw) + xr
            y_w = x_cam * math.sin(yaw) + y_cam * math.cos(yaw) + yr

            # Publish
            msg = Float32MultiArray(data=[
                float(marker_id), x_w, y_w, z_cam
            ])
            self.measured_pub.publish(msg)
            rospy.loginfo(f"Published /measured_data: id={marker_id}, x={x_w:.2f}, y={y_w:.2f}, z={z_cam:.2f}")

        # (Optional) display the image
        cv2.imshow("Aruco Detection", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    try:
        node = ArucoDetection()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
