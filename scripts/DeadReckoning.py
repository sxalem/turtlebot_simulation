#!/usr/bin/env python3
# imports
import numpy as np
import rospy
import math
from std_msgs.msg import Header, ColorRGBA
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import JointState, Imu
import tf
from geometry_msgs.msg import PoseStamped,Quaternion
from visualization_msgs.msg import Marker, MarkerArray

'AUTHOR: Salim Alblooshi'

class DeadReckoning:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('DeadReckoning', anonymous=True)  
        # Robot param
        self.xB_dim = 3 # robot state dimensions
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_velocity_received = False
        self.right_wheel_velocity_received = False
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.imu_received = False


        # Initial robot state 
        self.xk= np.zeros((self.xB_dim,1))
        self.Pk = (0.2) *np.eye(self.xB_dim,self.xB_dim)

         # Odomatry noise covariance
        self.rightwheelnoise_sigma = 0.2
        self.leftwheelnoise_sigma = 0.2
        self.Qk = np.diag(np.array([self.rightwheelnoise_sigma**2, self.leftwheelnoise_sigma**2]))

        # time 
        self.time_last = rospy.Time.now()  
        self.dt = 0

        # ros subscribers
        self.JS_sub=rospy.Subscriber("turtlebot/joint_states", JointState, self.joint_states_callback, queue_size=1)
        
        # ros publishers
        self.odom_pub = rospy.Publisher("turtlebot/kobuki/odom", Odometry, queue_size=1) 

        
        self.imu_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/imu_data", Imu, self.imu_callback)

        # TF broadcaster 
        self.TF_bradcaster = tf.TransformBroadcaster()
    
    def grt_callback(self, msg):
        # Store the latest ground truth position and orientation
        self.grt_x = msg.pose.pose.position.x
        self.grt_y = msg.pose.pose.position.y
        self.grt_q = msg.pose.pose.orientation
        self.grt_received = True

    
    def joint_states_callback(self, msg):
        """
        Proccess msg recived from joint state topic
        """
         # Topic names for the left and right wheel joints 
        self.left_wheel_name = 'turtlebot/kobuki/wheel_left_joint'
        self.right_wheel_name = 'turtlebot/kobuki/wheel_right_joint'
        
        # check if left or right wheel is being published
        if msg.name[0] == self.left_wheel_name:
            self.left_wheel_velocity = msg.velocity[0]
            self.left_wheel_velocity_received = True   
                
        elif msg.name[0] == self.right_wheel_name:
            self.right_wheel_velocity = msg.velocity[0]
            self.right_wheel_velocity_received = True

        if self.left_wheel_velocity_received and self.right_wheel_velocity_received:
            # Get the left wheel and right wheel velocity
            self.left_linear_velocity = self.left_wheel_velocity * self.wheel_radius
            self.right_linear_velocity = self.right_wheel_velocity * self.wheel_radius

            # Get the linear and angular velocity
            self.linear_velocity = (self.left_linear_velocity + self.right_linear_velocity) / 2
            self.angular_velocity = (self.left_linear_velocity - self.right_linear_velocity) / self.wheel_base_distance

            # calculate dt
            self.time_now = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            self.dt = (self.time_now - self.time_last).to_sec()
            self.time_last = self.time_now

            # Calculate motion model of the robot pose and oridentation
            self.xk[0,0] = self.xk[0,0] +  (np.cos(self.xk[2,0]) * self.linear_velocity * self.dt)
            self.xk[1,0] = self.xk[1,0] + (np.sin(self.xk[2,0]) * self.linear_velocity * self.dt)
            theta = self.xk[2,0] + (self.angular_velocity * self.dt)
            self.xk[2,0] = self.wrap_angle(theta)

              # Jacobian of the motion model with respect to the xk
            Ak = np.array([
                [1.0, 0.0, -np.sin(self.xk[2,0]) * self.linear_velocity * self.dt],
                [0.0, 1.0,  np.cos(self.xk[2,0]) * self.linear_velocity * self.dt],
                [0.0, 0.0, 1.0]])

            # Jacobian of the motion model with respect to noise
            Wk = np.array([
                [(0.5 * np.cos(self.xk[2,0]) * self.dt), (0.5 * np.cos(self.xk[2,0]) * self.dt)],
                [(0.5 * np.sin(self.xk[2,0]) * self.dt), (0.5 * np.sin(self.xk[2,0]) * self.dt)],
                [-self.dt/self.wheel_base_distance, self.dt/self.wheel_base_distance]])
            
            self.Pk = (Ak @ self.Pk @ Ak.T) + (Wk @ self.Qk @ Wk.T)
            self.xk = self.xk

            


            self.left_wheel_velocity_received = False
            self.right_wheel_velocity_received = False  

    def imu_callback(self, msg):
        orientation_q = msg.orientation
        quaternion = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, imu_yaw = euler_from_quaternion(quaternion)
        self.imu_received = True

        H = np.array([[0, 0, 1]])
        R = np.array([[np.deg2rad(5.0)**2]])
        z = np.array([[imu_yaw]])
        z_pred = H @ self.xk
        y = z - z_pred
        y[0,0] = self.wrap_angle(y[0,0])
        S = H @ self.Pk @ H.T + R
        K = self.Pk @ H.T @ np.linalg.inv(S)
        self.xk = self.xk + K @ y
        self.xk[2,0] = self.wrap_angle(self.xk[2,0])
        self.Pk = (np.eye(self.xB_dim) - K @ H) @ self.Pk

        # Publish odometry and path after prediction + update
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"
        odom.pose.pose.position.x = self.xk[0,0]
        odom.pose.pose.position.y = self.xk[1,0]
        self.q = quaternion_from_euler(0, 0, self.xk[2,0])
        odom.pose.pose.orientation = Quaternion(*self.q)
        odom.twist.twist.linear.x = self.linear_velocity
        odom.twist.twist.angular.z = self.angular_velocity
        odom.pose.covariance = [self.Pk[0, 0], self.Pk[0, 1], 0, 0, 0, self.Pk[0, 2],self.Pk[1, 0], self.Pk[1, 1], 0, 0, 0, self.Pk[1, 2], 
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.Pk[2, 0], self.Pk[2, 1], 0, 0, 0, self.Pk[2, 2]]

        self.odom_pub.publish(odom)
        self.TF_bradcaster.sendTransform((self.xk[0,0], self.xk[1,0], 0.0), self.q, rospy.Time.now(), odom.child_frame_id, odom.header.frame_id)
        self.imu_received = False


    def wrap_angle(self, angle):  
        """
        Wrap the angle between [-pi, pi], prevents the robot from taking unnecessary long rotations.
        """
        return (angle + (2.0 * np.pi * np.floor((np.pi - angle) / (2.0 * np.pi))))
    

  

if __name__ == '__main__':
    try:
        robot = DeadReckoning()
        # DeadReckoning()
        rospy.spin() 
    except rospy.ROSInterruptException:
        pass


