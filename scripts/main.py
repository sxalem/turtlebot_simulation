#!/usr/bin/env python3

'AUTHOR: Salim Alblooshi'

import rospy
import numpy as np
import tf
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Bool
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry, Path
from tf.transformations import quaternion_from_euler
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from MB import forward_kinematics, compute_jacobian, weighted_DLS, PositionTask3D, JointAngleTask, JointBoundaryTask


class TaskPriorityController():
    
    def process_odometry(self, odometry_msg):
        _, _, yaw_angle = tf.transformations.euler_from_quaternion([
            odometry_msg.pose.pose.orientation.x,
            odometry_msg.pose.pose.orientation.y,
            odometry_msg.pose.pose.orientation.z,
            odometry_msg.pose.pose.orientation.w
        ])
        self.robot_position = np.array([
            odometry_msg.pose.pose.position.x,
            odometry_msg.pose.pose.position.y,
            yaw_angle
        ])
    
    def publish_ee_trajectory(self, transformation_matrix, joint_angles):
        ee_position = transformation_matrix[:, -1]
        ee_pose_stamped = PoseStamped()
        ee_pose_stamped.header.stamp = rospy.Time.now()
        ee_pose_stamped.header.frame_id = self.ee_path_message.header.frame_id
        ee_pose_stamped.pose.position.x = ee_position[0]
        ee_pose_stamped.pose.position.y = ee_position[1]
        ee_pose_stamped.pose.position.z = ee_position[2]
        quaternion = quaternion_from_euler(0, 0, joint_angles[3])
        ee_pose_stamped.pose.orientation.x = quaternion[0]
        ee_pose_stamped.pose.orientation.y = quaternion[1]
        ee_pose_stamped.pose.orientation.z = quaternion[2]
        ee_pose_stamped.pose.orientation.w = quaternion[3]
        self.ee_path_message.header.stamp = ee_pose_stamped.header.stamp
        self.ee_path_message.poses.append(ee_pose_stamped)
        self.ee_trajectory_pub.publish(self.ee_path_message)
    
    def deactivate_gripper(self):
        rospy.wait_for_service("/turtlebot/swiftpro/vacuum_gripper/set_pump")
        try:
            gripper_service = rospy.ServiceProxy("/turtlebot/swiftpro/vacuum_gripper/set_pump", SetBool)
            response = gripper_service(False)
            if response.success:
                rospy.loginfo("Vacuum pump deactivated")
            else:
                rospy.logwarn("Failed to deactivate pump: %s", response.message)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
    
    def activate_gripper(self):
        rospy.wait_for_service("/turtlebot/swiftpro/vacuum_gripper/set_pump")
        try:
            gripper_service = rospy.ServiceProxy("/turtlebot/swiftpro/vacuum_gripper/set_pump", SetBool)
            response = gripper_service(True)
            if response.success:
                rospy.loginfo("Vacuum pump activated")
            else:
                rospy.logwarn("Failed to activate pump: %s", response.message)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
    
    def publish_target_marker(self, target_position):
        target_marker = Marker()
        target_marker.header.frame_id = "world_ned"
        target_marker.header.stamp = rospy.Time.now()
        target_marker.type = target_marker.SPHERE
        target_marker.action = target_marker.ADD
        target_marker.pose.position.x = target_position[0]
        target_marker.pose.position.y = target_position[1]
        target_marker.pose.position.z = target_position[2]
        target_marker.scale.x = 0.05
        target_marker.scale.y = 0.05
        target_marker.scale.z = 0.05
        target_marker.color.r = 0.5
        target_marker.color.g = 0.1
        target_marker.color.a = 1
        self.target_marker_pub.publish(target_marker)
    
    def execute_movement(self, velocity_commands):
        # Send arm commands
        self.manipulator_publisher.publish(Float64MultiArray(data=[
            float(velocity_commands[2,0]),
            float(velocity_commands[3,0]),
            float(velocity_commands[4,0]),
            float(velocity_commands[5,0])
        ]))
        
        # Calculate wheel velocities
        linear_vel = velocity_commands[0,0]
        angular_vel = velocity_commands[1,0]
        wheel_base = 0.230
        wheel_radius = 0.035
        
        left_lin_vel = linear_vel - (angular_vel * wheel_base / 2.0)
        right_lin_vel = linear_vel + (angular_vel * wheel_base / 2.0)
        left_wheel_vel = left_lin_vel / wheel_radius
        right_wheel_vel = right_lin_vel / wheel_radius
        
        self.mobile_base_publisher.publish(Float64MultiArray(data=[right_wheel_vel, left_wheel_vel]))
    
    def execute_control(self, joint_state_msg):
        # Task configuration based on sequence
        if self.current_sequence == 1:
            self.priority_weights = np.diag([500, 500, 1, 1, 1, 1])
            self.task_list = [
                PositionTask3D("End-effector target", self.desired_position)
            ]
        elif self.current_sequence == 2:
            self.priority_weights = np.diag([5000, 5000, 1, 1, 1, 1])
            self.task_list = [
                JointAngleTask("Joint angle 3", np.array([0]).reshape(1,1), link=4),
                JointAngleTask("Joint angle 1", np.array([np.pi/2]).reshape(1,1), link=2)
            ]
        elif self.current_sequence == 3:
            self.priority_weights = np.diag([5000, 5000, 1, 1, 1, 1])
            self.task_list = [
                JointAngleTask("Joint angle 3", np.array([0]).reshape(1,1), link=4),
                JointAngleTask("Joint angle 2", np.array([0.04]).reshape(1,1), link=3)
            ]
        elif self.current_sequence == 4:
            self.priority_weights = np.diag([1, 1, 1, 1, 1, 1])
            self.desired_position = np.array([0, 0, -0.33]).reshape(3,1)
            self.task_list = [
                JointAngleTask("Joint angle 1", np.array([np.pi/2]).reshape(1,1), link=2),
                PositionTask3D("End-effector target", np.array([2, 2, -0.24]).reshape(3,1))
            ]
        elif self.current_sequence == 5:
            self.priority_weights = np.diag([2000, 2000, 1, 1, 1, 1])
            self.task_list = [
                PositionTask3D("End-effector target", np.array([2, 2, -0.29]).reshape(3,1))
            ]
        elif self.current_sequence == 6:
            self.priority_weights = np.diag([5000, 5000, 1, 1, 1, 1])
            self.task_list = [
                JointAngleTask("Joint angle 3", np.array([0]).reshape(1,1), link=4),
                JointAngleTask("Joint angle 2", np.array([0.04]).reshape(1,1), link=3),
                JointAngleTask("Joint angle 1", np.array([-np.pi/2]).reshape(1,1), link=2)
            ]
        
        # Check for required joints
        required_joints = [
            'turtlebot/swiftpro/joint1',
            'turtlebot/swiftpro/joint2',
            'turtlebot/swiftpro/joint3',
            'turtlebot/swiftpro/joint4'
        ]
        
        if not all(joint in joint_state_msg.name for joint in required_joints):
            return
        
        if self.desired_position is None:
            self.manipulator_publisher.publish(Float64MultiArray(data=[0,0,0,0]))
            self.mobile_base_publisher.publish(Float64MultiArray(data=[0.0,0.0]))
            return
        
        # Extract joint angles
        joint_map = dict(zip(joint_state_msg.name, joint_state_msg.position))
        current_joints = np.array([
            joint_map['turtlebot/swiftpro/joint1'],
            joint_map['turtlebot/swiftpro/joint2'],
            joint_map['turtlebot/swiftpro/joint3'],
            joint_map['turtlebot/swiftpro/joint4']
        ])
        
        self.publish_target_marker(self.desired_position)
        
        # Forward kinematics
        T_matrix = forward_kinematics(current_joints, self.robot_position)
        ee_pos = T_matrix[0:3, 3]
        R_matrix = T_matrix[0:3, 0:3]
        
        # Task priority algorithm
        velocities = np.zeros((self.degrees_of_freedom, 1))
        null_space = np.eye(self.degrees_of_freedom)
        
        for task in self.task_list:
            task.update(current_joints, self.robot_position)
            if task.isActive() != 0:
                J_proj = task.getJacobian() @ null_space
                velocities += weighted_DLS(J_proj, self.damping_factor, self.priority_weights) @ (task.getError() - J_proj @ velocities)
                null_space -= np.linalg.pinv(J_proj) @ J_proj
        
        pos_error = self.task_list[-1].getError()
        
        # Log data
        self.joint_vel_1.append(velocities[0])
        self.joint_vel_2.append(velocities[1])
        self.joint_vel_3.append(velocities[2])
        self.joint_vel_4.append(velocities[3])
        self.joint_vel_5.append(velocities[4])
        self.joint_vel_6.append(velocities[5])
        self.endeffector_x.append(ee_pos[0])
        self.endeffector_y.append(ee_pos[1])
        self.base_x.append(self.robot_position[0])
        self.base_y.append(self.robot_position[1])
        
        if self.timestamp_log == []:
            self.timestamp_log.append(0)
        else:
            self.timestamp_log.append(self.timestamp_log[-1] + 1.0/60.0)
        
        # Sequence-specific control logic
        if self.current_sequence == 0:
            self.tracking_error.append(np.linalg.norm(pos_error))
            if self.error_log == []:
                self.error_log.append(0)
            else:
                self.error_log.append(self.error_log[-1] + 1.0/60.0)
            
            if abs(pos_error[0,0]) < 0.01 and abs(pos_error[1,0]) < 0.01:
                rospy.loginfo("Reached object location - Task 0 complete --> Task 1")
                self.current_sequence = 1
                return
            
            # Orientation control
            tgt_x, tgt_y = self.desired_position[0,0], self.desired_position[1,0]
            rob_x, rob_y, rob_yaw = self.robot_position
            desired_yaw = np.arctan2(tgt_y - rob_y, tgt_x - rob_x)
            yaw_err = desired_yaw - rob_yaw
            yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
            velocities[1,0] = yaw_err * 1.5
            
            dist = np.linalg.norm([tgt_x - rob_x, tgt_y - rob_y])
            if dist < 0.3:
                velocities[0,0] *= 0.5
            
            velocities = velocities / 2
            self.execute_movement(velocities)
            
        elif self.current_sequence == 1:
            self.tracking_error.append(np.linalg.norm(pos_error))
            if self.error_log == []:
                self.error_log.append(0)
            else:
                self.error_log.append(self.error_log[-1] + 1.0/60.0)
            
            if pos_error[2,0] < 0.01 and abs(pos_error[0,0]) < 0.01 and abs(pos_error[1,0]) < 0.01:
                self.activate_gripper()
                rospy.loginfo("Object grasped - Task 1 complete --> Task 2")
                self.current_sequence = 2
                return
            
            velocities = velocities / 3
            self.execute_movement(velocities)
            
        elif self.current_sequence == 2:
            if int(current_joints[2]) == 0 and round(current_joints[0], 2) == 1.57:
                rospy.loginfo("Object lifted - Task 2 complete --> Task 3")
                self.current_sequence = 3
                return
            velocities = velocities / 4
            self.execute_movement(velocities)
            
        elif self.current_sequence == 3:
            if int(current_joints[2]) == 0 and round(current_joints[1], 2) == 0.04:
                rospy.loginfo("Object ready for new place - Task 3 complete --> Task 4")
                self.current_sequence = 4
                return
            velocities = velocities / 4
            self.execute_movement(velocities)
            
        elif self.current_sequence == 4:
            self.tracking_error.append(np.linalg.norm(pos_error))
            if self.error_log == []:
                self.error_log.append(0)
            else:
                self.error_log.append(self.error_log[-1] + 1.0/60.0)
            
            if abs(pos_error[0,0]) < 0.1 and abs(pos_error[1,0]) < 0.1:
                rospy.loginfo("Ready for drop operation - Task 4 complete --> Task 5")
                self.current_sequence = 5
                self.manipulator_publisher.publish(Float64MultiArray(data=[0,0,0,0]))
                self.mobile_base_publisher.publish(Float64MultiArray(data=[0,0]))
                return
            velocities = velocities / 2
            self.execute_movement(velocities)
            
        elif self.current_sequence == 5:
            self.tracking_error.append(np.linalg.norm(pos_error))
            if self.error_log == []:
                self.error_log.append(0)
            else:
                self.error_log.append(self.error_log[-1] + 1.0/60.0)
            
            if pos_error[2,0] < 0.03:
                rospy.loginfo("Object dropped - Task 5 complete --> Task 6")
                self.deactivate_gripper()
                self.current_sequence = 6
                return
            velocities = velocities / 5
            self.execute_movement(velocities)
            
        elif self.current_sequence == 6:
            if int(current_joints[2]) == 0 and round(current_joints[0], 2) == -1.57 and round(current_joints[1], 2) == 0.04:
                rospy.loginfo("Returned to home position")
                rospy.signal_shutdown("Mission completed")
                return
            velocities = velocities / 4
            self.execute_movement(velocities)
    
    def __init__(self):
        # Initialize state variables
        self.robot_position = np.zeros((3))
        self.degrees_of_freedom = 6
        self.priority_weights = np.diag([0.4, 0.8, 1, 1, 1, 1])
        self.damping_factor = 0.04
        self.marker_detected = True
        self.target_achieved = False
        self.current_sequence = 0
        self.release_object = False
        
        # Initialize logging arrays
        self.timestamp_log = []
        self.error_log = []
        self.joint_vel_1 = []
        self.joint_vel_2 = []
        self.joint_vel_3 = []
        self.joint_vel_4 = []
        self.joint_vel_5 = []
        self.joint_vel_6 = []
        self.endeffector_x = []
        self.endeffector_y = []
        self.base_x = []
        self.base_y = []
        self.tracking_error = []
        
        # Get target from sensor
        try:
            sensor_msg = rospy.wait_for_message("measured_data", Float32MultiArray, timeout=5.0)
            target_x = sensor_msg.data[1]
            target_y = sensor_msg.data[2]
            self.desired_position = np.array([target_x+0.03, target_y+0.01, -0.145]).reshape(3,1)
        except rospy.ROSException as e:
            rospy.logwarn("Timeout waiting for sensor data: %s", e)
        
        rospy.sleep(1.0)
        
        # Initialize tasks
        self.task_list = [
            JointAngleTask("Joint angle 1", np.array([np.pi/2]).reshape(1,1), link=2),
            JointBoundaryTask("Joint boundary 2", np.array([0.03, 0.04]), np.array([np.pi/2, 0]), link=3),
            JointBoundaryTask("Joint boundary 3", np.array([0.03, 0.04]), np.array([0.1, 1.5]), link=4),
            PositionTask3D("End-effector target", self.desired_position)
        ]
        
        # Setup publishers
        self.manipulator_publisher = rospy.Publisher(
            "/turtlebot/swiftpro/joint_velocity_controller/command",
            Float64MultiArray,
            queue_size=1
        )
        self.mobile_base_publisher = rospy.Publisher(
            "/turtlebot/kobuki/commands/wheel_velocities",
            Float64MultiArray,
            queue_size=1
        )
        self.target_marker_pub = rospy.Publisher('goal_pose', Marker, queue_size=1)
        self.ee_trajectory_pub = rospy.Publisher('/position_EE', Path, queue_size=1)
        
        # Initialize path message
        self.ee_path_message = Path()
        self.ee_path_message.header.frame_id = "world_ned"
        
        # Setup subscribers
        self.odometry_subscriber = rospy.Subscriber(
            '/turtlebot/kobuki/odom',
            Odometry,
            self.process_odometry
        )
        self.joint_state_subscriber = rospy.Subscriber(
            "/turtlebot/joint_states",
            JointState,
            self.execute_control
        )


def plot_results(controller):
    # Error plot
    plt.figure()
    plt.plot(controller.error_log, controller.tracking_error, label='Position tracking error')
    plt.xlabel('Time Step(s)')
    plt.ylabel('Error (m)')
    plt.title('Control Error Evolution Over Time')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Trajectory plot
    plt.figure()
    plt.plot(controller.endeffector_x, controller.endeffector_y, label='End-effector trajectory')
    plt.plot(controller.base_x, controller.base_y, label='Mobile base trajectory')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Motion Trajectories in X-Y Plane')
    plt.grid()
    plt.legend()
    plt.show()
    
    # Velocity plot
    plt.figure()
    plt.plot(controller.timestamp_log, controller.joint_vel_1, label='Joint velocity 1')
    plt.plot(controller.timestamp_log, controller.joint_vel_2, label='Joint velocity 2')
    plt.plot(controller.timestamp_log, controller.joint_vel_3, label='Joint velocity 3')
    plt.plot(controller.timestamp_log, controller.joint_vel_4, label='Joint velocity 4')
    plt.plot(controller.timestamp_log, controller.joint_vel_5, label='Joint velocity 5')
    plt.plot(controller.timestamp_log, controller.joint_vel_6, label='Joint velocity 6')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    plt.title('Joint Velocity Evolution Over Time')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    rospy.init_node('mobile_manipulator_controller', anonymous=True)
    controller_node = TaskPriorityController()
    rospy.spin()
    
    # Generate plots after completion
    plot_results(controller_node)





































































