#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float32MultiArray, Bool
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry, Path
from tf.transformations import quaternion_from_euler
import tf
import numpy as np
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from MB import forward_kinematics, compute_jacobian, weighted_DLS, PositionTask3D, JointAngleTask, JointBoundaryTask 
import matplotlib.pyplot as plt

'AUTHOR: Salim Alblooshi'

class TaskPriorityController():

    def __init__(self):      
        self.robot_position = np.zeros((3))
        self.degrees_of_freedom = 6
        self.priority_weights = np.diag([0.4, 0.8, 1, 1, 1, 1]) 
        self.damping_factor = 0.04
        self.marker_detected = True
        self.target_achieved = False
        self.current_sequence = 0
        self.release_object = False
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

        try:
            sensor_msg = rospy.wait_for_message("measured_data",Float32MultiArray,timeout=5.0)  
            target_x = sensor_msg.data[1]
            target_y = sensor_msg.data[2]
            self.desired_position = np.array([target_x+0.03, target_y+0.01, -0.145]).reshape(3,1)
        except rospy.ROSException as e:
            rospy.logwarn("Timeout waiting for sensor data: %s", e)
        rospy.sleep(1.0)
        self.task_list = [
                JointAngleTask("Joint angle 1",np.array([np.pi/2]).reshape(1,1), link = 2),
                JointBoundaryTask("Joint boundary 2", np.array([0.03, 0.04]), np.array([np.pi/2, 0]), link = 3),
                JointBoundaryTask("Joint boundary 3", np.array([0.03, 0.04]), np.array([0.1, 1.5]), link = 4),
                PositionTask3D("End-effector target", self.desired_position)] 
        self.manipulator_publisher = rospy.Publisher("/turtlebot/swiftpro/joint_velocity_controller/command", Float64MultiArray, queue_size=1) 
        
        self.mobile_base_publisher = rospy.Publisher("/turtlebot/kobuki/commands/wheel_velocities", Float64MultiArray, queue_size=1)
        self.target_marker_pub = rospy.Publisher('goal_pose', Marker, queue_size=1)
        self.ee_trajectory_pub = rospy.Publisher('/position_EE', Path, queue_size=1)
        self.ee_path_message = Path()
        self.ee_path_message.header.frame_id = "world_ned"
        self.odometry_subscriber = rospy.Subscriber('/turtlebot/kobuki/odom', Odometry, self.process_odometry) 
        self.joint_state_subscriber = rospy.Subscriber("/turtlebot/joint_states", JointState, self.execute_control)   
        
        

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
            
    def execute_movement(self, velocity_commands):
        # Send manipulator joint velocities
        self.manipulator_publisher.publish(Float64MultiArray(data=[float(velocity_commands[2,0]),float(velocity_commands[3,0]),float(velocity_commands[4,0]),float(velocity_commands[5,0])]))
        # Calculate and send wheel velocities
        linear_velocity = velocity_commands[0,0]
        angular_velocity = velocity_commands[1,0]
        left_linear_vel = linear_velocity - (angular_velocity * 0.230  / 2.0)
        right_linear_vel = linear_velocity + (angular_velocity * 0.230  / 2.0)
        left_wheel_velocity = left_linear_vel / 0.035
        right_wheel_velocity = right_linear_vel / 0.035
        self.mobile_base_publisher.publish(Float64MultiArray(data = [right_wheel_velocity, left_wheel_velocity]))

    def execute_control(self, joint_state_msg):
        if self.current_sequence == 1:
            self.priority_weights = np.diag([500, 500, 1, 1, 1, 1])
            self.task_list = [PositionTask3D("End-effector target", self.desired_position)]
        elif self.current_sequence == 2:
            self.priority_weights = np.diag([5000, 5000, 1, 1, 1, 1])
            self.task_list = [
                JointAngleTask("Joint angle 3",np.array([0]).reshape(1,1), link = 4),
                JointAngleTask("Joint angle 1",np.array([np.pi/2]).reshape(1,1), link = 2)] 
        elif self.current_sequence == 3:
            self.priority_weights = np.diag([5000, 5000, 1, 1, 1, 1])
            self.task_list = [
                JointAngleTask("Joint angle 3",np.array([0]).reshape(1,1), link = 4),
                JointAngleTask("Joint angle 2",np.array([0.04]).reshape(1,1), link = 3),
                ] 

        elif self.current_sequence == 4:
            self.priority_weights = np.diag([1, 1, 1, 1, 1, 1])
            self.desired_position =  np.array([0, 0, -0.33]).reshape(3,1)
            self.task_list = [
                JointAngleTask("Joint angle 1",np.array([np.pi/2]).reshape(1,1), link = 2),
                PositionTask3D("End-effector target", np.array([2,2, -0.24]).reshape(3,1))
                 ] 
        elif self.current_sequence == 5:
            self.priority_weights = np.diag([2000, 2000, 1, 1, 1, 1])
            self.task_list = [PositionTask3D("End-effector target", np.array([2, 2, -0.29]).reshape(3,1))]
        elif self.current_sequence == 6:
            self.priority_weights = np.diag([5000, 5000, 1, 1, 1, 1])
            self.task_list = [
                JointAngleTask("Joint angle 3",np.array([0]).reshape(1,1), link = 4),
                JointAngleTask("Joint angle 2",np.array([0.04]).reshape(1,1), link = 3),
                JointAngleTask("Joint angle 1",np.array([-np.pi/2]).reshape(1,1), link = 2)] 

        required_joints = ['turtlebot/swiftpro/joint1','turtlebot/swiftpro/joint2','turtlebot/swiftpro/joint3','turtlebot/swiftpro/joint4']
        if not all(joint_name in joint_state_msg.name for joint_name in required_joints):
            return   
        if self.desired_position is None:
            self.manipulator_publisher.publish(Float64MultiArray(data=[0,0,0,0]))
            self.mobile_base_publisher.publish(Float64MultiArray(data=[0.0,0.0]))
            return
        joint_position_map = dict(zip(joint_state_msg.name, joint_state_msg.position))
        current_joint_angles = np.array([joint_position_map['turtlebot/swiftpro/joint1'],joint_position_map['turtlebot/swiftpro/joint2'],joint_position_map['turtlebot/swiftpro/joint3'],joint_position_map['turtlebot/swiftpro/joint4']])
        self.publish_target_marker(self.desired_position)

        transformation_matrix = forward_kinematics(current_joint_angles,self.robot_position)
        end_effector_position = transformation_matrix[0:3,3]
        rotation_matrix = transformation_matrix[0:3,0:3]
        
        # Task-priority control algorithm
        joint_velocities = np.zeros((self.degrees_of_freedom,1))
        projection_matrix  = np.eye(self.degrees_of_freedom)
        for current_task in self.task_list:
            current_task.update(current_joint_angles, self.robot_position)
            if current_task.isActive() != 0:
                projected_jacobian = current_task.getJacobian() @ projection_matrix
                joint_velocities += weighted_DLS(projected_jacobian, self.damping_factor, self.priority_weights) @ (current_task.getError() - projected_jacobian @ joint_velocities)
                projection_matrix  -= np.linalg.pinv(projected_jacobian) @ projected_jacobian    
        position_error = self.task_list[-1].getError()  # 3×1: [ex, ey, ez]
        
        # Data logging for analysis
        self.joint_vel_1.append(joint_velocities[0])
        self.joint_vel_2.append(joint_velocities[1])
        self.joint_vel_3.append(joint_velocities[2])
        self.joint_vel_4.append(joint_velocities[3])
        self.joint_vel_5.append(joint_velocities[4])
        self.joint_vel_6.append(joint_velocities[5])
        self.endeffector_x.append(end_effector_position[0])
        self.endeffector_y.append(end_effector_position[1])
        self.base_x.append(self.robot_position[0])
        self.base_y.append(self.robot_position[1])
        if self.timestamp_log == []:
            self.timestamp_log.append(0)
        else:
            self.timestamp_log.append(self.timestamp_log[-1] + 1.0 / 60.0)

        if self.current_sequence == 0:
            self.tracking_error.append(np.linalg.norm(position_error))
            if self.error_log == []:
                self.error_log.append(0)
            else:
                self.error_log.append(self.error_log[-1] + 1.0 / 60.0)

            # Convergence check
            if (abs(position_error[0,0]) < 0.01 and abs(position_error[1,0]) < 0.01):
                rospy.loginfo("Reached object location - Task 0 complete --> Task 1")
                self.current_sequence = 1
                return

            # Base orientation alignment
            target_x, target_y = self.desired_position[0,0], self.desired_position[1,0]
            robot_x, robot_y, robot_yaw = self.robot_position
            required_yaw = np.arctan2(target_y - robot_y, target_x - robot_x)
            yaw_difference = required_yaw - robot_yaw
            yaw_difference = np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference))
            joint_velocities[1,0] = yaw_difference * 1.5  # Tunable gain parameter
            
            distance_to_target = np.linalg.norm([target_x - robot_x, target_y - robot_y])
            if distance_to_target < 0.3:
                joint_velocities[0,0] *= 0.5

            joint_velocities = joint_velocities / 2
            self.execute_movement(joint_velocities)
            joint_velocities = joint_velocities/2
            self.execute_movement(joint_velocities)

        elif  self.current_sequence == 1:
            self.tracking_error.append(np.linalg.norm(position_error))
            if self.error_log == []:
                self.error_log.append(0)
            else:
                self.error_log.append(self.error_log[-1] + 1.0 / 60.0)
            if position_error[2,0]< 0.01  and abs(position_error[0,0]) < 0.01 and abs(position_error[1,0]) < 0.01:
                self.activate_gripper()
                rospy.loginfo("Object grasped - Task 1 complete --> Task 2")
                self.current_sequence = 2        
                return
            
            joint_velocities = joint_velocities/3   
            self.execute_movement(joint_velocities)  
        elif self.current_sequence == 2:
            if int(current_joint_angles[2]) == 0 and round(current_joint_angles[0],2) == 1.57:
                rospy.loginfo("Object lifted - Task 2 complete --> Task 3")
                self.current_sequence = 3
                return
            joint_velocities = joint_velocities/4
            self.execute_movement(joint_velocities)
        elif self.current_sequence == 3:
            if int(current_joint_angles[2]) == 0 and round(current_joint_angles[1],2) == 0.04:
                rospy.loginfo("Object ready for new place - Task 3 complete --> Task 4")
                self.current_sequence = 4
                return
            joint_velocities = joint_velocities/4
            self.execute_movement(joint_velocities)

        elif self.current_sequence == 4:
            self.tracking_error.append(np.linalg.norm(position_error))
            if self.error_log == []:
                self.error_log.append(0)
            else:
                self.error_log.append(self.error_log[-1] + 1.0 / 60.0)
            if abs(position_error[0,0]) < 0.1 and abs(position_error[1,0]) < 0.1:
                rospy.loginfo("Ready for drop operation - Task 4 complete --> Task 5")
                self.current_sequence = 5
                self.manipulator_publisher.publish(Float64MultiArray(data=[0,0,0,0]))
                self.mobile_base_publisher.publish(Float64MultiArray(data = [0, 0]))
                return
            joint_velocities = joint_velocities/2
            self.execute_movement(joint_velocities)
        elif self.current_sequence == 5:
            self.tracking_error.append(np.linalg.norm(position_error))
            if self.error_log == []:
                self.error_log.append(0)
            else:
                self.error_log.append(self.error_log[-1] + 1.0 / 60.0)
            if position_error[2,0]< 0.03:
                rospy.loginfo("Object dropped - Task 5 complete --> Task 6")
                self.deactivate_gripper()
                self.current_sequence = 6
                return
            joint_velocities = joint_velocities/5
            self.execute_movement(joint_velocities)
        elif self.current_sequence == 6:
            if int(current_joint_angles[2]) == 0 and round(current_joint_angles[0],2) == -1.57 and round(current_joint_angles[1],2) == 0.04:
                rospy.loginfo("Returned to home position")
                rospy.signal_shutdown("Mission completed")
                return
            joint_velocities = joint_velocities/4
            self.execute_movement(joint_velocities)

    def process_odometry(self,odometry_msg):
        _, _, yaw_angle = tf.transformations.euler_from_quaternion([odometry_msg.pose.pose.orientation.x, 
                                                              odometry_msg.pose.pose.orientation.y,
                                                              odometry_msg.pose.pose.orientation.z,
                                                              odometry_msg.pose.pose.orientation.w])
        self.robot_position = np.array([odometry_msg.pose.pose.position.x, odometry_msg.pose.pose.position.y, yaw_angle])
        
    def publish_target_marker(self, target_position):
        target_marker = Marker()
        target_marker.header.frame_id = "world_ned"
        target_marker.type = target_marker.SPHERE
        target_marker.action = target_marker.ADD
        target_marker.header.stamp = rospy.Time.now()
        target_marker.pose.position.x = target_position[0]
        target_marker.pose.position.y = target_position[1]
        target_marker.pose.position.z = target_position[2]
        target_marker.scale.x = 0.05
        target_marker.scale.y = 0.05
        target_marker.scale.z = 0.05
        target_marker.color.g = 0.1
        target_marker.color.r = 0.5
        target_marker.color.a = 1
        self.target_marker_pub.publish(target_marker)
        
    def publish_ee_trajectory(self,transformation_matrix,joint_angles):
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
        
if __name__ == "__main__":
    rospy.init_node('mobile_manipulator_controller', anonymous=True)
    controller_node = TaskPriorityController()
    rospy.spin()

    # Visualization of results
    plt.figure()
    plt.plot(controller_node.error_log, controller_node.tracking_error, label='Position tracking error')
    plt.xlabel('Time Step(s)')
    plt.ylabel('Error (m)')
    plt.title('Control Error Evolution Over Time')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(controller_node.endeffector_x, controller_node.endeffector_y, label='End-effector trajectory')
    plt.plot(controller_node.base_x, controller_node.base_y, label='Mobile base trajectory')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Motion Trajectories in X-Y Plane')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(controller_node.timestamp_log, controller_node.joint_vel_1, label='Joint velocity 1')
    plt.plot(controller_node.timestamp_log, controller_node.joint_vel_2, label='Joint velocity 2')
    plt.plot(controller_node.timestamp_log, controller_node.joint_vel_3, label='Joint velocity 3')
    plt.plot(controller_node.timestamp_log, controller_node.joint_vel_4, label='Joint velocity 4')
    plt.plot(controller_node.timestamp_log, controller_node.joint_vel_5, label='Joint velocity 5')
    plt.plot(controller_node.timestamp_log, controller_node.joint_vel_6, label='Joint velocity 6')
    plt.xlabel('Time Step(s)')
    plt.ylabel('Velocity(m/s)')
    plt.title('Joint Velocity Evolution Over Time')
    plt.grid()
    plt.legend()
    plt.show()









































































    # import rospy
# from sensor_msgs.msg import JointState
# from std_msgs.msg import Float64MultiArray, Float32MultiArray, Bool
# from geometry_msgs.msg import Twist
# from geometry_msgs.msg import PoseStamped
# from visualization_msgs.msg import Marker
# from nav_msgs.msg import Odometry, Path
# from tf.transformations import quaternion_from_euler
# import tf
# import numpy as np
# from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
# from mobilemanipulator import kinematics, jacobian, DLS, Position3D , JointPosition, JointLimitTask 
# import matplotlib.pyplot as plt


# class TP_Control():

#     def __init__(self):      
#         self.robot_pose = np.zeros((3))
#         self.dof = 6
#         self.weights = np.diag([0.3, 0.7, 1, 1, 1, 1])
#         self.damping = 0.05
#         self.aruco_recieved = True
#         self.goal_reached = False
#         self.seq = 0
#         self.drop = False
#         self.time = []
#         self.time_err = []
#         self.q1 = []
#         self.q2 = []
#         self.q3 = []
#         self.q4 = []
#         self.q5 = []
#         self.q6 = []
#         self.EE_x = []
#         self.EE_y = []
#         self.mb_x = []
#         self.mb_y = []
#         self.error = []

#         try:
#             msg = rospy.wait_for_message("measured_data",Float32MultiArray,timeout=5.0)  
#             x = msg.data[1]
#             y = msg.data[2]
#             self.sigma_d = np.array([x+0.05, y, -0.145]).reshape(3,1)
#         except rospy.ROSException as e:
#             rospy.logwarn("Timed out waiting for measured_data: %s", e)
#         rospy.sleep(1.0)
#         self.tasks = [
#                 JointPosition("Joint position 1",np.array([np.pi/2]).reshape(1,1), link = 2),
#                 JointLimitTask("Joint limit 2", np.array([0.03, 0.04]), np.array([np.pi/2, 0]), link = 3),
#                 JointLimitTask("Joint limit 3", np.array([0.03, 0.04]), np.array([0.1, 1.5]), link = 4),
#                 Position3D("End-effector position", self.sigma_d)] 
#         self.pub_joints = rospy.Publisher("/turtlebot/swiftpro/joint_velocity_controller/command", Float64MultiArray, queue_size=1) 
        
#         self.pub_base = rospy.Publisher("/turtlebot/kobuki/commands/wheel_velocities", Float64MultiArray, queue_size=1)
#         self.goalPose_pub = rospy.Publisher('goal_pose', Marker, queue_size=1)
#         self.EE_path_pub = rospy.Publisher('/position_EE', Path, queue_size=1)
#         self.EE_path_msg = Path()
#         self.EE_path_msg.header.frame_id = "world_ned"
#         self.odom_sub = rospy.Subscriber('/turtlebot/kobuki/odom', Odometry, self.get_odom) 
#         self.control_sub = rospy.Subscriber("/turtlebot/joint_states", JointState, self.control)   
        
        

#     def gripper_on(self):
#         rospy.wait_for_service("/turtlebot/swiftpro/vacuum_gripper/set_pump")
#         try:
#             gripper = rospy.ServiceProxy("/turtlebot/swiftpro/vacuum_gripper/set_pump", SetBool)
#             x = gripper(True)
#             if x.success:
#                 rospy.loginfo("Pump turned on")
#             else:
#                 rospy.logwarn("Pump failed to turn on: %s", x.message)
#         except rospy.ServiceException as e:
#             rospy.logerr("Service call failed: %s", e)
#     def gripper_off(self):
#         rospy.wait_for_service("/turtlebot/swiftpro/vacuum_gripper/set_pump")
#         try:
#             gripper = rospy.ServiceProxy("/turtlebot/swiftpro/vacuum_gripper/set_pump", SetBool)
#             x = gripper(False)
#             if x.success:
#                 rospy.loginfo("Pump turned off")
#             else:
#                 rospy.logwarn("Pump failed to turn off: %s", x.message)
#         except rospy.ServiceException as e:
#             rospy.logerr("Service call failed: %s", e)
#     def move(self,dq):
#         # publish the manipulator velocities
#         self.pub_joints.publish(Float64MultiArray(data=[float(dq[2,0]),float(dq[3,0]),float(dq[4,0]),float(dq[5,0])]))
#         # Publish wheel velocities
#         v = dq[0,0]
#         w = dq[1,0]
#         left_lin_vel = v - (w * 0.230  / 2.0)
#         right_lin_vel = v + (w * 0.230  / 2.0)
#         left_wheel_vel = left_lin_vel / 0.035
#         right_wheel_vel = right_lin_vel / 0.035
#         self.pub_base.publish(Float64MultiArray(data = [right_wheel_vel, left_wheel_vel]))

#     def control(self, joint_states):
#         if self.seq == 1:
#             self.weights = np.diag([500, 500, 1, 1, 1, 1])
#             self.tasks = [Position3D("End-effector position", self.sigma_d)]
#         elif self.seq == 2:
#             self.weights = np.diag([5000, 5000, 1, 1, 1, 1])
#             self.tasks = [
#                 JointPosition("Joint position 3",np.array([0]).reshape(1,1), link = 4),
#                 JointPosition("Joint position 1",np.array([np.pi/2]).reshape(1,1), link = 2)] 
#         elif self.seq == 3:
#             self.weights = np.diag([5000, 5000, 1, 1, 1, 1])
#             self.tasks = [
#                 JointPosition("Joint position 3",np.array([0]).reshape(1,1), link = 4),
#                 JointPosition("Joint position 2",np.array([0.04]).reshape(1,1), link = 3),

#                 ] 
#         elif self.seq == 4:
#             pass
#         elif self.seq == 5:
#             self.weights = np.diag([1, 1, 1, 1, 1, 1])
#             self.sigma_d =  np.array([0, 0, -0.33]).reshape(3,1)
#             self.tasks = [
#                 JointPosition("Joint position 1",np.array([np.pi/2]).reshape(1,1), link = 2),

#                 Position3D("End-effector position", np.array([2,2, -0.2]).reshape(3,1))
#                  ] 

#         elif self.seq == 6:

#             self.weights = np.diag([2000, 2000, 1, 1, 1, 1])
#             self.tasks = [Position3D("End-effector position", np.array([2, 2, -0.2]).reshape(3,1))] #################

#         elif self.seq == 7:
#             self.weights = np.diag([5000, 5000, 1, 1, 1, 1])
#             self.tasks = [
#                 JointPosition("Joint position 3",np.array([0]).reshape(1,1), link = 4),
#                 JointPosition("Joint position 2",np.array([0.05]).reshape(1,1), link = 3),
#                 JointPosition("Joint position 1",np.array([-np.pi/2]).reshape(1,1), link = 2)] 




#         required = ['turtlebot/swiftpro/joint1','turtlebot/swiftpro/joint2','turtlebot/swiftpro/joint3','turtlebot/swiftpro/joint4']
#         if not all(name in joint_states.name for name in required):
#             return   
#         if self.sigma_d is None:
#             self.pub_joints.publish(Float64MultiArray(data=[0,0,0,0]))
#             self.pub_base.publish(Float64MultiArray(data=[0.0,0.0]))
#             return
#         pos_map = dict(zip(joint_states.name, joint_states.position))
#         q = np.array([pos_map['turtlebot/swiftpro/joint1'],pos_map['turtlebot/swiftpro/joint2'],pos_map['turtlebot/swiftpro/joint3'],pos_map['turtlebot/swiftpro/joint4']])
#         self.goalpose_pub(self.sigma_d)

#         T = kinematics(q,self.robot_pose)
#         ee = T[0:3,3]
#         R = T[0:3,0:3]
        
#         # 4) Task‐priority loop
#         dq = np.zeros((self.dof,1))
#         P  = np.eye(self.dof)
#         for task in self.tasks:
#             task.update(q, self.robot_pose)
#             if task.isActive() != 0:
#                 A = task.getJacobian() @ P
#                 dq += DLS(A, self.damping, self.weights) @ (task.getError() - A @ dq)
#                 P  -= np.linalg.pinv(A) @ A    
#         err = self.tasks[-1].getError()  # 3×1: [ex, ey, ez]
#         self.q1.append(dq[0])
#         self.q2.append(dq[1])
#         self.q3.append(dq[2])
#         self.q4.append(dq[3])
#         self.q5.append(dq[4])
#         self.q6.append(dq[5])
#         self.EE_x.append(ee[0])
#         self.EE_y.append(ee[1])
#         self.mb_x.append(self.robot_pose[0])
#         self.mb_y.append(self.robot_pose[1])
#         if self.time == []:
#             self.time.append(0)
#         else:
#             self.time.append(self.time[-1] + 1.0 / 60.0)


#         if self.seq == 0:
#             self.error.append(np.linalg.norm(err))
#             if self.time_err == []:
#                 self.time_err.append(0)
#             else:
#                 self.time_err.append(self.time_err[-1] + 1.0 / 60.0)

#             # Stop condition
#             if (abs(err[0,0]) < 0.01 and abs(err[1,0]) < 0.01):
#                 rospy.loginfo("Went to object) Task 0 done --> Task 1")
#                 self.seq = 1
#                 return

#             # Base yaw alignment
#             goal_x, goal_y = self.sigma_d[0,0], self.sigma_d[1,0]
#             robot_x, robot_y, robot_yaw = self.robot_pose
#             desired_yaw = np.arctan2(goal_y - robot_y, goal_x - robot_x)
#             yaw_error = desired_yaw - robot_yaw
#             yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
#             dq[1,0] = yaw_error * 1.5  # tune this gain

            
#             distance_error = np.linalg.norm([goal_x - robot_x, goal_y - robot_y])
#             if distance_error < 0.3:
#                 dq[0,0] *= 0.5

#             dq = dq / 2
#             self.move(dq)
#             dq = dq/2
#             self.move(dq)

#         elif  self.seq == 1:
#             self.error.append(np.linalg.norm(err))
#             if self.time_err == []:
#                 self.time_err.append(0)
#             else:
#                 self.time_err.append(self.time_err[-1] + 1.0 / 60.0)
#             if err[2,0]< 0.01  and abs(err[0,0]) < 0.01 and abs(err[1,0]) < 0.01:
#                 self.gripper_on()
#                 rospy.loginfo("(Grapped object) Task 1 done --> Task 2")
#                 self.seq =2        
#                 return
            
#             dq = dq/3   
#             self.move(dq)  
#         elif self.seq == 2:
#             if int(q[2]) == 0 and round(q[0],2) == 1.57:
#                 rospy.loginfo("(Object lifted) Task 2 done --> Task 3")
#                 self.seq = 3
#                 return
#             dq = dq/4
#             self.move(dq)
#         elif self.seq == 3:
#             if int(q[2]) == 0 and round(q[1],2) == 0.04:   #  if int(q[2]) == 0 and round(q[0],2) == -1.57 and round(q[1],2) == 0.05:
#                 rospy.loginfo("(Object on the back) Task 3 done --> Task 4")
#                 self.seq = 4
#                 return
#             dq = dq/4
#             self.move(dq)
#         elif self.seq == 4:
#             # rospy.loginfo("(Truning)")
#             # self.pub_base.publish(Float64MultiArray(data = [1, -1]))
#             # rospy.sleep(5)
#             # rospy.loginfo("Moving ))")
#             # self.pub_base.publish(Float64MultiArray(data = [1, 1]))
#             # rospy.sleep(5)
#             # rospy.loginfo("( Task 4 Robot reached location --> Task 5)")
#             # self.pub_base.publish(Float64MultiArray(data = [0, 0]))
#             self.seq = 5
#             return
        

#         elif self.seq == 5:
#             self.error.append(np.linalg.norm(err))
#             if self.time_err == []:
#                 self.time_err.append(0)
#             else:
#                 self.time_err.append(self.time_err[-1] + 1.0 / 60.0)
#             if abs(err[0,0]) < 0.3 and abs(err[1,0]) < 0.3:
#                 rospy.loginfo("(Task 5 Ready to Drop Task --> 6)")
#                 self.seq = 6
#                 self.pub_joints.publish(Float64MultiArray(data=[0,0,0,0]))
#                 self.pub_base.publish(Float64MultiArray(data = [0, 0]))
#                 return
#             dq = dq/5
#             self.move(dq)



#         elif self.seq == 6:
#             self.error.append(np.linalg.norm(err))
#             if self.time_err == []:
#                 self.time_err.append(0)
#             else:
#                 self.time_err.append(self.time_err[-1] + 1.0 / 60.0)
#             if err[2,0]< 0.03:###########################################
#                 rospy.loginfo("(Task 6 Object droped Task --> Task 7)")
#                 self.gripper_off()
#                 self.seq = 7
#                 return
#             dq = dq/5
#             self.move(dq)


#         elif self.seq == 7:
#             if int(q[2]) == 0 and round(q[0],2) == -1.57 and round(q[1],2) == 0.04:
#                 rospy.loginfo("(Home postion)")
#                 rospy.signal_shutdown("End")
#                 return
#             dq = dq/4
#             self.move(dq)


#     def get_odom(self,odom):
#         _, _, yaw = tf.transformations.euler_from_quaternion([odom.pose.pose.orientation.x, 
#                                                               odom.pose.pose.orientation.y,
#                                                               odom.pose.pose.orientation.z,
#                                                               odom.pose.pose.orientation.w])
#         self.robot_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, yaw])
#     def goalpose_pub(self, goal):
#         marker = Marker()
#         marker.header.frame_id = "world_ned"
#         marker.type = marker.SPHERE
#         marker.action = marker.ADD
#         marker.header.stamp = rospy.Time.now()
#         marker.pose.position.x = goal[0]
#         marker.pose.position.y = goal[1]
#         marker.pose.position.z = goal[2]
#         marker.scale.x = 0.05
#         marker.scale.y = 0.05
#         marker.scale.z = 0.05
#         marker.color.g = 0.1
#         marker.color.r = 0.5
#         marker.color.a = 1
#         self.goalPose_pub.publish(marker)
#     def EE_marker(self,T,q):
#         EE_pose = T[:, -1]
#         # print("position_EE", EE_pose)
#         EE = PoseStamped()
#         EE.header.stamp = rospy.Time.now()
#         EE.header.frame_id = self.EE_path_msg.header.frame_id 
#         EE.pose.position.x = EE_pose[0]
#         EE.pose.position.y = EE_pose[1]
#         EE.pose.position.z = EE_pose[2]
#         Q = quaternion_from_euler(0, 0, q[3])
#         EE.pose.orientation.x = Q[0]
#         EE.pose.orientation.y = Q[1]
#         EE.pose.orientation.z = Q[2]
#         EE.pose.orientation.w = Q[3]
#         self.EE_path_msg.header.stamp = EE.header.stamp
#         self.EE_path_msg.poses.append(EE)
#         self.EE_path_pub.publish(self.EE_path_msg)
# if __name__ == "__main__":
#     rospy.init_node('controller', anonymous=True)
#     TPC_node = TP_Control()
#     rospy.spin()

#     plt.figure()
#     plt.plot(TPC_node.time_err, TPC_node.error, label='e1 (end-effector position)')
#     plt.xlabel('Time Step(s)')
#     plt.ylabel('Error (m)')
#     plt.title('evolution of control errors over time')
#     plt.grid()
#     plt.legend()
#     plt.show()
    
#     plt.figure()
#     plt.plot(TPC_node.EE_x, TPC_node.EE_y, label='end-effector position')
#     plt.plot(TPC_node.mb_x, TPC_node.mb_y, label='mobile base position')
#     plt.xlabel('x-plane')
#     plt.ylabel('y-plane')
#     plt.title('motion of the mobile base and the end-effector in the X-Y plane')
#     plt.grid()
#     plt.legend()
#     plt.show()
    
#     plt.figure()
#     plt.plot(TPC_node.time, TPC_node.q1, label='q1')
#     plt.plot(TPC_node.time, TPC_node.q2, label='q2')
#     plt.plot(TPC_node.time, TPC_node.q3, label='q3')
#     plt.plot(TPC_node.time, TPC_node.q4, label='q4')
#     plt.plot(TPC_node.time, TPC_node.q5, label='q5')
#     plt.plot(TPC_node.time, TPC_node.q6, label='q6')
#     plt.xlabel('Time Step(s)')
#     plt.ylabel('Velocity(m/s)')
#     plt.title('evolution of joint velocities over time')
#     plt.grid()
#     plt.legend()
#     plt.show()
