#!/usr/bin/env python3

import numpy as np
from tf.transformations import euler_from_matrix

def forward_kinematics(joint_angles, robot_base_pose):
    """
    Calculate forward kinematics for Swiftpro manipulator
    """
    theta1, theta2, theta3, _ = joint_angles

    radial_distance = 0.0132 - 0.142*np.sin(theta2) + 0.1588*np.cos(theta3) + 0.0565
    x_position = np.cos(theta1)*radial_distance
    y_position = np.sin(theta1)*radial_distance
    z_position = -0.108 -0.142*np.cos(theta2) - 0.1588*np.sin(theta3) + 0.0722

    # Transformation from manipulator base to end-effector
    manipulator_to_ee = np.array([
        [1, 0, 0, x_position],
        [0, 1, 0, y_position],
        [0, 0, 1, z_position],
        [0, 0, 0, 1]
    ])
    
    # Transformation from robot base to manipulator base
    base_to_manipulator = np.array([
        [0,1,0,0.051],
        [-1,0,0,0],
        [0,0,1, -0.198],
        [0,0,0,1]
    ])

    # Transformation from world frame to robot base 
    base_yaw = float(robot_base_pose[2])
    world_to_base = np.array([
        [np.cos(base_yaw), -np.sin(base_yaw), 0, float(robot_base_pose[0])],
        [np.sin(base_yaw),  np.cos(base_yaw), 0, float(robot_base_pose[1])],
        [0,            0,           1, 0],
        [0,            0,           0, 1]
    ])

    world_to_manipulator = world_to_base @ base_to_manipulator
    world_to_endeffector = world_to_manipulator @ manipulator_to_ee
    return world_to_endeffector


def compute_jacobian(joint_angles, robot_base_pose):
    """
    Calculate the geometric Jacobian matrix for the mobile manipulator
    """
    sin_theta = [np.sin(joint_angles[0]), np.sin(joint_angles[1]), np.sin(joint_angles[2]), np.sin(robot_base_pose[2])]
    cos_theta = [np.cos(joint_angles[0]), np.cos(joint_angles[1]), np.cos(joint_angles[2]), np.cos(robot_base_pose[2])]
    
    # Partial derivatives with respect to joint 1
    dx_dtheta1 = -0.0565 * sin_theta[0] * sin_theta[3] + 0.0565 * cos_theta[0] * (cos_theta[3] - sin_theta[3])
    # Partial derivatives with respect to joint 2
    dx_dtheta2 = -0.142 * cos_theta[1] * cos_theta[0] * sin_theta[3] - 0.142 * cos_theta[1] * sin_theta[0] * (cos_theta[3] - sin_theta[3])
    # Partial derivatives with respect to joint 3
    dx_dtheta3 = -0.1588 * sin_theta[2] * cos_theta[0] * sin_theta[3] - 0.1588 * sin_theta[2] * sin_theta[0] * (cos_theta[3] - sin_theta[3])
    # Partial derivatives with respect to base rotation
    dx_dbase_rot = ((0.0132 - 0.142 * sin_theta[1] + 0.1588 * cos_theta[2] + 0.0565) * cos_theta[0]) * cos_theta[3] - 0.051 * sin_theta[3] - robot_base_pose[0] * sin_theta[3] + ((0.0132 - 0.142 * sin_theta[1] + 0.1588 * cos_theta[2]+0.0565)*sin_theta[0])*(-sin_theta[3]-cos_theta[3])
    
    dy_dtheta1 = ((0.0132-0.142 * sin_theta[1] + 0.1588 * cos_theta[2] + 0.0565) * cos_theta[0]) * (cos_theta[3] + sin_theta[3]) + ((0.0132 - 0.142 * sin_theta[1] + 0.1588 * cos_theta[2] + 0.0565) * cos_theta[0]) * sin_theta[3]
    dy_dtheta2 = -0.142 * cos_theta[1] * sin_theta[0] * (cos_theta[3] + sin_theta[3]) + 0.142 * cos_theta[1] * cos_theta[0] * cos_theta[3]
    dy_dtheta3 = -0.1588 * sin_theta[2] * sin_theta[0] * (cos_theta[3] + sin_theta[3]) + 0.1588 * sin_theta[2] * cos_theta[0] * cos_theta[3]
    dy_dbase_rot = 0.051 * cos_theta[3]+ robot_base_pose[0] * cos_theta[3] + ((0.0132 - 0.142 * sin_theta[1] + 0.1588 * cos_theta[2] + 0.0565) * sin_theta[0]) * (cos_theta[3] - sin_theta[3]) + ((0.0132 - 0.142 * sin_theta[1]+0.1588*cos_theta[2]+0.0565)*cos_theta[0])*(sin_theta[3])
    
    jacobian_matrix = np.array([
        [cos_theta[3], dx_dbase_rot, dx_dtheta1, dx_dtheta2, dx_dtheta3, 0],
        [sin_theta[3], dy_dbase_rot, dy_dtheta1, dy_dtheta2, dy_dtheta3, 0],
        [0, 0, 0, (0.142 * sin_theta[1]), (-0.1588 * cos_theta[2]), 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1]])
    return jacobian_matrix


def weighted_DLS(jacobian_matrix, damping_param, weight_matrix=None):
    """
    Weighted Damped Least Squares pseudoinverse calculation:
      jacobian_matrix: m×n Jacobian matrix
      weight_matrix: n×n weight matrix (defaults to identity)
    """
    if weight_matrix is None:
        weight_matrix = np.eye(jacobian_matrix.shape[1])
    inverse_weights = np.linalg.inv(weight_matrix)
    damped_matrix = jacobian_matrix @ inverse_weights @ jacobian_matrix.T + (damping_param**2)*np.eye(jacobian_matrix.shape[0])
    return inverse_weights @ jacobian_matrix.T @ np.linalg.inv(damped_matrix)


# --- Base Task class and specific task implementations ---
class ControlTask:
    def __init__(self, task_name, target_value):
        self.task_name = task_name
        self.target_value = target_value
        self.task_jacobian = None
        self.task_error = None

    def update(self, joint_config, base_pose):
        raise NotImplementedError

    def getJacobian(self):
        return self.task_jacobian

    def getError(self):
        return self.task_error

    def isActive(self):
        return True


class PositionTask3D(ControlTask):
    def __init__(self, task_name, target_position):
        super().__init__(task_name, target_position)
        self.task_error = np.zeros((3,1))

    def update(self, joint_config, base_pose):
        full_jacobian = compute_jacobian(joint_config, base_pose)        # 6×6
        self.task_jacobian = full_jacobian[0:3, :]     # Extract position rows only
        current_xyz = forward_kinematics(joint_config, base_pose)[0:3,3].reshape(3,1)
        self.task_error = self.target_value - current_xyz


class OrientationTask3D(ControlTask):
    def __init__(self, task_name, target_orientation):
        super().__init__(task_name, target_orientation)
        self.task_error = np.zeros((3,1))

    def update(self, joint_config, base_pose):
        full_jacobian = compute_jacobian(joint_config, base_pose)        # 6×6
        self.task_jacobian = full_jacobian[3:6, :]     # Extract orientation rows only

        rotation_matrix = forward_kinematics(joint_config, base_pose)[0:3,0:3]
        roll_angle, pitch_angle, yaw_angle = euler_from_matrix(rotation_matrix, 'sxyz')
        current_angles = np.array([roll_angle, pitch_angle, yaw_angle]).reshape(3,1)
        self.task_error = self.target_value - current_angles


class PoseTask3D(ControlTask):
    """
    Combined task for simultaneous control of 3D position and 3D orientation
    of the end-effector as a single 6×1 error vector.
    """
    def __init__(self, task_name, target_pose):
        super().__init__(task_name, target_pose)
        # 6×1 error vector (x,y,z, roll, pitch, yaw)
        self.task_error = np.zeros((6,1))
        # 6×6 Jacobian matrix (complete geometric Jacobian)
        self.task_jacobian = np.zeros((6,6))

    def update(self, joint_config, base_pose):
        # Complete 6×6 Jacobian
        full_jacobian = compute_jacobian(joint_config, base_pose)
        self.task_jacobian = full_jacobian
        # Current end-effector pose
        transformation_matrix = forward_kinematics(joint_config, base_pose)
        x_pos, y_pos, z_pos = transformation_matrix[0:3,3]
        # Extract euler angles
        rotation_matrix = transformation_matrix[0:3,0:3]
        roll_val, pitch_val, yaw_val = euler_from_matrix(rotation_matrix, 'sxyz')
        current_pose = np.array([x_pos, y_pos, z_pos, roll_val, pitch_val, yaw_val]).reshape(6,1)
        # 6×1 error vector
        self.task_error = self.target_value - current_pose


class JointAngleTask(ControlTask):
    """
    Task for controlling a single joint to reach a desired angle.
    """
    def __init__(self, task_name, target_angle, link):
        super().__init__(task_name, target_angle)
        self.joint_link = link
        self.task_error = np.zeros((1,1))
        self.task_jacobian = np.zeros((1,6))

    def update(self, joint_config, base_pose):
        # Jacobian has unit value only for the target joint
        self.task_jacobian[:] = 0.0
        self.task_jacobian[0, self.joint_link] = 1.0
        # Current joint angle
        current_angle = float(joint_config[self.joint_link-2])
        self.task_error = self.target_value - np.array([[current_angle]])


class JointBoundaryTask(ControlTask):
    """
    Soft joint limit avoidance using hysteresis mechanism.
    target_value = [activation_threshold, deactivation_delta]
    joint_limits = [minimum_limit, maximum_limit]
    """
    def __init__(self, task_name, target_value, joint_limits, link):
        super().__init__(task_name, target_value)
        self.joint_link = link
        self.min_limit = joint_limits[0]
        self.max_limit = joint_limits[1]
        # 1×1 error indicator: +1 push up, -1 push down, 0 inactive
        self.task_error = np.zeros((1,1))
        self.task_jacobian = np.zeros((1,6))
        self.hysteresis_state = 0   # Current hysteresis state
        self.gain_matrix = np.eye(1)
        # Extract threshold parameters
        self.activation_threshold = target_value[0]
        self.deactivation_offset = target_value[1]
    
    def set_gain_matrix(self, gain_matrix):
        """Set custom 1×1 gain matrix."""
        self.gain_matrix = gain_matrix

    def get_gain_matrix(self):
        return self.gain_matrix

    def update(self, joint_config, base_pose):
        # Simple joint-space Jacobian
        self.task_jacobian[:] = 0.0
        self.task_jacobian[0, self.joint_link] = 1.0
        current_angle = float(joint_config[self.joint_link-2])
        
        # Hysteresis logic for soft limits
        if self.hysteresis_state == 0 and current_angle >= (self.max_limit - self.activation_threshold):
            self.hysteresis_state = -1
        elif self.hysteresis_state == 0 and current_angle <= (self.min_limit + self.activation_threshold):
            self.hysteresis_state = +1
        elif self.hysteresis_state == -1 and current_angle <= (self.max_limit - self.deactivation_offset):
            self.hysteresis_state = 0
        elif self.hysteresis_state == +1 and current_angle >= (self.min_limit + self.deactivation_offset):
            self.hysteresis_state = 0
        
        # Error represents push direction
        self.task_error = np.array([[self.hysteresis_state]])

    def isActive(self):
        return self.hysteresis_state != 0
    

class BaseOrientationTask(ControlTask):
    def __init__(self, task_name, target_orientation):
        super().__init__(task_name, target_orientation)
        self.task_error = np.zeros((1,1))                       

    def update(self, joint_config, base_pose):
        self.task_jacobian = np.zeros((1,6))                        
        self.task_jacobian[0, 0] = 1   
        self.task_error = np.array([self.target_value - base_pose[2]])
