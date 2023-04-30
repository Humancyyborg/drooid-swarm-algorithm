import numpy as np
from scipy.spatial.transform import Rotation

vehicle_params = {
    "Rotor0Angle": 45.0 * np.pi / 180.0,
    "Rotor1Angle": 135.0 * np.pi / 180.0,
    "Rotor2Angle": 225.0 * np.pi / 180.0,
    "Rotor3Angle": 315.0 * np.pi / 180.0,
    "Rotor0Direction": 1,
    "Rotor1Direction": -1,
    "Rotor2Direction": 1,
    "Rotor3Direction": -1,
    "Mass": 0.028,
    "ArmLength": 0.045962,
    "RotorForceConstant": 0.009,  # 3.16e-3
    "RotorMomentConstant": 0.05 # 1.41e-1
}


class CommandTrajectory:
    def __init__(self, dynamics):
        self.yaw = np.zeros(dynamics.rot.shape)
        self.velocity_W = np.zeros(dynamics.vel.shape)
        self.acceleration_W = np.array([0.0, 0.0, 0.0])
        self.position_W = np.zeros(dynamics.pos.shape)
        self.yaw_rate = np.zeros(dynamics.omega.shape)

    def update_command(self, goal):
        self.position_W = goal

class ControllerParams:
    def __init__(self, position_gain=np.array([0.1, 0.1, 0.1]), velocity_gain=np.array([0.2, 0.2, 0.3]),
                 attitude_gain=np.array([.07, .07, 0.0035]), angular_rate_gain=np.array([0.01, 0.01, 0.0025])):
        self.position_gain = position_gain/1000
        self.velocity_gain = velocity_gain/1000
        self.attitude_gain = attitude_gain/1000
        self.angular_rate_gain = angular_rate_gain/1000

class LeeController:
    def __init__(self, dynamics):
        self.vehicle_params = dynamics
        self.controller_active = False
        self.controller = ControllerParams()
        self.command_trajectory = CommandTrajectory(dynamics)
        self.allocation_matrix = np.zeros((4, 4))
        self.inertia = np.zeros((3, 3))
        self.gravity = 9.81
        self.inertia[0, 0] = self.vehicle_params.inertia[0]
        self.inertia[1, 1] = self.vehicle_params.inertia[1]
        self.inertia[2, 2] = self.vehicle_params.inertia[2]
        self.calculateAllocationMatrix()
        self.normalized_attitude_gain = np.dot(self.controller.attitude_gain.T, np.linalg.inv(self.inertia))
        self.normalized_angular_rate_gain = np.dot(self.controller.angular_rate_gain.T, np.linalg.inv(self.inertia))
        self.I = np.zeros((4, 4))
        for i in range(3):
            for j in range(3):
                self.I[i, j] = self.inertia[i, j]
        self.I[3, 3] = 1
        self.angular_acc_to_rotor_velocities_ = np.dot(np.linalg.pinv(self.allocation_matrix), self.I)
        print()

    def calculateAllocationMatrix(self):
        for i in range(4):
            self.allocation_matrix[0, i] = np.sin(vehicle_params["Rotor"+str(i)+"Angle"]) * vehicle_params["ArmLength"] * vehicle_params["RotorForceConstant"]
            self.allocation_matrix[1, i] = -np.cos(vehicle_params["Rotor"+str(i)+"Angle"]) * vehicle_params["ArmLength"] * vehicle_params["RotorForceConstant"]
            self.allocation_matrix[2, i] = -vehicle_params["Rotor"+str(i)+"Direction"] * vehicle_params["RotorForceConstant"] * vehicle_params["RotorMomentConstant"]
            self.allocation_matrix[3, i] = vehicle_params["RotorForceConstant"]

    def set_command_trajectory(self, goal):
        self.command_trajectory.update_command(goal)
        self.controller_active = True

    def update_odometry(self, dynamics):
        self.vehicle_params = dynamics

    def calculate_rotor_velocities(self):
        self.rotor_velocities = np.zeros(4)
        if not self.controller_active:
            return

        self.compute_desired_acceleration()
        self.compute_desired_angular_acceleration()

        thrust = -self.vehicle_params.mass * np.dot(self.acceleration, self.vehicle_params.rot[:, 2])

        angular_acceleration_thrust = np.zeros((4,))
        for i in range(3):
            angular_acceleration_thrust[i] = self.angular_acceleration[i]
        angular_acceleration_thrust[3] = thrust

        self.rotor_velocities = np.dot(self.angular_acc_to_rotor_velocities_, angular_acceleration_thrust)
        self.rotor_velocities = np.maximum(self.rotor_velocities, np.zeros((self.rotor_velocities.shape[0],)))
        if np.all(self.rotor_velocities == 0):
            print()
        #self.rotor_velocities = np.clip(self.rotor_velocities, 0, 1)
        return self.rotor_velocities

    def compute_desired_acceleration(self):
        position_error = self.vehicle_params.pos - self.command_trajectory.position_W
        R_W_I = self.vehicle_params.rot
        velocity_W = np.dot(R_W_I, self.vehicle_params.vel)
        velocity_error = velocity_W - self.command_trajectory.velocity_W

        e3 = np.array([0, 0, 1])

        self.acceleration = (np.multiply(position_error, self.controller.position_gain) + np.multiply(velocity_error, self.controller.velocity_gain)) / self.vehicle_params.mass - self.gravity*e3 - self.command_trajectory.acceleration_W

    def compute_desired_angular_acceleration(self):
        R = self.vehicle_params.rot
        quat = Rotation.from_matrix(R).as_quat()
        yaw = np.math.atan2(2.0*(quat[3]*quat[2]+quat[0]*quat[1]), 1.0 - 2.0*(quat[1]*quat[1] + quat[2]*quat[2]))

        b1_des = np.array([np.cos(yaw), np.sin(yaw), 0])
        b3_des = -self.acceleration / np.linalg.norm(self.acceleration)
        b2_des = np.cross(b3_des, b1_des)
        b2_des = b2_des / np.linalg.norm(b2_des)

        R_des = np.zeros((3, 3))
        R_des[:, 0] = np.cross(b2_des, b3_des)
        R_des[:, 1] = b2_des
        R_des[:, 2] = b3_des

        angle_error_matrix = 0.5 * (np.dot(R_des.T, R) - np.dot(R.T, R_des))
        self.angle_error = np.array([angle_error_matrix[2, 1], angle_error_matrix[0, 2], angle_error_matrix[1, 0]])

        angular_rate_des = np.array([0, 0, self.command_trajectory.yaw_rate[2]])

        self.angular_rate_error = self.vehicle_params.omega - np.dot(np.dot(R_des.T, R), angular_rate_des)

        self.angular_acceleration = -1 * np.multiply(self.angle_error, self.normalized_attitude_gain) - np.multiply(self.angular_rate_error, self.normalized_angular_rate_gain) + np.cross(self.vehicle_params.omega, self.vehicle_params.omega)