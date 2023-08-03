from numpy.linalg import norm
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *
import qpsolvers
import math

GRAV = 9.81


class NominalSBC:
    class State:
        def __init__(self, position, velocity):
            self.position = position
            self.velocity = velocity

    class ObjectDescription:
        def __init__(self, state, radius, maximum_linf_acceleration_lower_bound):
            self.state = state
            self.radius = radius
            self.maximum_linf_acceleration_lower_bound = maximum_linf_acceleration_lower_bound

    def __init__(self, maximum_linf_acceleration, aggressiveness, radius):
        self.maximum_linf_acceleration = maximum_linf_acceleration
        self.aggressiveness = aggressiveness
        self.radius = radius

    def plan(self, self_state, object_descriptions, desired_acceleration):
        P = 2.0 * np.eye(3)
        q = -2.0 * desired_acceleration
        G = np.ndarray((0, 3), np.float64)
        h = np.array([])
        A = np.ndarray((0, 3), np.float64)
        b = np.array([])
        lb = np.array([-self.maximum_linf_acceleration]*3)
        ub = np.array([self.maximum_linf_acceleration]*3)

        for object_description in object_descriptions:
            relative_position = self_state.position - object_description.state.position
            relative_position_norm = np.linalg.norm(relative_position)
            if abs(relative_position_norm) < 1e-10:
                return None

            safety_distance = self.radius + object_description.radius

            if relative_position_norm < safety_distance:
                return None

            relative_velocity = self_state.velocity - object_description.state.velocity
            relative_position_dot_relative_velocity = np.dot(relative_position, relative_velocity)

            hij = math.sqrt(2.0 * (self.maximum_linf_acceleration + object_description.maximum_linf_acceleration_lower_bound) * (
                relative_position_norm - safety_distance)) + (relative_position_dot_relative_velocity / relative_position_norm)

            bij = -relative_position_dot_relative_velocity * np.dot(relative_position, self_state.velocity) / \
                (relative_position_norm * relative_position_norm) + np.dot(relative_velocity, self_state.velocity) + \
                (self.maximum_linf_acceleration / (self.maximum_linf_acceleration +
                 object_description.maximum_linf_acceleration_lower_bound)) * \
                (self.aggressiveness * hij * hij * hij * relative_position_norm
                 + (math.sqrt(self.maximum_linf_acceleration +
                              object_description.maximum_linf_acceleration_lower_bound) *
                    relative_position_dot_relative_velocity) /
                 (math.sqrt(2.0 * (relative_position_norm - safety_distance))))

            G = np.append(G, [-1.0 * relative_position], axis=0)
            h = np.append(h, bij)

        x = qpsolvers.solve_qp(P, q, G, h, A, b, lb, ub, solver="osqp")

        return x

class RawControl(object):
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        self.action = None
        self.step_func = self.step
        self.high = np.ones(4)
        self.low = -np.ones(4)
        self.bias = 1.0
        self.scale = 0.5

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(4)
            self.bias = 0.0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(4)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(4)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()

    # @profile
    def step_tf(self, dynamics, action, goal, dt, observation=None):
        # print('bias/scale: ', self.scale, self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()


class VerticalControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.step = self.step1D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(1)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(1)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(1)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]] * 4), dt)

    # modifies the dynamics in place.
    # @profile
    def step1D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]]), dt)


class VertPlaneControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '2D':
            self.step = self.step2D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(2)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(2)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(2)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0], action[0], action[1], action[1]]), dt)

    # modifies the dynamics in place.
    # @profile
    def step2D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array(action), dt)


# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2, :] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1, 4))
    dw = (1.0 / dynamics.inertia)[:, None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    if J_cond > 50:
        print("WARN: Jacobian conditioning is high: ", J_cond)
    return J


# P-only linear controller on angular velocity.
# direct (ignoring motor lag) control of thrust magnitude.
class OmegaThrustControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        kp = 5.0  # could be more aggressive
        omega_err = dynamics.omega - action[1:]
        dw_des = -kp * omega_err
        acc_des = GRAV * (action[0] + 1.0)
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


# TODO: this has not been tested well yet.
class VelocityYawControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        vmax = 20.0  # meters / sec
        dymax = 4 * np.pi  # radians / sec
        high = np.array([vmax, vmax, vmax, dymax])
        return spaces.Box(-high, high, dtype=np.float32)

    # @profile
    def step(self, dynamics, action, dt):
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        kp_v = 5.0
        kp_a, kd_a = 100.0, 50.0

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, R[:, 0]))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        omega_des = np.array([0, 0, action[3]])
        e_w = dynamics.omega - omega_des

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        # thrust_mag = np.dot(acc_des, dynamics.rot[:,2])
        thrust_mag = get_blas_funcs("thrust_mag", [acc_des, dynamics.rot[:, 2]])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        dynamics.step(thrusts, dt)


# this is an "oracle" policy to drive the quadrotor towards a goal
# using the controller from Mellinger et al. 2011
class MellingerController(object):
    # @profile
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        ## Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        self.action = None

        self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0

        self.rot_des = np.eye(3)

        self.enable_sbc = True
        self.sbc = NominalSBC(maximum_linf_acceleration=5.0,
                              aggressiveness=0.1, radius=0.15)
        self.sbc_last_safe_acc = None

        self.step_func = self.step

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, acc_des, dt, observation=None):
        # to_goal = goal - dynamics.pos
        # e_p = -clamp_norm(to_goal, 4.0)
        # e_v = dynamics.vel
        # acc_des = -self.kp_p * e_p - self.kd_p * e_v

        if self.enable_sbc and observation is not None:
            new_acc = self.sbc.plan(observation["self_state"], observation["neighbor_descriptions"], acc_des)

            if new_acc is not None:
                self.sbc_last_safe_acc = new_acc
                acc_des = new_acc
            else:
                if self.sbc_last_safe_acc is not None:
                    acc_des = self.sbc_last_safe_acc

        # Question: Why do we need to do this???
        acc_des += np.array([0, 0, GRAV])
        xc_des = self.rot_des[:, 0]

        # see Mellinger and Kumar 2011
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2  # slow down yaw dynamics
        e_w = dynamics.omega

        dw_des = -self.kp_a * e_R - self.kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, R[:, 2])

        des = np.append(thrust_mag, dw_des)

        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()
        return self.action, new_acc

    # def action_space(self, dynamics):
    #     circle_per_sec = 2 * np.pi
    #     max_rp = 5 * circle_per_sec
    #     max_yaw = 1 * circle_per_sec
    #     min_g = -1.0
    #     max_g = dynamics.thrust_to_weight - 1.0
    #     low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
    #     high = np.array([max_g, max_rp, max_rp, max_yaw])
    #     return spaces.Box(low, high, dtype=np.float32)
